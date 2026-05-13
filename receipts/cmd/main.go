/*
receipts — The "Look at this!!" engine.

Takes a canonical concept from your corpus and generates a plain-English
multi-tradition synthesis document with receipts: quotes from wildly different
sources/eras all saying the same thing, a 5th-grade explanation, and the
Hardware Glitch where it gets weird.

Pipeline:
 1. Embed the query concept via OpenRouter (Gemini)
 2. Search meta_reflections for top matches across distinct source files
 3. Search misfit_reports for Hardware Glitch receipts on those same chunks
 4. Send everything to DeepSeek R1 to write the synthesis doc
 5. Write output as Markdown to ./output/<slug>.md

Usage:

	receipts --concept "belief creates reality"
	receipts --concept "consciousness survives death" --sources 6 --out ./output
	receipts --concept "non-linear time" --model deepseek-chat   (fast, no R1)
	receipts --list-concepts                                      (show top canonical concepts)

Environment (.env or shell):

	QDRANT_URL            default http://localhost:6333
	OPENROUTER_API_KEY    required for embedding + synthesis
	DEEPSEEK_API_KEY      required if DEEPSEEK_CHAT_URL points to DeepSeek directly
	DEEPSEEK_CHAT_URL     default https://api.deepseek.com/v1/chat/completions
	DEEPSEEK_MODEL        default deepseek-reasoner
	EMBED_MODEL           default google/gemini-embedding-001
	REFLECTIONS_COLLECTION  default meta_reflections
	REPORTS_COLLECTION      default misfit_reports
*/
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/joho/godotenv"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ── version ──────────────────────────────────────────────────────────────────

var version = "dev"

// ── config ────────────────────────────────────────────────────────────────────

type config struct {
	qdrantHost            string
	qdrantPort            int
	openrouterKey         string
	deepseekKey           string
	deepseekURL           string
	deepseekModel         string
	embedModel            string
	reflectionsCollection string
	reportsCollection     string
	concept               string
	numSources            int
	outDir                string
	listConcepts          bool
	topConcepts           int
}

func loadConfig() config {
	_ = godotenv.Load(".env")

	cfg := config{
		qdrantHost:            envOr("QDRANT_HOST", "localhost"),
		qdrantPort:            6334,
		openrouterKey:         os.Getenv("OPENROUTER_API_KEY"),
		deepseekKey:           os.Getenv("DEEPSEEK_API_KEY"),
		deepseekURL:           envOr("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/v1/chat/completions"),
		deepseekModel:         envOr("DEEPSEEK_MODEL", "deepseek-reasoner"),
		embedModel:            envOr("EMBED_MODEL", "google/gemini-embedding-001"),
		reflectionsCollection: envOr("REFLECTIONS_COLLECTION", "meta_reflections"),
		reportsCollection:     envOr("REPORTS_COLLECTION", "misfit_reports"),
		numSources:            envIntOr("SOURCE_TRADITIONS_COUNT", 6),
	}

	flag.StringVar(&cfg.concept, "concept", "", "Concept to generate receipts for (e.g. 'belief creates reality')")
	flag.IntVar(&cfg.numSources, "sources", cfg.numSources, "Number of distinct source traditions to include")
	flag.IntVar(&cfg.numSources, "source-traditions", cfg.numSources, "Alias for --sources")
	flag.StringVar(&cfg.outDir, "out", "./output", "Output directory for generated Markdown files")
	flag.StringVar(&cfg.deepseekModel, "model", cfg.deepseekModel, "DeepSeek model (deepseek-reasoner or deepseek-chat)")
	flag.BoolVar(&cfg.listConcepts, "list-concepts", false, "List top canonical concepts from the corpus and exit")
	flag.IntVar(&cfg.topConcepts, "top", 20, "How many concepts to show with --list-concepts")
	flag.StringVar(&cfg.reflectionsCollection, "collection", cfg.reflectionsCollection, "Qdrant reflections collection")
	flag.BoolVar(new(bool), "version", false, "Print version and exit") // handled below
	flag.Parse()

	for _, arg := range os.Args[1:] {
		if arg == "--version" || arg == "-version" {
			fmt.Println("receipts", version)
			os.Exit(0)
		}
	}

	if cfg.numSources < 1 {
		fmt.Fprintln(os.Stderr, "error: --sources / --source-traditions must be >= 1")
		os.Exit(1)
	}

	return cfg
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOr(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

// ── Qdrant helpers ────────────────────────────────────────────────────────────

func qdrantConn(host string, port int) (*grpc.ClientConn, error) {
	addr := fmt.Sprintf("%s:%d", host, port)
	return grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(256*1024*1024)),
	)
}

func detectEmbedModel(ctx context.Context, conn *grpc.ClientConn, collection, vectorName, fallback string) (string, uint64, error) {
	client := qdrant.NewCollectionsClient(conn)
	resp, err := client.Get(ctx, &qdrant.GetCollectionInfoRequest{CollectionName: collection})
	if err != nil {
		return fallback, 0, err
	}

	vectors := resp.GetResult().GetConfig().GetParams().GetVectorsConfig()
	size := detectVectorSize(vectors, vectorName)
	if size == 0 {
		return fallback, 0, nil
	}
	if model := modelForVectorSize(size); model != "" {
		return model, size, nil
	}
	return fallback, size, nil
}

func detectVectorSize(vectors *qdrant.VectorsConfig, vectorName string) uint64 {
	if vectors == nil {
		return 0
	}

	if params := vectors.GetParams(); params != nil {
		return params.GetSize()
	}

	paramsMap := vectors.GetParamsMap()
	if paramsMap == nil {
		return 0
	}
	if params := paramsMap.GetMap()[vectorName]; params != nil {
		return params.GetSize()
	}
	if params := paramsMap.GetMap()[""]; params != nil {
		return params.GetSize()
	}

	for _, params := range paramsMap.GetMap() {
		if params != nil {
			return params.GetSize()
		}
	}
	return 0
}

func modelForVectorSize(size uint64) string {
	switch size {
	case 3072:
		return "google/gemini-embedding-001"
	case 1536:
		return "openai/text-embedding-3-small"
	default:
		return ""
	}
}

var vectorDimErrorRE = regexp.MustCompile(`(?i)expected\s+dim:\s*(\d+)\s*,\s*got:?\s*(\d+)`)

func expectedVectorDimFromErr(err error) (uint64, bool) {
	if err == nil {
		return 0, false
	}
	m := vectorDimErrorRE.FindStringSubmatch(strings.TrimSpace(err.Error()))
	if len(m) < 2 {
		return 0, false
	}
	dim, parseErr := strconv.ParseUint(m[1], 10, 64)
	if parseErr != nil {
		return 0, false
	}
	return dim, true
}

// ── OpenRouter embedding ──────────────────────────────────────────────────────

func embed(ctx context.Context, text, model, apiKey string) ([]float32, error) {
	body, _ := json.Marshal(map[string]any{
		"model": model,
		"input": text,
	})
	req, err := http.NewRequestWithContext(ctx, "POST",
		"https://openrouter.ai/api/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed %d: %s", resp.StatusCode, b)
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 {
		return nil, fmt.Errorf("embed: empty response")
	}
	return out.Data[0].Embedding, nil
}

// ── Qdrant search ─────────────────────────────────────────────────────────────

type reflection struct {
	ID         string
	SourceID   string
	SourceFile string
	Summary    string
	Claims     []string
	Concepts   []string
	Echoes     []string
	Tone       string
	Score      float32
}

type glitch struct {
	SourceFile string
	Report     string
	Verdict    string
	Score      float32
}

func searchReflections(
	ctx context.Context,
	conn *grpc.ClientConn,
	collection string,
	vec []float32,
	limit uint64,
) ([]reflection, error) {
	client := qdrant.NewPointsClient(conn)

	f32 := make([]float32, len(vec))
	copy(f32, vec)

	resp, err := client.Search(ctx, &qdrant.SearchPoints{
		CollectionName: collection,
		Vector:         f32,
		Limit:          limit * 6, // oversample to get diversity
		WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
		VectorName:     strPtr("claims_vec"),
	})
	if err != nil {
		return nil, fmt.Errorf("search reflections: %w", err)
	}

	// deduplicate by source ID — keep best score per source book
	best := map[string]reflection{}
	for _, pt := range resp.Result {
		pl := pt.Payload
		srcID := stringVal(pl, "source_id")
		srcFile := stringVal(pl, "source_file")
		if srcFile == "" || srcFile == "General Content" {
			continue
		}
		dedupeKey := srcID
		if dedupeKey == "" {
			dedupeKey = srcFile
		}
		r := reflection{
			ID:         fmt.Sprintf("%v", pt.Id),
			SourceID:   srcID,
			SourceFile: srcFile,
			Summary:    stringVal(pl, "summary"),
			Claims:     stringSliceVal(pl, "claims_norm"),
			Concepts:   stringSliceVal(pl, "concepts_norm"),
			Echoes:     stringSliceVal(pl, "echoes"),
			Tone:       stringVal(pl, "tone"),
			Score:      pt.Score,
		}
		if existing, ok := best[dedupeKey]; !ok || r.Score > existing.Score {
			best[dedupeKey] = r
		}
	}

	// sort by score, return top limit
	results := make([]reflection, 0, len(best))
	for _, r := range best {
		results = append(results, r)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if uint64(len(results)) > limit {
		results = results[:limit]
	}
	return results, nil
}

func searchGlitches(
	ctx context.Context,
	conn *grpc.ClientConn,
	collection string,
	vec []float32,
	limit uint64,
) ([]glitch, error) {
	client := qdrant.NewPointsClient(conn)

	f32 := make([]float32, len(vec))
	copy(f32, vec)

	resp, err := client.Search(ctx, &qdrant.SearchPoints{
		CollectionName: collection,
		Vector:         f32,
		Limit:          limit,
		WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
		VectorName:     strPtr("claims_vec"),
	})
	if err != nil {
		// misfit_reports might be empty — not fatal
		return nil, nil
	}

	var results []glitch
	for _, pt := range resp.Result {
		pl := pt.Payload
		results = append(results, glitch{
			SourceFile: stringVal(pl, "source_file"),
			Report:     stringVal(pl, "report"),
			Verdict:    stringVal(pl, "verdict"),
			Score:      pt.Score,
		})
	}
	return results, nil
}

// ── list concepts ─────────────────────────────────────────────────────────────

func listConcepts(ctx context.Context, conn *grpc.ClientConn, collection string, topN int) error {
	client := qdrant.NewPointsClient(conn)
	counts := map[string]int{}
	sources := map[string]map[string]struct{}{}
	var offset *qdrant.PointId

	for {
		resp, err := client.Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: collection,
			Limit:          uint32Ptr(500),
			Offset:         offset,
			WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
			WithVectors:    &qdrant.WithVectorsSelector{SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: false}},
		})
		if err != nil {
			return err
		}
		for _, pt := range resp.Result {
			src := stringVal(pt.Payload, "source_file")
			for _, c := range stringSliceVal(pt.Payload, "concepts_norm") {
				c = strings.TrimSpace(strings.ToLower(c))
				if c == "" {
					continue
				}
				counts[c]++
				if sources[c] == nil {
					sources[c] = map[string]struct{}{}
				}
				sources[c][src] = struct{}{}
			}
		}
		if resp.NextPageOffset == nil {
			break
		}
		offset = resp.NextPageOffset
	}

	type row struct {
		concept string
		count   int
		nsrc    int
	}
	rows := make([]row, 0, len(counts))
	for c, n := range counts {
		rows = append(rows, row{c, n, len(sources[c])})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].nsrc != rows[j].nsrc {
			return rows[i].nsrc > rows[j].nsrc
		}
		return rows[i].count > rows[j].count
	})
	if len(rows) > topN {
		rows = rows[:topN]
	}

	fmt.Printf("\n%-4s %-40s %7s %7s\n", "Rank", "Concept", "Sources", "Count")
	fmt.Println(strings.Repeat("-", 62))
	for i, r := range rows {
		fmt.Printf("%-4d %-40s %7d %7d\n", i+1, r.concept, r.nsrc, r.count)
	}
	fmt.Println()
	return nil
}

// ── DeepSeek synthesis ────────────────────────────────────────────────────────

const synthesisSystemPrompt = `You are writing for curious everyday people — not academics, not spiritual insiders.
Your job is to show them something genuinely surprising: that people from completely different 
cultures, centuries, and backgrounds all independently arrived at the same idea.

Write like you're explaining it to a smart 12-year-old who just asked "wait, is that real?"
No jargon. No fluff. Short sentences. Use plain English throughout.

Structure your response EXACTLY like this:

---
## The Big Idea (plain English, 2-3 sentences max)
## The Receipts
[For each source, one paragraph: who/what/when it is, the exact idea they expressed, why it's the same thing]
## Why This Is Wild
[2-3 sentences on why it's remarkable that these sources converge — different eras, cultures, no contact]
## The Weird Part (Hardware Glitch)
[The place where the idea gets paradoxical or contradicts itself — be honest, don't hide it]
## What It Means For You
[One practical paragraph — so what? what do you do with this?]
---

Do NOT use academic language. Do NOT say "it is noteworthy that". Do NOT use the word "delve".
Write like you're texting a friend who just asked you about something mind-blowing.`

type deepSeekMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type deepSeekRequest struct {
	Model    string            `json:"model"`
	Messages []deepSeekMessage `json:"messages"`
}

type deepSeekResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
}

func synthesize(
	ctx context.Context,
	cfg config,
	concept string,
	reflections []reflection,
	glitches []glitch,
) (string, string, error) {
	// build the prompt
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("CONCEPT: %q\n\n", concept))
	sb.WriteString("## SOURCE RECEIPTS\n\n")

	for i, r := range reflections {
		sb.WriteString(fmt.Sprintf("### Source %d: %s\n", i+1, friendlySourceName(r.SourceID, r.SourceFile)))
		sb.WriteString(fmt.Sprintf("Summary: %s\n", r.Summary))
		if len(r.Claims) > 0 {
			sb.WriteString("Key claims:\n")
			for _, c := range r.Claims {
				if c != "" {
					sb.WriteString(fmt.Sprintf("- %s\n", c))
				}
			}
		}
		if len(r.Echoes) > 0 {
			sb.WriteString(fmt.Sprintf("Connects to: %s\n", strings.Join(r.Echoes, ", ")))
		}
		sb.WriteString(fmt.Sprintf("(similarity score: %.3f)\n\n", r.Score))
	}

	if len(glitches) > 0 {
		sb.WriteString("## HARDWARE GLITCHES FOUND BY CRITICS\n\n")
		for _, g := range glitches {
			if g.Report == "" {
				continue
			}
			sb.WriteString(fmt.Sprintf("From %s:\n", friendlySourceName("", g.SourceFile)))
			// truncate long reports
			report := g.Report
			if len(report) > 800 {
				report = report[:800] + "..."
			}
			sb.WriteString(report + "\n\n")
		}
	}

	sb.WriteString("\nNow write the receipts document following the system prompt format exactly.")

	apiKey := cfg.openrouterKey
	if !isOpenRouterURL(cfg.deepseekURL) && cfg.deepseekKey != "" {
		apiKey = cfg.deepseekKey
	}

	body, _ := json.Marshal(deepSeekRequest{
		Model: cfg.deepseekModel,
		Messages: []deepSeekMessage{
			{Role: "system", Content: synthesisSystemPrompt},
			{Role: "user", Content: sb.String()},
		},
	})

	req, err := http.NewRequestWithContext(ctx, "POST", cfg.deepseekURL, bytes.NewReader(body))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return "", "", fmt.Errorf("deepseek %d: %s", resp.StatusCode, b)
	}

	var out deepSeekResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", "", err
	}
	if len(out.Choices) == 0 {
		return "", "", fmt.Errorf("deepseek: empty choices")
	}
	msg := out.Choices[0].Message
	return msg.ReasoningContent, msg.Content, nil
}

// ── output ────────────────────────────────────────────────────────────────────

func writeOutput(outDir, concept, reasoning, document string, refs []reflection) error {
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return err
	}
	slug := slugify(concept)
	ts := time.Now().Format("2006-01-02_15-04")
	filename := fmt.Sprintf("%s_%s.md", slug, ts)
	path := filepath.Join(outDir, filename)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# The Receipts: %q\n\n", concept))
	sb.WriteString(fmt.Sprintf("*Generated %s from %d source traditions*\n\n", time.Now().Format("January 2, 2006 15:04"), len(refs)))
	sb.WriteString("---\n\n")
	sb.WriteString(document)
	sb.WriteString("\n\n---\n\n")
	sb.WriteString("## Sources Used\n\n")
	for i, r := range refs {
		sb.WriteString(fmt.Sprintf("%d. **%s** (similarity: %.3f)\n", i+1, friendlySourceName(r.SourceID, r.SourceFile), r.Score))
	}
	if reasoning != "" {
		sb.WriteString("\n\n<details>\n<summary>R1 Reasoning Chain</summary>\n\n")
		sb.WriteString("```\n")
		sb.WriteString(reasoning)
		sb.WriteString("\n```\n\n</details>\n")
	}

	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		return err
	}
	fmt.Printf("✅ Wrote: %s\n", path)
	return nil
}

// ── helpers ───────────────────────────────────────────────────────────────────

func friendlySourceName(sourceID, sourceFile string) string {
	// map known source IDs to human-readable names
	names := map[string]string{
		"the_nature_of_personal_reality":                                     "Seth / Jane Roberts — The Nature of Personal Reality (1974)",
		"seth_speaks":                                                        "Seth / Jane Roberts — Seth Speaks (1972)",
		"the_education_of_oversoul_seven":                                    "Jane Roberts — The Education of Oversoul Seven (1973)",
		"108_upanishads":                                                     "The 108 Upanishads (~800 BCE – 200 CE)",
		"dolores_cannon_conversations_with_nostradamusv1":                    "Dolores Cannon — Conversations with Nostradamus Vol.1 (1989)",
		"root_access_a_misfits_complete_guide_to_reality_engineering":        "ROOT ACCESS — A Misfit's Complete Guide to Reality Engineering (2025)",
		"the_dance_of_belief_unlocking_the_power_of_perception_and_creation": "The Dance of Belief — Mark J. Hubrich (2025)",
		"the_misfits_guide_to_the_clairs":                                    "The Misfit's Guide to the Clairs",
		"themisfit_spathtopower_fromburnouttobrilliance":                     "The Misfit's Path to Power — From Burnout to Brilliance",
		"the_magus": "The Magus — Francis Barrett (1801)",
	}
	if friendly, ok := names[sourceID]; ok {
		return friendly
	}
	if friendly, ok := names[sourceFile]; ok {
		return friendly
	}

	trimmed := strings.TrimSpace(sourceFile)
	if sessionHeader.MatchString(trimmed) || strings.HasPrefix(trimmed, "Chapter") || strings.HasPrefix(trimmed, "CHAPTER") {
		return trimmed
	}

	fallback := sourceFile
	if fallback == "" {
		fallback = sourceID
	}
	s := strings.ReplaceAll(fallback, "_", " ")
	return strings.Title(s)
}

var sessionHeader = regexp.MustCompile(`(?i)\bsession\s+\d+\b`)

var nonAlpha = regexp.MustCompile(`[^a-z0-9]+`)

func slugify(s string) string {
	s = strings.ToLower(s)
	s = strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return r
		}
		return '-'
	}, s)
	s = nonAlpha.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-")
	if len(s) > 60 {
		s = s[:60]
	}
	return s
}

func isOpenRouterURL(url string) bool {
	return strings.Contains(strings.ToLower(url), "openrouter.ai")
}

func strPtr(s string) *string { return &s }

func uint32Ptr(n uint32) *uint32 { return &n }

func stringVal(pl map[string]*qdrant.Value, key string) string {
	v, ok := pl[key]
	if !ok || v == nil {
		return ""
	}
	switch x := v.Kind.(type) {
	case *qdrant.Value_StringValue:
		return x.StringValue
	default:
		return ""
	}
}

func stringSliceVal(pl map[string]*qdrant.Value, key string) []string {
	v, ok := pl[key]
	if !ok || v == nil {
		return nil
	}
	lst, ok := v.Kind.(*qdrant.Value_ListValue)
	if !ok || lst.ListValue == nil {
		return nil
	}
	var out []string
	for _, item := range lst.ListValue.Values {
		if s, ok := item.Kind.(*qdrant.Value_StringValue); ok {
			out = append(out, s.StringValue)
		}
	}
	return out
}

// confirm prompts the user before doing a long R1 run
func confirm(prompt string) bool {
	fmt.Printf("%s [y/N] ", prompt)
	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		return strings.ToLower(strings.TrimSpace(scanner.Text())) == "y"
	}
	return false
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	cfg := loadConfig()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// connect to Qdrant
	conn, err := qdrantConn(cfg.qdrantHost, cfg.qdrantPort)
	if err != nil {
		fmt.Fprintf(os.Stderr, "qdrant connect: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	// --list-concepts mode
	if cfg.listConcepts {
		if err := listConcepts(ctx, conn, cfg.reflectionsCollection, cfg.topConcepts); err != nil {
			fmt.Fprintf(os.Stderr, "list concepts: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if cfg.concept == "" {
		fmt.Fprintln(os.Stderr, "error: --concept is required (or use --list-concepts to browse)")
		flag.Usage()
		os.Exit(1)
	}

	if cfg.openrouterKey == "" {
		fmt.Fprintln(os.Stderr, "error: OPENROUTER_API_KEY not set")
		os.Exit(1)
	}

	fmt.Printf("🔍 Searching corpus for receipts on: %q\n", cfg.concept)

	embedModel, detectedDim, err := detectEmbedModel(ctx, conn, cfg.reflectionsCollection, "claims_vec", cfg.embedModel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warn: embed model auto-detect failed for %s: %v\n", cfg.reflectionsCollection, err)
	} else {
		if detectedDim > 0 {
			if embedModel != cfg.embedModel {
				fmt.Printf("🧭 Using %s (detected from %s claims_vec=%d)\n", embedModel, cfg.reflectionsCollection, detectedDim)
			} else {
				fmt.Printf("🧭 Keeping %s (matched %s claims_vec=%d)\n", embedModel, cfg.reflectionsCollection, detectedDim)
			}
		}
		cfg.embedModel = embedModel
	}

	// embed the concept
	fmt.Printf("📡 Embedding with %s...\n", cfg.embedModel)
	vec, err := embed(ctx, cfg.concept, cfg.embedModel, cfg.openrouterKey)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✅ Got %d-dim vector\n", len(vec))

	// search reflections
	fmt.Printf("🔎 Searching %s for top %d source traditions...\n", cfg.reflectionsCollection, cfg.numSources)
	refs, err := searchReflections(ctx, conn, cfg.reflectionsCollection, vec, uint64(cfg.numSources))
	if err != nil {
		if expectedDim, ok := expectedVectorDimFromErr(err); ok {
			if retryModel := modelForVectorSize(expectedDim); retryModel != "" && retryModel != cfg.embedModel {
				fmt.Fprintf(os.Stderr, "⚠️ Vector dim mismatch for %s (expected %d). Retrying with %s...\n", cfg.reflectionsCollection, expectedDim, retryModel)
				vec, err = embed(ctx, cfg.concept, retryModel, cfg.openrouterKey)
				if err != nil {
					fmt.Fprintf(os.Stderr, "embed retry: %v\n", err)
					os.Exit(1)
				}
				cfg.embedModel = retryModel
				fmt.Printf("✅ Retry got %d-dim vector\n", len(vec))
				refs, err = searchReflections(ctx, conn, cfg.reflectionsCollection, vec, uint64(cfg.numSources))
			}
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "search: %v\n", err)
			os.Exit(1)
		}
	}
	if len(refs) == 0 {
		fmt.Fprintln(os.Stderr, "no reflections found — is the collection populated?")
		os.Exit(1)
	}

	fmt.Printf("✅ Found %d source traditions:\n", len(refs))
	for i, r := range refs {
		fmt.Printf("  %d. %s (score %.3f)\n", i+1, friendlySourceName(r.SourceID, r.SourceFile), r.Score)
	}

	// search misfit_reports for glitches
	fmt.Printf("🔎 Searching %s for Hardware Glitches...\n", cfg.reportsCollection)
	glitches, err := searchGlitches(ctx, conn, cfg.reportsCollection, vec, 3)
	if err != nil {
		fmt.Fprintf(os.Stderr, "glitch search (non-fatal): %v\n", err)
	}
	if len(glitches) > 0 {
		fmt.Printf("✅ Found %d Hardware Glitch receipts\n", len(glitches))
	} else {
		fmt.Println("ℹ️  No Hardware Glitch receipts yet (misfit_crew still running)")
	}

	// confirm before R1 run
	model := cfg.deepseekModel
	if model == "deepseek-reasoner" {
		if !confirm(fmt.Sprintf("🧠 Run R1 synthesis? (can take 3-5 min with %s)", model)) {
			fmt.Println("Aborted.")
			os.Exit(0)
		}
	}

	// synthesize
	fmt.Printf("🧠 Synthesizing with %s...\n", model)
	reasoning, document, err := synthesize(ctx, cfg, cfg.concept, refs, glitches)
	if err != nil {
		fmt.Fprintf(os.Stderr, "synthesize: %v\n", err)
		os.Exit(1)
	}

	// write output
	if err := writeOutput(cfg.outDir, cfg.concept, reasoning, document, refs); err != nil {
		fmt.Fprintf(os.Stderr, "write: %v\n", err)
		os.Exit(1)
	}
}
