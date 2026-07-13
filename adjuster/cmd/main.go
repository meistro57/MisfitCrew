/*
adjuster — The Broom.

Follows vectoreologist through vectoreology_findings and acts on each
finding exactly once. Idempotent by design — findings already processed
are skipped via the processed_by_adjuster flag written back to Qdrant.

Actions by finding type:
  cluster            → tag all meta_reflections points in the cluster with
                       cluster_id, cluster_name, cluster_source
  bridge             → set is_bridge=true + bridge_partners on member points
  moat               → set needs_review=true on isolated points
  anomaly            → set priority_critique=true for misfit_crew
  source_contradiction → set has_contradiction=true for receipts Hardware Glitch

After acting, writes back to the finding point in vectoreology_findings:
  processed_by_adjuster: true
  adjuster_action:       summary of what was done
  adjuster_at:           RFC3339 timestamp
  adjuster_version:      binary version

Usage:
  adjuster                          # run once, process all unswept findings
  adjuster --watch 60               # run every 60 seconds
  adjuster --dry-run                # print actions without writing
  adjuster --findings vectoreology_findings
  adjuster --reflections meta_reflections

Environment (.env or shell):
  QDRANT_URL      default http://localhost:6333
*/
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var version = "dev"

// ── config ────────────────────────────────────────────────────────────────────

type config struct {
	qdrantHost         string
	qdrantPort         int
	findingsCollection string
	reflectCollection  string
	watchSecs          int
	dryRun             bool
	reflag             bool // backfill loop_* flags on unflagged meta_reflections
	reflagBatch        int  // set_payload batch size for reflag mode
}

func loadConfig() config {
	_ = godotenv.Load(".env")

	cfg := config{
		qdrantHost:         envOr("QDRANT_HOST", "localhost"),
		qdrantPort:         6334,
		findingsCollection: "vectoreology_findings",
		reflectCollection:  "meta_reflections",
	}

	flag.StringVar(&cfg.findingsCollection, "findings", cfg.findingsCollection, "vectoreology findings collection")
	flag.StringVar(&cfg.reflectCollection, "reflections", cfg.reflectCollection, "meta reflections collection")
	flag.IntVar(&cfg.watchSecs, "watch", 0, "re-run every N seconds (0 = run once)")
	flag.BoolVar(&cfg.dryRun, "dry-run", false, "print actions without writing to Qdrant")
	flag.BoolVar(&cfg.reflag, "reflag", false, "backfill loop_* flags on meta_reflections points that are missing them")
	flag.IntVar(&cfg.reflagBatch, "reflag-batch", 500, "set_payload batch size for --reflag mode")
	flag.Parse()

	for _, a := range os.Args[1:] {
		if a == "--version" || a == "-version" {
			fmt.Println("adjuster", version)
			os.Exit(0)
		}
	}
	return cfg
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ── Qdrant client ─────────────────────────────────────────────────────────────

func qdrantConn(host string, port int) (*grpc.ClientConn, error) {
	return grpc.NewClient(
		fmt.Sprintf("%s:%d", host, port),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(256*1024*1024)),
	)
}

// ── payload helpers ───────────────────────────────────────────────────────────

func strVal(pl map[string]*qdrant.Value, key string) string {
	v, ok := pl[key]
	if !ok || v == nil {
		return ""
	}
	if s, ok := v.Kind.(*qdrant.Value_StringValue); ok {
		return s.StringValue
	}
	return ""
}

func boolVal(pl map[string]*qdrant.Value, key string) bool {
	v, ok := pl[key]
	if !ok || v == nil {
		return false
	}
	if b, ok := v.Kind.(*qdrant.Value_BoolValue); ok {
		return b.BoolValue
	}
	return false
}

func strSliceVal(pl map[string]*qdrant.Value, key string) []string {
	v, ok := pl[key]
	if !ok || v == nil {
		return nil
	}
	lv, ok := v.Kind.(*qdrant.Value_ListValue)
	if !ok || lv.ListValue == nil {
		return nil
	}
	var out []string
	for _, item := range lv.ListValue.Values {
		if s, ok := item.Kind.(*qdrant.Value_StringValue); ok {
			out = append(out, s.StringValue)
		}
	}
	return out
}

func qStr(s string) *qdrant.Value {
	return &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: s}}
}

func qBool(b bool) *qdrant.Value {
	return &qdrant.Value{Kind: &qdrant.Value_BoolValue{BoolValue: b}}
}

func qList(items []string) *qdrant.Value {
	vals := make([]*qdrant.Value, len(items))
	for i, s := range items {
		vals[i] = qStr(s)
	}
	return &qdrant.Value{Kind: &qdrant.Value_ListValue{
		ListValue: &qdrant.ListValue{Values: vals},
	}}
}

func pointIDFromString(s string) *qdrant.PointId {
	return &qdrant.PointId{PointIdOptions: &qdrant.PointId_Uuid{Uuid: s}}
}

// ── finding ───────────────────────────────────────────────────────────────────

type finding struct {
	id      string
	payload map[string]*qdrant.Value
}

func (f finding) findingType() string   { return strVal(f.payload, "type") }
func (f finding) subject() string       { return strVal(f.payload, "subject") }
func (f finding) alreadySwept() bool    { return boolVal(f.payload, "processed_by_adjuster") }
func (f finding) clusterLabel() string  { return strVal(f.payload, "cluster_label") }
func (f finding) memberIDs() []string   { return strSliceVal(f.payload, "member_point_ids") }
func (f finding) bridgePartners() []string {
	// stored as "cluster_a_label ↔ cluster_b_label" in subject, or as list
	partners := strSliceVal(f.payload, "bridge_clusters")
	if len(partners) == 0 {
		// parse from subject "X ↔ Y"
		subj := f.subject()
		if strings.Contains(subj, "↔") {
			parts := strings.Split(subj, "↔")
			for i := range parts {
				parts[i] = strings.TrimSpace(parts[i])
			}
			return parts
		}
	}
	return partners
}

func clusterLabelFromSubject(subject string) string {
	if i := strings.Index(subject, ":"); i >= 0 {
		return strings.TrimSpace(subject[i+1:])
	}
	return strings.TrimSpace(subject)
}

// ── reflag: backfill loop_* flags ────────────────────────────────────────────
//
// Scrolls meta_reflections and finds every point that is missing the
// loop_decision field (i.e. pre-dates the reflect_loop flag persistence work).
// For each unflagged point it evaluates the existing payload using the same
// logic as reflect_loop.py and writes loop_interesting, loop_contradiction,
// loop_decision, and loop_flagged_at via set_payload.
// No LLM calls — pure local computation on stored payload fields.

type reflagStats struct {
	total       int
	alreadyDone int
	interesting int
	contradiction int
	continueScan  int
	written       int
	errors        int
}

// contradictionMarkers mirrors reflect_loop.py detect_possible_contradiction.
var contradictionMarkers = []string{
	"not ", "never", "cannot", "opposite", "contradict",
	"conflict", "but ", "however", "rather than", "instead",
	"unlike", "deny", "denies", "reject", "rejects", "inconsistent",
}

func floatVal(pl map[string]*qdrant.Value, key string) float64 {
	v, ok := pl[key]
	if !ok || v == nil {
		return 0
	}
	if d, ok := v.Kind.(*qdrant.Value_DoubleValue); ok {
		return d.DoubleValue
	}
	return 0
}

func listLen(pl map[string]*qdrant.Value, key string) int {
	v, ok := pl[key]
	if !ok || v == nil {
		return 0
	}
	lv, ok := v.Kind.(*qdrant.Value_ListValue)
	if !ok || lv.ListValue == nil {
		return 0
	}
	return len(lv.ListValue.Values)
}

// listContainsMarker returns true if any string item in the list contains a
// contradiction marker substring.
func listContainsMarker(pl map[string]*qdrant.Value, key string) bool {
	v, ok := pl[key]
	if !ok || v == nil {
		return false
	}
	lv, ok := v.Kind.(*qdrant.Value_ListValue)
	if !ok || lv.ListValue == nil {
		return false
	}
	for _, item := range lv.ListValue.Values {
		s := ""
		if sv, ok := item.Kind.(*qdrant.Value_StringValue); ok {
			s = strings.ToLower(sv.StringValue)
		}
		for _, m := range contradictionMarkers {
			if strings.Contains(s, m) {
				return true
			}
		}
	}
	return false
}

func evaluatePoint(pl map[string]*qdrant.Value) (decision string, interesting bool, contradiction bool) {
	confidence   := floatVal(pl, "reflection_confidence")
	claimCount   := listLen(pl, "claims")
	conceptCount := listLen(pl, "concepts")
	questionCount := listLen(pl, "questions")
	echoCount    := listLen(pl, "echoes")

	hasContradiction := listContainsMarker(pl, "claims")

	isInteresting := confidence >= 0.60 ||
		claimCount >= 3 ||
		conceptCount >= 4 ||
		questionCount >= 2 ||
		echoCount >= 2

	switch {
	case hasContradiction:
		return "track_contradiction", false, true
	case isInteresting:
		return "store_interesting", true, false
	default:
		return "continue_scan", false, false
	}
}

func qInt(n int64) *qdrant.Value {
	return &qdrant.Value{Kind: &qdrant.Value_IntegerValue{IntegerValue: n}}
}

func reflagReflections(ctx context.Context, conn *grpc.ClientConn, cfg config) error {
	client := qdrant.NewPointsClient(conn)

	fmt.Printf("\n🏷️  reflag mode — scanning %s for unflagged points\n", cfg.reflectCollection)
	if cfg.dryRun {
		fmt.Println("  mode: DRY RUN")
	}

	var stats reflagStats
	var offset *qdrant.PointId

	// batch of (pointID, decision, interesting, contradiction) to flush
	type pendingTag struct {
		pid          *qdrant.PointId
		decision     string
		interesting  bool
		contradiction bool
	}
	var batch []pendingTag

	flushBatch := func() error {
		if len(batch) == 0 || cfg.dryRun {
			stats.written += len(batch)
			batch = batch[:0]
			return nil
		}
		now := time.Now().Unix()
		// We must call SetPayload once per unique payload combination, OR
		// call it per-point. Per-point is simplest and still fast in batches.
		for _, p := range batch {
			payload := map[string]*qdrant.Value{
				"loop_interesting":   qBool(p.interesting),
				"loop_contradiction": qBool(p.contradiction),
				"loop_decision":      qStr(p.decision),
				"loop_flagged_at":    qInt(now),
			}
			_, err := client.SetPayload(ctx, &qdrant.SetPayloadPoints{
				CollectionName: cfg.reflectCollection,
				Payload:        payload,
				PointsSelector: &qdrant.PointsSelector{
					PointsSelectorOneOf: &qdrant.PointsSelector_Points{
						Points: &qdrant.PointsIdsList{Ids: []*qdrant.PointId{p.pid}},
					},
				},
			})
			if err != nil {
				stats.errors++
				fmt.Fprintf(os.Stderr, "  set_payload error: %v\n", err)
				continue
			}
			stats.written++
		}
		batch = batch[:0]
		return nil
	}

	for {
		resp, err := client.Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: cfg.reflectCollection,
			Limit:          uint32Ptr(250),
			Offset:         offset,
			WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
			WithVectors:    &qdrant.WithVectorsSelector{SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: false}},
		})
		if err != nil {
			return fmt.Errorf("scroll: %w", err)
		}

		for _, pt := range resp.Result {
			stats.total++
			pl := pt.Payload

			// skip if already flagged
			if _, exists := pl["loop_decision"]; exists {
				stats.alreadyDone++
				continue
			}

			decision, interesting, contradiction := evaluatePoint(pl)
			switch decision {
			case "store_interesting":
				stats.interesting++
			case "track_contradiction":
				stats.contradiction++
			default:
				stats.continueScan++
			}

			batch = append(batch, pendingTag{
				pid:          pt.Id,
				decision:     decision,
				interesting:  interesting,
				contradiction: contradiction,
			})

			if len(batch) >= cfg.reflagBatch {
				if err := flushBatch(); err != nil {
					return err
				}
				fmt.Printf("  ↳ flagged %d so far (interesting=%d contradiction=%d scan=%d errors=%d)\n",
					stats.written, stats.interesting, stats.contradiction, stats.continueScan, stats.errors)
			}
		}

		if resp.NextPageOffset == nil {
			break
		}
		offset = resp.NextPageOffset
	}

	// flush remainder
	if err := flushBatch(); err != nil {
		return err
	}

	dryTag := ""
	if cfg.dryRun {
		dryTag = " (dry-run)"
	}
	fmt.Printf("\n✅ reflag complete%s\n", dryTag)
	fmt.Printf("  total scanned:    %d\n", stats.total)
	fmt.Printf("  already flagged:  %d\n", stats.alreadyDone)
	fmt.Printf("  newly flagged:    %d\n", stats.written)
	fmt.Printf("    interesting:    %d\n", stats.interesting)
	fmt.Printf("    contradiction:  %d\n", stats.contradiction)
	fmt.Printf("    continue_scan:  %d\n", stats.continueScan)
	fmt.Printf("  errors:           %d\n", stats.errors)
	return nil
}

// ── scroll findings ───────────────────────────────────────────────────────────

func scrollFindings(ctx context.Context, client qdrant.PointsClient, collection string) ([]finding, error) {
	var findings []finding
	var offset *qdrant.PointId

	for {
		resp, err := client.Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: collection,
			Limit:          uint32Ptr(100),
			Offset:         offset,
			WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
			WithVectors:    &qdrant.WithVectorsSelector{SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: false}},
		})
		if err != nil {
			return nil, fmt.Errorf("scroll findings: %w", err)
		}
		for _, pt := range resp.Result {
			id := ""
			if u, ok := pt.Id.PointIdOptions.(*qdrant.PointId_Uuid); ok {
				id = u.Uuid
			} else if n, ok := pt.Id.PointIdOptions.(*qdrant.PointId_Num); ok {
				id = fmt.Sprintf("%d", n.Num)
			}
			findings = append(findings, finding{id: id, payload: pt.Payload})
		}
		if resp.NextPageOffset == nil {
			break
		}
		offset = resp.NextPageOffset
	}
	return findings, nil
}

// ── tag meta_reflections points ───────────────────────────────────────────────

func tagReflectionPoints(
	ctx context.Context,
	client qdrant.PointsClient,
	collection string,
	memberIDs []string,
	tags map[string]*qdrant.Value,
	dryRun bool,
) (int, error) {
	if len(memberIDs) == 0 {
		return 0, nil
	}

	// build point ID list
	var pointIDs []*qdrant.PointId
	for _, id := range memberIDs {
		// try UUID first, then numeric
		if isUUID(id) {
			pointIDs = append(pointIDs, pointIDFromString(id))
		}
		// numeric IDs handled via search — we tag by payload filter instead
	}

	if len(pointIDs) == 0 {
		// fall back: nothing to do if no valid IDs
		return 0, nil
	}

	if dryRun {
		return len(pointIDs), nil
	}

	// SetPayload on specific point IDs
	_, err := client.SetPayload(ctx, &qdrant.SetPayloadPoints{
		CollectionName: collection,
		Payload:        tags,
		PointsSelector: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{Ids: pointIDs},
			},
		},
	})
	if err != nil {
		return 0, fmt.Errorf("set payload on reflections: %w", err)
	}
	return len(pointIDs), nil
}

// ── mark finding as swept ─────────────────────────────────────────────────────

func markSwept(
	ctx context.Context,
	client qdrant.PointsClient,
	collection string,
	findingID string,
	action string,
	dryRun bool,
) error {
	if dryRun {
		return nil
	}

	payload := map[string]*qdrant.Value{
		"processed_by_adjuster": qBool(true),
		"adjuster_action":       qStr(action),
		"adjuster_at":           qStr(time.Now().UTC().Format(time.RFC3339)),
		"adjuster_version":      qStr(version),
	}

	var pid *qdrant.PointId
	if isUUID(findingID) {
		pid = pointIDFromString(findingID)
	} else {
		// numeric
		var n uint64
		fmt.Sscanf(findingID, "%d", &n)
		pid = &qdrant.PointId{PointIdOptions: &qdrant.PointId_Num{Num: n}}
	}

	_, err := client.SetPayload(ctx, &qdrant.SetPayloadPoints{
		CollectionName: collection,
		Payload:        payload,
		PointsSelector: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{Ids: []*qdrant.PointId{pid}},
			},
		},
	})
	return err
}

// ── process one finding ───────────────────────────────────────────────────────

func processFinding(
	ctx context.Context,
	client qdrant.PointsClient,
	cfg config,
	f finding,
) (string, error) {
	ftype := f.findingType()
	subject := f.subject()
	members := f.memberIDs()
	clusterLabel := f.clusterLabel()
	if ftype == "cluster_analysis" {
		if parsed := clusterLabelFromSubject(subject); parsed != "" {
			clusterLabel = parsed
		}
	}
	if clusterLabel == "" {
		clusterLabel = subject
	}

	switch ftype {

	case "cluster_analysis":
		// Tag member reflections with cluster identity
		tags := map[string]*qdrant.Value{
			"cluster_name":   qStr(clusterLabel),
			"cluster_source": qStr(f.id),
			"in_cluster":     qBool(true),
		}
		n, err := tagReflectionPoints(ctx, client, cfg.reflectCollection, members, tags, cfg.dryRun)
		if err != nil {
			return "", err
		}
		action := fmt.Sprintf("tagged %d meta_reflections as cluster %q", n, clusterLabel)
		fmt.Printf("  ✅ cluster: %s → %s\n", clusterLabel, action)
		return action, nil

	case "bridge_analysis":
		// Tag member reflections as bridge points
		partners := f.bridgePartners()
		tags := map[string]*qdrant.Value{
			"is_bridge":       qBool(true),
			"bridge_name":     qStr(subject),
			"bridge_partners": qList(partners),
			"bridge_source":   qStr(f.id),
		}
		n, err := tagReflectionPoints(ctx, client, cfg.reflectCollection, members, tags, cfg.dryRun)
		if err != nil {
			return "", err
		}
		action := fmt.Sprintf("tagged %d meta_reflections as bridge %q (partners: %s)", n, subject, strings.Join(partners, ", "))
		fmt.Printf("  🌉 bridge: %s → %s\n", subject, action)
		return action, nil

	case "density_anomaly":
		// Tag isolated points for review
		tags := map[string]*qdrant.Value{
			"needs_review": qBool(true),
			"moat_name":    qStr(subject),
			"moat_source":  qStr(f.id),
		}
		n, err := tagReflectionPoints(ctx, client, cfg.reflectCollection, members, tags, cfg.dryRun)
		if err != nil {
			return "", err
		}
		action := fmt.Sprintf("flagged %d meta_reflections as moat (needs_review) for %q", n, subject)
		fmt.Printf("  🏝  moat: %s → %s\n", subject, action)
		return action, nil

	case "coherence_anomaly", "source_contradiction":
		// Flag for priority critique by misfit_crew
		tags := map[string]*qdrant.Value{
			"priority_critique":  qBool(true),
			"has_contradiction":  qBool(ftype == "source_contradiction"),
			"anomaly_name":       qStr(subject),
			"anomaly_source":     qStr(f.id),
		}
		n, err := tagReflectionPoints(ctx, client, cfg.reflectCollection, members, tags, cfg.dryRun)
		if err != nil {
			return "", err
		}
		action := fmt.Sprintf("flagged %d meta_reflections as priority_critique for anomaly %q", n, subject)
		fmt.Printf("  ⚠️  anomaly: %s → %s\n", subject, action)
		return action, nil

	default:
		action := fmt.Sprintf("unknown finding type %q — skipped", ftype)
		fmt.Printf("  ❓ unknown type %q: %s\n", ftype, subject)
		return action, nil
	}
}

// ── run one sweep ─────────────────────────────────────────────────────────────

func sweep(ctx context.Context, conn *grpc.ClientConn, cfg config) error {
	client := qdrant.NewPointsClient(conn)

	findings, err := scrollFindings(ctx, client, cfg.findingsCollection)
	if err != nil {
		return err
	}

	var pending []finding
	for _, f := range findings {
		if !f.alreadySwept() {
			pending = append(pending, f)
		}
	}

	total := len(findings)
	skipped := total - len(pending)
	fmt.Printf("\n🧹 adjuster sweep — %d findings total, %d already swept, %d pending\n\n",
		total, skipped, len(pending))

	if len(pending) == 0 {
		fmt.Println("  nothing to do.")
		return nil
	}

	acted := 0
	errors := 0
	for _, f := range pending {
		action, err := processFinding(ctx, client, cfg, f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  ❌ error processing finding %s: %v\n", f.id, err)
			errors++
			continue
		}
		if err := markSwept(ctx, client, cfg.findingsCollection, f.id, action, cfg.dryRun); err != nil {
			fmt.Fprintf(os.Stderr, "  ❌ error marking swept %s: %v\n", f.id, err)
			errors++
			continue
		}
		acted++
	}

	dryTag := ""
	if cfg.dryRun {
		dryTag = " (dry-run)"
	}
	fmt.Printf("\n✅ sweep complete%s — acted on %d findings, %d errors\n", dryTag, acted, errors)
	return nil
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	cfg := loadConfig()

	conn, err := qdrantConn(cfg.qdrantHost, cfg.qdrantPort)
	if err != nil {
		fmt.Fprintf(os.Stderr, "qdrant connect: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	fmt.Printf("adjuster %s\n", version)
	fmt.Printf("  findings:    %s\n", cfg.findingsCollection)
	fmt.Printf("  reflections: %s\n", cfg.reflectCollection)
	if cfg.dryRun {
		fmt.Println("  mode:        DRY RUN")
	}
	if cfg.watchSecs > 0 {
		fmt.Printf("  watch:       every %ds\n", cfg.watchSecs)
	}

	// --reflag runs once and exits — no watch mode needed
	if cfg.reflag {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		if err := reflagReflections(ctx, conn, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "reflag error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	run := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		if err := sweep(ctx, conn, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "sweep error: %v\n", err)
		}
	}

	run()

	if cfg.watchSecs > 0 {
		ticker := time.NewTicker(time.Duration(cfg.watchSecs) * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			run()
		}
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func isUUID(s string) bool {
	// simple UUID check: 8-4-4-4-12 hex chars
	if len(s) != 36 {
		return false
	}
	dashes := []int{8, 13, 18, 23}
	for _, d := range dashes {
		if s[d] != '-' {
			return false
		}
	}
	return true
}

func uint32Ptr(n uint32) *uint32 { return &n }

// jsonStr is a debug helper
func jsonStr(v any) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}

var _ = jsonStr // suppress unused warning
