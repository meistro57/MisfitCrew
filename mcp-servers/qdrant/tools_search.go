package main

import (
	"fmt"
	"strings"
)

// ── Search ────────────────────────────────────────────────────────────────────

func toolSearch(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	vec := float32SliceArg(args, "query_vector")
	if len(vec) == 0 {
		return "", fmt.Errorf("query_vector required")
	}
	limit := intArg(args, "limit", 10)
	withPayload := boolArg(args, "with_payload", true)
	withVector := boolArg(args, "with_vector", false)
	filter := mapArg(args, "filter")
	vectorName := strArg(args, "vector_name")

	body := map[string]any{
		"limit":        limit,
		"with_payload": withPayload,
		"with_vector":  withVector,
	}
	if vectorName != "" {
		body["vector"] = map[string]any{"name": vectorName, "vector": vec}
	} else {
		body["vector"] = vec
	}
	if filter != nil {
		body["filter"] = filter
	}
	if st, ok := args["score_threshold"].(float64); ok {
		body["score_threshold"] = st
	}
	if params := mapArg(args, "params"); params != nil {
		body["params"] = params
	}

	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/search", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].([]any)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Search: **%s** — %d results\n\n", collection, len(result)))
	sb.WriteString(fmt.Sprintf("```json\n%s\n```", formatJSON(result)))
	return sb.String(), nil
}

func toolSearchBatch(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	searches, ok := args["searches"].([]any)
	if !ok || len(searches) == 0 {
		return "", fmt.Errorf("searches array required")
	}
	body := map[string]any{"searches": searches}
	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/search/batch", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].([]any)
	return fmt.Sprintf("## Batch Search: **%s** — %d result sets\n\n```json\n%s\n```",
		collection, len(result), formatJSON(result)), nil
}

func toolRecommend(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	positiveIDs := strSliceArg(args, "positive_ids")
	if len(positiveIDs) == 0 {
		return "", fmt.Errorf("positive_ids required (at least one)")
	}

	positive := make([]any, len(positiveIDs))
	for i, id := range positiveIDs {
		positive[i] = parseID(id)
	}
	negative := []any{}
	for _, id := range strSliceArg(args, "negative_ids") {
		negative = append(negative, parseID(id))
	}

	limit := intArg(args, "limit", 10)
	withPayload := boolArg(args, "with_payload", true)
	filter := mapArg(args, "filter")
	strategy := strArg(args, "strategy")

	body := map[string]any{
		"positive":     positive,
		"negative":     negative,
		"limit":        limit,
		"with_payload": withPayload,
		"with_vector":  false,
	}
	if filter != nil {
		body["filter"] = filter
	}
	if strategy != "" {
		body["strategy"] = strategy
	}

	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/recommend", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].([]any)
	return fmt.Sprintf("## Recommend: **%s** — %d results\n\n```json\n%s\n```",
		collection, len(result), formatJSON(result)), nil
}

func toolDiscover(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	context, ok := args["context"].([]any)
	if !ok || len(context) == 0 {
		return "", fmt.Errorf("context pairs required: [{positive: id, negative: id}, ...]")
	}

	limit := intArg(args, "limit", 10)
	withPayload := boolArg(args, "with_payload", true)
	filter := mapArg(args, "filter")

	body := map[string]any{
		"context":      context,
		"limit":        limit,
		"with_payload": withPayload,
	}
	if filter != nil {
		body["filter"] = filter
	}
	if target, ok := args["target"]; ok {
		body["target"] = target
	}

	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/discover", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].([]any)
	return fmt.Sprintf("## Discover: **%s** — %d results\n\n```json\n%s\n```",
		collection, len(result), formatJSON(result)), nil
}

func toolQuery(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	limit := intArg(args, "limit", 10)
	withPayload := boolArg(args, "with_payload", true)
	withVector := boolArg(args, "with_vector", false)
	filter := mapArg(args, "filter")
	offset := strArg(args, "offset")
	orderBy := strArg(args, "order_by")

	body := map[string]any{
		"limit":        limit,
		"with_payload": withPayload,
		"with_vector":  withVector,
	}
	if filter != nil {
		body["filter"] = filter
	}
	if offset != "" {
		body["offset"] = parseID(offset)
	}
	if orderBy != "" {
		body["order_by"] = map[string]any{"key": orderBy}
	}
	// optional vector query
	if vec := float32SliceArg(args, "query_vector"); len(vec) > 0 {
		body["query"] = vec
	}
	if fusionStr := strArg(args, "fusion"); fusionStr != "" {
		body["query"] = map[string]any{"fusion": fusionStr}
	}

	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/query", collection), body)
	if err != nil {
		// fallback to scroll if query endpoint not available
		data, err = qdrantPost(fmt.Sprintf("/collections/%s/points/scroll", collection), body)
		if err != nil {
			return "", err
		}
		result, _ := data["result"].(map[string]any)
		points, _ := result["points"].([]any)
		nextOffset := result["next_page_offset"]
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("## Query (scroll): **%s** — %d points\n\n", collection, len(points)))
		if nextOffset != nil {
			sb.WriteString(fmt.Sprintf("> Next page: `%v`\n\n", nextOffset))
		}
		sb.WriteString(fmt.Sprintf("```json\n%s\n```", formatJSON(points)))
		return sb.String(), nil
	}

	result, _ := data["result"].(map[string]any)
	points, _ := result["points"].([]any)
	nextOffset := result["next_page_offset"]

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Query: **%s** — %d points\n\n", collection, len(points)))
	if nextOffset != nil {
		sb.WriteString(fmt.Sprintf("> Next page: `%v`\n\n", nextOffset))
	}
	sb.WriteString(fmt.Sprintf("```json\n%s\n```", formatJSON(points)))
	return sb.String(), nil
}

func toolQueryBatch(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	searches, ok := args["searches"].([]any)
	if !ok || len(searches) == 0 {
		return "", fmt.Errorf("searches array required")
	}
	body := map[string]any{"searches": searches}
	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/query/batch", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].([]any)
	return fmt.Sprintf("## Query Batch: **%s** — %d result sets\n\n```json\n%s\n```",
		collection, len(result), formatJSON(result)), nil
}
