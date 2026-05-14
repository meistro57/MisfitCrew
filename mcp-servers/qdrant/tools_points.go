package main

import (
	"fmt"
	"strings"
)

// ── Points ────────────────────────────────────────────────────────────────────

func toolUpsertPoints(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	points, ok := args["points"].([]any)
	if !ok || len(points) == 0 {
		return "", fmt.Errorf("points array required")
	}
	wait := boolArg(args, "wait", true)
	ordering := strArg(args, "ordering")

	path := fmt.Sprintf("/collections/%s/points", collection)
	if wait {
		path += "?wait=true"
	}
	if ordering != "" {
		if wait {
			path += "&ordering=" + ordering
		} else {
			path += "?ordering=" + ordering
		}
	}

	body := map[string]any{"points": points}
	data, err := qdrantPut(path, body)
	if err != nil {
		return "", err
	}
	result := getResult(data)
	return fmt.Sprintf("✅ Upserted %d points into **%s**.\n%s", len(points), collection, formatJSON(result)), nil
}

func toolGetPoint(collection, pointID string, withPayload, withVector bool) (string, error) {
	if collection == "" || pointID == "" {
		return "", fmt.Errorf("collection_name and point_id required")
	}
	path := fmt.Sprintf("/collections/%s/points/%s?with_payload=%v&with_vector=%v",
		collection, pointID, withPayload, withVector)
	data, err := qdrantGet(path)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Point `%s` in **%s**\n\n```json\n%s\n```", pointID, collection, formatJSON(getResult(data))), nil
}

func toolGetPoints(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	ids := strSliceArg(args, "point_ids")
	if len(ids) == 0 {
		return "", fmt.Errorf("point_ids required")
	}
	withPayload := boolArg(args, "with_payload", true)
	withVector := boolArg(args, "with_vector", false)

	parsedIDs := make([]any, len(ids))
	for i, id := range ids {
		parsedIDs[i] = parseID(id)
	}

	body := map[string]any{
		"ids":          parsedIDs,
		"with_payload": withPayload,
		"with_vector":  withVector,
	}
	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := getResult(data).([]any)
	return fmt.Sprintf("## Points from **%s** (%d retrieved)\n\n```json\n%s\n```",
		collection, len(result), formatJSON(result)), nil
}

func toolScrollPoints(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	limit := intArg(args, "limit", 10)
	if limit > 100 {
		limit = 100
	}
	offset := strArg(args, "offset")
	withPayload := boolArg(args, "with_payload", true)
	withVector := boolArg(args, "with_vector", false)
	filter := mapArg(args, "filter")
	orderBy := strArg(args, "order_by")

	body := map[string]any{
		"limit":        limit,
		"with_payload": withPayload,
		"with_vector":  withVector,
	}
	if offset != "" {
		body["offset"] = parseID(offset)
	}
	if filter != nil {
		body["filter"] = filter
	}
	if orderBy != "" {
		body["order_by"] = map[string]any{"key": orderBy}
	}

	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/scroll", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].(map[string]any)
	points, _ := result["points"].([]any)
	nextOffset := result["next_page_offset"]

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Scroll: **%s** — %d points\n\n", collection, len(points)))
	if nextOffset != nil {
		sb.WriteString(fmt.Sprintf("> **Next page offset:** `%v`\n\n", nextOffset))
	} else {
		sb.WriteString("> End of collection.\n\n")
	}
	sb.WriteString(fmt.Sprintf("```json\n%s\n```", formatJSON(points)))
	return sb.String(), nil
}

func toolCountPoints(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	exact := boolArg(args, "exact", true)
	filter := mapArg(args, "filter")

	body := map[string]any{"exact": exact}
	if filter != nil {
		body["filter"] = filter
	}

	data, err := qdrantPost(fmt.Sprintf("/collections/%s/points/count", collection), body)
	if err != nil {
		return "", err
	}
	result, _ := data["result"].(map[string]any)
	count, _ := result["count"].(float64)

	qualifier := ""
	if filter != nil {
		qualifier = " (filtered)"
	}
	return fmt.Sprintf("**%s**%s: **%d** points", collection, qualifier, int(count)), nil
}

func toolDeletePoints(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}

	body := map[string]any{}
	ids := strSliceArg(args, "point_ids")
	filter := mapArg(args, "filter")

	if len(ids) > 0 {
		parsedIDs := make([]any, len(ids))
		for i, id := range ids {
			parsedIDs[i] = parseID(id)
		}
		body["points"] = parsedIDs
	} else if filter != nil {
		body["filter"] = filter
	} else {
		return "", fmt.Errorf("provide point_ids or filter")
	}

	_, err := qdrantPost(fmt.Sprintf("/collections/%s/points/delete?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	if len(ids) > 0 {
		return fmt.Sprintf("🗑️ Deleted %d points from **%s**.", len(ids), collection), nil
	}
	return fmt.Sprintf("🗑️ Deleted points matching filter from **%s**.", collection), nil
}

func toolUpdateVectors(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	points, ok := args["points"].([]any)
	if !ok || len(points) == 0 {
		return "", fmt.Errorf("points array required — each item: {id, vector}")
	}
	body := map[string]any{"points": points}
	_, err := qdrantPut(fmt.Sprintf("/collections/%s/points/vectors?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Updated vectors for %d points in **%s**.", len(points), collection), nil
}

func toolDeleteVectors(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	ids := strSliceArg(args, "point_ids")
	vectors := strSliceArg(args, "vector_names")
	if len(ids) == 0 || len(vectors) == 0 {
		return "", fmt.Errorf("point_ids and vector_names required")
	}
	parsedIDs := make([]any, len(ids))
	for i, id := range ids {
		parsedIDs[i] = parseID(id)
	}
	body := map[string]any{
		"points":  parsedIDs,
		"vectors": vectors,
	}
	_, err := qdrantPost(fmt.Sprintf("/collections/%s/points/vectors/delete?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Deleted vectors %v from %d points in **%s**.", vectors, len(ids), collection), nil
}

// ── Payload ───────────────────────────────────────────────────────────────────

func toolSetPayload(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	payload := mapArg(args, "payload")
	if collection == "" || payload == nil {
		return "", fmt.Errorf("collection_name and payload required")
	}
	body := map[string]any{"payload": payload}
	ids := strSliceArg(args, "point_ids")
	filter := mapArg(args, "filter")
	if len(ids) > 0 {
		parsedIDs := make([]any, len(ids))
		for i, id := range ids {
			parsedIDs[i] = parseID(id)
		}
		body["points"] = parsedIDs
	} else if filter != nil {
		body["filter"] = filter
	} else {
		return "", fmt.Errorf("provide point_ids or filter")
	}
	_, err := qdrantPost(fmt.Sprintf("/collections/%s/points/payload?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Payload set on points in **%s**.", collection), nil
}

func toolOverwritePayload(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	payload := mapArg(args, "payload")
	if collection == "" || payload == nil {
		return "", fmt.Errorf("collection_name and payload required")
	}
	body := map[string]any{"payload": payload}
	ids := strSliceArg(args, "point_ids")
	if len(ids) > 0 {
		parsedIDs := make([]any, len(ids))
		for i, id := range ids {
			parsedIDs[i] = parseID(id)
		}
		body["points"] = parsedIDs
	} else if filter := mapArg(args, "filter"); filter != nil {
		body["filter"] = filter
	} else {
		return "", fmt.Errorf("provide point_ids or filter")
	}
	_, err := qdrantPut(fmt.Sprintf("/collections/%s/points/payload?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Payload overwritten on points in **%s**.", collection), nil
}

func toolDeletePayload(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	keys := strSliceArg(args, "keys")
	if collection == "" || len(keys) == 0 {
		return "", fmt.Errorf("collection_name and keys required")
	}
	body := map[string]any{"keys": keys}
	ids := strSliceArg(args, "point_ids")
	if len(ids) > 0 {
		parsedIDs := make([]any, len(ids))
		for i, id := range ids {
			parsedIDs[i] = parseID(id)
		}
		body["points"] = parsedIDs
	} else if filter := mapArg(args, "filter"); filter != nil {
		body["filter"] = filter
	} else {
		return "", fmt.Errorf("provide point_ids or filter")
	}
	_, err := qdrantPost(fmt.Sprintf("/collections/%s/points/payload/delete?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Deleted payload keys %v from points in **%s**.", keys, collection), nil
}

func toolClearPayload(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	body := map[string]any{}
	ids := strSliceArg(args, "point_ids")
	if len(ids) > 0 {
		parsedIDs := make([]any, len(ids))
		for i, id := range ids {
			parsedIDs[i] = parseID(id)
		}
		body["points"] = parsedIDs
	} else if filter := mapArg(args, "filter"); filter != nil {
		body["filter"] = filter
	} else {
		return "", fmt.Errorf("provide point_ids or filter")
	}
	_, err := qdrantPost(fmt.Sprintf("/collections/%s/points/payload/clear?wait=true", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Cleared all payload from points in **%s**.", collection), nil
}

// ── ID parsing ────────────────────────────────────────────────────────────────

func parseID(s string) any {
	// Try uint64 first, fall back to string (UUID)
	var n uint64
	if _, err := fmt.Sscanf(s, "%d", &n); err == nil {
		return n
	}
	return s
}
