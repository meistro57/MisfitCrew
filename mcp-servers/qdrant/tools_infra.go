package main

import "fmt"

// ── Field Indexes ─────────────────────────────────────────────────────────────

func toolListIndexes(collection string) (string, error) {
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	data, err := qdrantGet("/collections/" + collection)
	if err != nil {
		return "", err
	}
	result, _ := getResult(data).(map[string]any)
	config, _ := result["config"].(map[string]any)
	params, _ := config["params"].(map[string]any)
	indexes, _ := params["payload_schema"].(map[string]any)
	if indexes == nil {
		return fmt.Sprintf("No field indexes found on **%s**.", collection), nil
	}
	return fmt.Sprintf("## Field Indexes: **%s**\n\n```json\n%s\n```", collection, formatJSON(indexes)), nil
}

func toolCreateIndex(args map[string]any) (string, error) {
	collection := strArg(args, "collection_name")
	fieldName := strArg(args, "field_name")
	fieldSchema := strArg(args, "field_schema")
	if collection == "" || fieldName == "" || fieldSchema == "" {
		return "", fmt.Errorf("collection_name, field_name, and field_schema required")
	}

	validSchemas := map[string]bool{
		"keyword": true, "integer": true, "float": true,
		"bool": true, "geo": true, "text": true, "datetime": true,
	}
	if !validSchemas[fieldSchema] {
		return "", fmt.Errorf("field_schema must be one of: keyword, integer, float, bool, geo, text, datetime")
	}

	body := map[string]any{
		"field_name":   fieldName,
		"field_schema": fieldSchema,
	}
	if textParams := mapArg(args, "text_index_params"); textParams != nil && fieldSchema == "text" {
		body["field_schema"] = map[string]any{
			"type":   "text",
			"params": textParams,
		}
	}

	_, err := qdrantPut(fmt.Sprintf("/collections/%s/index", collection), body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Index created on **%s**.%s (%s).", collection, fieldName, fieldSchema), nil
}

func toolDeleteIndex(collection, fieldName string) (string, error) {
	if collection == "" || fieldName == "" {
		return "", fmt.Errorf("collection_name and field_name required")
	}
	_, err := qdrantDelete(fmt.Sprintf("/collections/%s/index/%s", collection, fieldName), nil)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Index on **%s**.%s deleted.", collection, fieldName), nil
}

// ── Snapshots ─────────────────────────────────────────────────────────────────

func toolListSnapshots(collection string) (string, error) {
	path := "/snapshots"
	label := "global"
	if collection != "" {
		path = fmt.Sprintf("/collections/%s/snapshots", collection)
		label = collection
	}
	data, err := qdrantGet(path)
	if err != nil {
		return "", err
	}
	result := getResult(data)
	return fmt.Sprintf("## Snapshots: **%s**\n\n```json\n%s\n```", label, formatJSON(result)), nil
}

func toolCreateSnapshot(collection string) (string, error) {
	path := "/snapshots"
	label := "global"
	if collection != "" {
		path = fmt.Sprintf("/collections/%s/snapshots", collection)
		label = collection
	}
	data, err := qdrantPost(path, nil)
	if err != nil {
		return "", err
	}
	result := getResult(data)
	return fmt.Sprintf("✅ Snapshot created for **%s**.\n\n```json\n%s\n```", label, formatJSON(result)), nil
}

func toolDeleteSnapshot(collection, snapshotName string) (string, error) {
	if snapshotName == "" {
		return "", fmt.Errorf("snapshot_name required")
	}
	path := fmt.Sprintf("/snapshots/%s", snapshotName)
	if collection != "" {
		path = fmt.Sprintf("/collections/%s/snapshots/%s", collection, snapshotName)
	}
	_, err := qdrantDelete(path, nil)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Snapshot **%s** deleted.", snapshotName), nil
}

// ── Cluster & Health ──────────────────────────────────────────────────────────

func toolHealth() (string, error) {
	data, err := qdrantGet("/healthz")
	if err != nil {
		// try alternate endpoint
		data, err = qdrantGet("/")
		if err != nil {
			return "", fmt.Errorf("qdrant health check failed: %w", err)
		}
	}
	return fmt.Sprintf("## Qdrant Health\n\n```json\n%s\n```", formatJSON(data)), nil
}

func toolTelemetry() (string, error) {
	data, err := qdrantGet("/telemetry")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Qdrant Telemetry\n\n```json\n%s\n```", formatJSON(getResult(data))), nil
}

func toolClusterInfo() (string, error) {
	data, err := qdrantGet("/cluster")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Cluster Info\n\n```json\n%s\n```", formatJSON(getResult(data))), nil
}

func toolCollectionClusterInfo(collection string) (string, error) {
	if collection == "" {
		return "", fmt.Errorf("collection_name required")
	}
	data, err := qdrantGet(fmt.Sprintf("/collections/%s/cluster", collection))
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Cluster Info: **%s**\n\n```json\n%s\n```", collection, formatJSON(getResult(data))), nil
}
