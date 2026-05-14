package main

import (
	"fmt"
	"strings"
)

// ── Collections ───────────────────────────────────────────────────────────────

func toolListCollections() (string, error) {
	data, err := qdrantGet("/collections")
	if err != nil {
		return "", err
	}
	result, _ := getResult(data).(map[string]any)
	collections, _ := result["collections"].([]any)

	var sb strings.Builder
	sb.WriteString("## Qdrant Collections\n\n")
	if len(collections) == 0 {
		sb.WriteString("No collections found.\n")
		return sb.String(), nil
	}
	for _, c := range collections {
		col := c.(map[string]any)
		name := col["name"].(string)
		info, err := qdrantGet("/collections/" + name)
		count := "unknown"
		if err == nil {
			if r, ok := getResult(info).(map[string]any); ok {
				if pc, ok := r["points_count"].(float64); ok {
					count = fmt.Sprintf("%d", int(pc))
				}
			}
		}
		sb.WriteString(fmt.Sprintf("- **%s** — %s points\n", name, count))
	}
	return sb.String(), nil
}

func toolGetCollection(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("collection_name required")
	}
	data, err := qdrantGet("/collections/" + name)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Collection: %s\n\n```json\n%s\n```", name, formatJSON(getResult(data))), nil
}

func toolCollectionExists(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("collection_name required")
	}
	data, err := qdrantGet("/collections/" + name + "/exists")
	if err != nil {
		return fmt.Sprintf("Collection **%s** does not exist.", name), nil
	}
	result, _ := getResult(data).(map[string]any)
	exists, _ := result["exists"].(bool)
	if exists {
		return fmt.Sprintf("✅ Collection **%s** exists.", name), nil
	}
	return fmt.Sprintf("❌ Collection **%s** does not exist.", name), nil
}

func toolCreateCollection(args map[string]any) (string, error) {
	name := strArg(args, "collection_name")
	if name == "" {
		return "", fmt.Errorf("collection_name required")
	}
	vectorSize := intArg(args, "vector_size", 0)
	if vectorSize <= 0 {
		return "", fmt.Errorf("vector_size must be a positive integer")
	}
	distance := strArg(args, "distance")
	if distance == "" {
		distance = "Cosine"
	}
	onDisk := boolArg(args, "on_disk", false)
	onDiskPayload := boolArg(args, "on_disk_payload", false)
	replicationFactor := intArg(args, "replication_factor", 0)

	body := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": distance,
			"on_disk":  onDisk,
		},
		"on_disk_payload": onDiskPayload,
	}
	if replicationFactor > 0 {
		body["replication_factor"] = replicationFactor
	}
	if hnsw := mapArg(args, "hnsw_config"); hnsw != nil {
		body["hnsw_config"] = hnsw
	}
	if opt := mapArg(args, "optimizers_config"); opt != nil {
		body["optimizers_config"] = opt
	}

	_, err := qdrantPut("/collections/"+name, body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Collection **%s** created.\n- Vector size: %d\n- Distance: %s\n- On disk: %v", name, vectorSize, distance, onDisk), nil
}

func toolUpdateCollection(args map[string]any) (string, error) {
	name := strArg(args, "collection_name")
	if name == "" {
		return "", fmt.Errorf("collection_name required")
	}
	body := map[string]any{}
	if opt := mapArg(args, "optimizers_config"); opt != nil {
		body["optimizers_config"] = opt
	}
	if hnsw := mapArg(args, "hnsw_config"); hnsw != nil {
		body["hnsw_config"] = hnsw
	}
	if params := mapArg(args, "params"); params != nil {
		body["params"] = params
	}
	if len(body) == 0 {
		return "", fmt.Errorf("provide at least one of: optimizers_config, hnsw_config, params")
	}
	_, err := qdrantPatch("/collections/"+name, body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Collection **%s** updated.", name), nil
}

func toolDeleteCollection(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("collection_name required")
	}
	_, err := qdrantDelete("/collections/"+name, nil)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Collection **%s** permanently deleted.", name), nil
}

// ── Aliases ───────────────────────────────────────────────────────────────────

func toolListAliases(collectionName string) (string, error) {
	path := "/aliases"
	if collectionName != "" {
		path = "/collections/" + collectionName + "/aliases"
	}
	data, err := qdrantGet(path)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Aliases\n\n```json\n%s\n```", formatJSON(getResult(data))), nil
}

func toolCreateAlias(collectionName, aliasName string) (string, error) {
	if collectionName == "" || aliasName == "" {
		return "", fmt.Errorf("collection_name and alias_name required")
	}
	body := map[string]any{
		"actions": []map[string]any{
			{"create_alias": map[string]any{
				"collection_name": collectionName,
				"alias_name":      aliasName,
			}},
		},
	}
	_, err := qdrantPost("/aliases", body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Alias **%s** → **%s** created.", aliasName, collectionName), nil
}

func toolDeleteAlias(aliasName string) (string, error) {
	if aliasName == "" {
		return "", fmt.Errorf("alias_name required")
	}
	body := map[string]any{
		"actions": []map[string]any{
			{"delete_alias": map[string]any{"alias_name": aliasName}},
		},
	}
	_, err := qdrantPost("/aliases", body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Alias **%s** deleted.", aliasName), nil
}

func toolRenameAlias(oldName, newName string) (string, error) {
	if oldName == "" || newName == "" {
		return "", fmt.Errorf("old_alias_name and new_alias_name required")
	}
	body := map[string]any{
		"actions": []map[string]any{
			{"rename_alias": map[string]any{
				"old_alias_name": oldName,
				"new_alias_name": newName,
			}},
		},
	}
	_, err := qdrantPost("/aliases", body)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Alias renamed **%s** → **%s**.", oldName, newName), nil
}
