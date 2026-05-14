package main

import "fmt"

// allTools returns every tool definition exposed by this MCP server.
func allTools() []ToolDef {
	return []ToolDef{

		// ── Collections ──────────────────────────────────────────────────────

		{Name: "qdrant_list_collections", Description: "List all Qdrant collections with point counts.",
			InputSchema: InputSchema{Type: "object", Properties: map[string]Property{}}},

		{Name: "qdrant_get_collection", Description: "Get full configuration and stats for a single collection.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
				}}},

		{Name: "qdrant_collection_exists", Description: "Check whether a collection exists.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
				}}},

		{Name: "qdrant_create_collection",
			Description: "Create a new Qdrant collection with full config options.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "vector_size"},
				Properties: map[string]Property{
					"collection_name":    prop("string", "Name for the new collection"),
					"vector_size":        prop("integer", "Vector dimension (e.g. 1536, 3072, 768)"),
					"distance":           enumProp("Distance metric (default: Cosine)", "Cosine", "Euclid", "Dot", "Manhattan"),
					"on_disk":            prop("boolean", "Store vectors on disk to reduce RAM usage"),
					"on_disk_payload":    prop("boolean", "Store payload on disk"),
					"replication_factor": prop("integer", "Replication factor for distributed mode"),
					"hnsw_config":        prop("object", "HNSW index config: {m, ef_construct, full_scan_threshold}"),
					"optimizers_config":  prop("object", "Optimizer config: {indexing_threshold, memmap_threshold}"),
				}}},

		{Name: "qdrant_update_collection",
			Description: "Update optimizer, HNSW, or params config on an existing collection.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name":   prop("string", "Collection name"),
					"optimizers_config": prop("object", "Optimizer config: {indexing_threshold, memmap_threshold, etc.}"),
					"hnsw_config":       prop("object", "HNSW config: {m, ef_construct, full_scan_threshold}"),
					"params":            prop("object", "Collection params: {replication_factor, write_consistency_factor}"),
				}}},

		{Name: "qdrant_delete_collection",
			Description: "PERMANENTLY DELETE a collection and all its data. IRREVERSIBLE.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to DELETE permanently"),
				}}},

		{Name: "qdrant_list_aliases",
			Description: "List all collection aliases, optionally scoped to one collection.",
			InputSchema: InputSchema{Type: "object",
				Properties: map[string]Property{
					"collection_name": prop("string", "Scope to this collection (optional — omit for all aliases)"),
				}}},

		{Name: "qdrant_create_alias",
			Description: "Create an alias pointing to a collection.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "alias_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Target collection"),
					"alias_name":      prop("string", "Alias name to create"),
				}}},

		{Name: "qdrant_delete_alias",
			Description: "Delete a collection alias.",
			InputSchema: InputSchema{Type: "object", Required: []string{"alias_name"},
				Properties: map[string]Property{
					"alias_name": prop("string", "Alias to delete"),
				}}},

		{Name: "qdrant_rename_alias",
			Description: "Rename a collection alias atomically.",
			InputSchema: InputSchema{Type: "object", Required: []string{"old_alias_name", "new_alias_name"},
				Properties: map[string]Property{
					"old_alias_name": prop("string", "Current alias name"),
					"new_alias_name": prop("string", "New alias name"),
				}}},

		// ── Points ───────────────────────────────────────────────────────────

		{Name: "qdrant_upsert_points",
			Description: "Insert or update points with vectors and payload. Each point: {id, vector, payload}.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "points"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Target collection"),
					"points":          arrProp("object", "Array of {id, vector, payload} objects"),
					"wait":            prop("boolean", "Wait for indexing (default: true)"),
					"ordering":        enumProp("Write ordering guarantee", "weak", "medium", "strong"),
				}}},

		{Name: "qdrant_get_point",
			Description: "Retrieve a single point by ID.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "point_id"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"point_id":        prop("string", "Point ID (numeric string or UUID)"),
					"with_payload":    prop("boolean", "Include payload (default: true)"),
					"with_vector":     prop("boolean", "Include vector (default: false)"),
				}}},

		{Name: "qdrant_get_points",
			Description: "Retrieve multiple specific points by ID.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "point_ids"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"point_ids":       arrProp("string", "Array of point IDs"),
					"with_payload":    prop("boolean", "Include payloads (default: true)"),
					"with_vector":     prop("boolean", "Include vectors (default: false)"),
				}}},

		{Name: "qdrant_scroll_points",
			Description: "Paginate through all points in a collection with optional filtering.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to scroll"),
					"limit":           prop("integer", "Points per page (default: 10, max: 100)"),
					"offset":          prop("string", "Pagination token from previous response"),
					"with_payload":    prop("boolean", "Include payload (default: true)"),
					"with_vector":     prop("boolean", "Include vectors (default: false)"),
					"filter":          prop("object", `Qdrant filter DSL e.g. {"must":[{"key":"field","match":{"value":"x"}}]}`),
					"order_by":        prop("string", "Payload field to order results by"),
				}}},

		{Name: "qdrant_count_points",
			Description: "Count points in a collection, optionally scoped to a filter.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to count"),
					"filter":          prop("object", "Qdrant filter DSL (optional)"),
					"exact":           prop("boolean", "Exact count vs approximate (default: true)"),
				}}},

		{Name: "qdrant_delete_points",
			Description: "Delete points by ID list or filter.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"point_ids":       arrProp("string", "Point IDs to delete (provide this OR filter)"),
					"filter":          prop("object", "Qdrant filter DSL — delete all matching (provide this OR point_ids)"),
				}}},

		{Name: "qdrant_update_vectors",
			Description: "Update vectors for existing points without touching payload.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "points"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"points":          arrProp("object", "Array of {id, vector} objects"),
				}}},

		{Name: "qdrant_delete_vectors",
			Description: "Delete specific named vectors from points (leaves point and payload intact).",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "point_ids", "vector_names"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"point_ids":       arrProp("string", "Point IDs to update"),
					"vector_names":    arrProp("string", "Named vector fields to delete"),
				}}},

		// ── Payload ──────────────────────────────────────────────────────────

		{Name: "qdrant_set_payload",
			Description: "Merge payload fields onto points (existing fields not in payload are preserved).",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "payload"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"payload":         prop("object", "Key-value pairs to merge into existing payload"),
					"point_ids":       arrProp("string", "Target point IDs (provide this OR filter)"),
					"filter":          prop("object", "Qdrant filter DSL (provide this OR point_ids)"),
				}}},

		{Name: "qdrant_overwrite_payload",
			Description: "Replace the entire payload on points (all existing payload fields are removed).",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "payload"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"payload":         prop("object", "New payload — replaces all existing fields"),
					"point_ids":       arrProp("string", "Target point IDs (provide this OR filter)"),
					"filter":          prop("object", "Qdrant filter DSL (provide this OR point_ids)"),
				}}},

		{Name: "qdrant_delete_payload",
			Description: "Remove specific payload keys from points.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "keys"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"keys":            arrProp("string", "Payload key names to remove"),
					"point_ids":       arrProp("string", "Target point IDs (provide this OR filter)"),
					"filter":          prop("object", "Qdrant filter DSL (provide this OR point_ids)"),
				}}},

		{Name: "qdrant_clear_payload",
			Description: "Remove ALL payload from points (vectors are preserved).",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"point_ids":       arrProp("string", "Target point IDs (provide this OR filter)"),
					"filter":          prop("object", "Qdrant filter DSL (provide this OR point_ids)"),
				}}},

		// ── Search ───────────────────────────────────────────────────────────

		{Name: "qdrant_search",
			Description: "Vector similarity search. Requires a query_vector matching the collection's vector size.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "query_vector"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to search"),
					"query_vector":    arrProp("number", "Query vector — float array matching collection's vector size"),
					"vector_name":     prop("string", "Named vector to search (for multi-vector collections)"),
					"limit":           prop("integer", "Number of results (default: 10, max: 100)"),
					"score_threshold": prop("number", "Minimum similarity score (0–1 for Cosine)"),
					"filter":          prop("object", "Qdrant filter DSL (optional)"),
					"with_payload":    prop("boolean", "Include payloads (default: true)"),
					"with_vector":     prop("boolean", "Include vectors (default: false)"),
					"params":          prop("object", "Search params: {hnsw_ef, exact, quantization}"),
				}}},

		{Name: "qdrant_search_batch",
			Description: "Run multiple vector searches in a single request.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "searches"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to search"),
					"searches":        arrProp("object", "Array of search request objects (same shape as qdrant_search params)"),
				}}},

		{Name: "qdrant_recommend",
			Description: "Find points similar to positive examples and dissimilar from negative examples.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "positive_ids"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to search"),
					"positive_ids":    arrProp("string", "Point IDs to find similar results to"),
					"negative_ids":    arrProp("string", "Point IDs to steer away from (optional)"),
					"limit":           prop("integer", "Number of results (default: 10)"),
					"filter":          prop("object", "Qdrant filter DSL (optional)"),
					"with_payload":    prop("boolean", "Include payloads (default: true)"),
					"strategy":        enumProp("Recommendation strategy", "average_vector", "best_score"),
				}}},

		{Name: "qdrant_discover",
			Description: "Discover points using context pairs (positive/negative examples without an anchor).",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "context"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to search"),
					"context":         arrProp("object", "Context pairs: [{positive: id, negative: id}, ...]"),
					"target":          prop("string", "Optional anchor point ID"),
					"limit":           prop("integer", "Number of results (default: 10)"),
					"filter":          prop("object", "Qdrant filter DSL (optional)"),
					"with_payload":    prop("boolean", "Include payloads (default: true)"),
				}}},

		{Name: "qdrant_query",
			Description: "Unified query — filter, optional vector, ordering, and pagination in one call.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to query"),
					"query_vector":    arrProp("number", "Optional query vector for similarity ranking"),
					"fusion":          enumProp("Fusion strategy for hybrid search", "rrf", "dbsf"),
					"filter":          prop("object", "Qdrant filter DSL"),
					"limit":           prop("integer", "Max results (default: 10, max: 100)"),
					"offset":          prop("string", "Pagination token"),
					"with_payload":    prop("boolean", "Include payloads (default: true)"),
					"with_vector":     prop("boolean", "Include vectors (default: false)"),
					"order_by":        prop("string", "Payload field to sort results by"),
				}}},

		{Name: "qdrant_query_batch",
			Description: "Run multiple query requests in a single call.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "searches"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection to query"),
					"searches":        arrProp("object", "Array of query request objects"),
				}}},

		// ── Indexes ──────────────────────────────────────────────────────────

		{Name: "qdrant_list_indexes",
			Description: "List field indexes for a collection.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
				}}},

		{Name: "qdrant_create_index",
			Description: "Create a payload field index to speed up filtered searches.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "field_name", "field_schema"},
				Properties: map[string]Property{
					"collection_name":   prop("string", "Collection name"),
					"field_name":        prop("string", "Payload field to index"),
					"field_schema":      enumProp("Index type", "keyword", "integer", "float", "bool", "geo", "text", "datetime"),
					"text_index_params": prop("object", "Text index params: {tokenizer, min_token_len, max_token_len} (text schema only)"),
				}}},

		{Name: "qdrant_delete_index",
			Description: "Delete a payload field index.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name", "field_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
					"field_name":      prop("string", "Field index to delete"),
				}}},

		// ── Snapshots ────────────────────────────────────────────────────────

		{Name: "qdrant_list_snapshots",
			Description: "List snapshots for a collection or globally.",
			InputSchema: InputSchema{Type: "object",
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name (omit for global snapshots)"),
				}}},

		{Name: "qdrant_create_snapshot",
			Description: "Create a snapshot of a collection or the full instance.",
			InputSchema: InputSchema{Type: "object",
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name (omit for global snapshot)"),
				}}},

		{Name: "qdrant_delete_snapshot",
			Description: "Delete a named snapshot.",
			InputSchema: InputSchema{Type: "object", Required: []string{"snapshot_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name (omit for global snapshot)"),
					"snapshot_name":   prop("string", "Snapshot filename to delete"),
				}}},

		// ── Cluster & Health ─────────────────────────────────────────────────

		{Name: "qdrant_health",
			Description: "Check Qdrant health and version.",
			InputSchema: InputSchema{Type: "object", Properties: map[string]Property{}}},

		{Name: "qdrant_telemetry",
			Description: "Get Qdrant telemetry data (requests, timings, memory usage).",
			InputSchema: InputSchema{Type: "object", Properties: map[string]Property{}}},

		{Name: "qdrant_cluster_info",
			Description: "Get cluster status and peer information.",
			InputSchema: InputSchema{Type: "object", Properties: map[string]Property{}}},

		{Name: "qdrant_collection_cluster_info",
			Description: "Get cluster distribution info for a specific collection.",
			InputSchema: InputSchema{Type: "object", Required: []string{"collection_name"},
				Properties: map[string]Property{
					"collection_name": prop("string", "Collection name"),
				}}},
	}
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

func dispatch(name string, args map[string]any) (string, error) {
	switch name {

	// Collections
	case "qdrant_list_collections":
		return toolListCollections()
	case "qdrant_get_collection":
		return toolGetCollection(strArg(args, "collection_name"))
	case "qdrant_collection_exists":
		return toolCollectionExists(strArg(args, "collection_name"))
	case "qdrant_create_collection":
		return toolCreateCollection(args)
	case "qdrant_update_collection":
		return toolUpdateCollection(args)
	case "qdrant_delete_collection":
		return toolDeleteCollection(strArg(args, "collection_name"))
	case "qdrant_list_aliases":
		return toolListAliases(strArg(args, "collection_name"))
	case "qdrant_create_alias":
		return toolCreateAlias(strArg(args, "collection_name"), strArg(args, "alias_name"))
	case "qdrant_delete_alias":
		return toolDeleteAlias(strArg(args, "alias_name"))
	case "qdrant_rename_alias":
		return toolRenameAlias(strArg(args, "old_alias_name"), strArg(args, "new_alias_name"))

	// Points
	case "qdrant_upsert_points":
		return toolUpsertPoints(args)
	case "qdrant_get_point":
		return toolGetPoint(strArg(args, "collection_name"), strArg(args, "point_id"),
			boolArg(args, "with_payload", true), boolArg(args, "with_vector", false))
	case "qdrant_get_points":
		return toolGetPoints(args)
	case "qdrant_scroll_points":
		return toolScrollPoints(args)
	case "qdrant_count_points":
		return toolCountPoints(args)
	case "qdrant_delete_points":
		return toolDeletePoints(args)
	case "qdrant_update_vectors":
		return toolUpdateVectors(args)
	case "qdrant_delete_vectors":
		return toolDeleteVectors(args)

	// Payload
	case "qdrant_set_payload":
		return toolSetPayload(args)
	case "qdrant_overwrite_payload":
		return toolOverwritePayload(args)
	case "qdrant_delete_payload":
		return toolDeletePayload(args)
	case "qdrant_clear_payload":
		return toolClearPayload(args)

	// Search
	case "qdrant_search":
		return toolSearch(args)
	case "qdrant_search_batch":
		return toolSearchBatch(args)
	case "qdrant_recommend":
		return toolRecommend(args)
	case "qdrant_discover":
		return toolDiscover(args)
	case "qdrant_query":
		return toolQuery(args)
	case "qdrant_query_batch":
		return toolQueryBatch(args)

	// Indexes
	case "qdrant_list_indexes":
		return toolListIndexes(strArg(args, "collection_name"))
	case "qdrant_create_index":
		return toolCreateIndex(args)
	case "qdrant_delete_index":
		return toolDeleteIndex(strArg(args, "collection_name"), strArg(args, "field_name"))

	// Snapshots
	case "qdrant_list_snapshots":
		return toolListSnapshots(strArg(args, "collection_name"))
	case "qdrant_create_snapshot":
		return toolCreateSnapshot(strArg(args, "collection_name"))
	case "qdrant_delete_snapshot":
		return toolDeleteSnapshot(strArg(args, "collection_name"), strArg(args, "snapshot_name"))

	// Cluster & Health
	case "qdrant_health":
		return toolHealth()
	case "qdrant_telemetry":
		return toolTelemetry()
	case "qdrant_cluster_info":
		return toolClusterInfo()
	case "qdrant_collection_cluster_info":
		return toolCollectionClusterInfo(strArg(args, "collection_name"))

	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}
