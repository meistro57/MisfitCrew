/*
redis-mcp — Full Redis MCP server for Claude Desktop.

Exposes Redis commands as MCP tools across six categories:
  - Strings / Keys  (get, set, del, exists, expire, ttl, incr, mget, mset, type, scan, rename)
  - Hashes          (hget, hset, hmget, hgetall, hdel, hkeys, hvals, hlen, hexists, hincrby)
  - Lists           (lpush, rpush, lpop, rpop, lrange, llen, lindex, lset, lrem)
  - Sets            (sadd, srem, smembers, sismember, scard, sunion, sinter, sdiff)
  - Sorted Sets     (zadd, zrem, zscore, zrank, zrange, zrangebyscore, zcard, zincrby)
  - Server          (info, dbsize, ping, flushdb, keys, type, dump)

Usage:
  redis-mcp                         # connects via stdio (MCP transport)

Environment:
  REDIS_ADDR      default localhost:6379
  REDIS_PASSWORD  optional
  REDIS_DB        default 0
*/
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/redis/go-redis/v9"
)

var version = "1.0.0"

var rdb *redis.Client
var ctx = context.Background()

// ── MCP protocol ──────────────────────────────────────────────────────────────

type JSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id"`
	Result  any       `json:"result,omitempty"`
	Error   *RPCError `json:"error,omitempty"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func errResponse(id any, code int, msg string) JSONRPCResponse {
	return JSONRPCResponse{JSONRPC: "2.0", ID: id, Error: &RPCError{Code: code, Message: msg}}
}

func okResponse(id any, result any) JSONRPCResponse {
	return JSONRPCResponse{JSONRPC: "2.0", ID: id, Result: result}
}

// ── Tool definitions ──────────────────────────────────────────────────────────

type ToolDef struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema InputSchema `json:"inputSchema"`
}

type InputSchema struct {
	Type       string              `json:"type"`
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required,omitempty"`
}

type Property struct {
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Items       *Property `json:"items,omitempty"`
	Enum        []string  `json:"enum,omitempty"`
}

func prop(t, desc string) Property           { return Property{Type: t, Description: desc} }
func arrProp(it, desc string) Property       { return Property{Type: "array", Description: desc, Items: &Property{Type: it}} }

func allTools() []ToolDef {
	return []ToolDef{
		// ── Strings / Keys ──────────────────────────────────────────────────
		{Name: "redis_get", Description: "Get the value of a key.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Redis key")}}},

		{Name: "redis_set", Description: "Set a key to a value with optional TTL and NX/XX flags.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "value"},
				Properties: map[string]Property{
					"key":     prop("string", "Redis key"),
					"value":   prop("string", "Value to set"),
					"ex":      prop("integer", "Expire in seconds"),
					"px":      prop("integer", "Expire in milliseconds"),
					"nx":      prop("boolean", "Only set if key does NOT exist"),
					"xx":      prop("boolean", "Only set if key DOES exist"),
					"keepttl": prop("boolean", "Retain existing TTL"),
				}}},

		{Name: "redis_del", Description: "Delete one or more keys.",
			InputSchema: InputSchema{Type: "object", Required: []string{"keys"},
				Properties: map[string]Property{"keys": arrProp("string", "Keys to delete")}}},

		{Name: "redis_exists", Description: "Check if one or more keys exist. Returns count of existing keys.",
			InputSchema: InputSchema{Type: "object", Required: []string{"keys"},
				Properties: map[string]Property{"keys": arrProp("string", "Keys to check")}}},

		{Name: "redis_expire", Description: "Set a TTL on a key in seconds.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "seconds"},
				Properties: map[string]Property{
					"key":     prop("string", "Redis key"),
					"seconds": prop("integer", "TTL in seconds"),
				}}},

		{Name: "redis_ttl", Description: "Get the TTL of a key in seconds. Returns -1 if no TTL, -2 if key doesn't exist.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Redis key")}}},

		{Name: "redis_incr", Description: "Increment a key's integer value by an amount (default: 1).",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{
					"key": prop("string", "Redis key"),
					"by":  prop("integer", "Amount to increment by (default: 1)"),
				}}},

		{Name: "redis_decr", Description: "Decrement a key's integer value by an amount (default: 1).",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{
					"key": prop("string", "Redis key"),
					"by":  prop("integer", "Amount to decrement by (default: 1)"),
				}}},

		{Name: "redis_mget", Description: "Get values of multiple keys at once.",
			InputSchema: InputSchema{Type: "object", Required: []string{"keys"},
				Properties: map[string]Property{"keys": arrProp("string", "Keys to retrieve")}}},

		{Name: "redis_mset", Description: "Set multiple key-value pairs atomically.",
			InputSchema: InputSchema{Type: "object", Required: []string{"pairs"},
				Properties: map[string]Property{"pairs": prop("object", "Object of {key: value} pairs to set")}}},

		{Name: "redis_type", Description: "Get the data type of a key.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Redis key")}}},

		{Name: "redis_rename", Description: "Rename a key.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "new_key"},
				Properties: map[string]Property{
					"key":     prop("string", "Current key name"),
					"new_key": prop("string", "New key name"),
				}}},

		{Name: "redis_persist", Description: "Remove the TTL from a key (make it persistent).",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Redis key")}}},

		{Name: "redis_scan", Description: "Scan keyspace with cursor-based iteration and optional pattern/count.",
			InputSchema: InputSchema{Type: "object",
				Properties: map[string]Property{
					"pattern": prop("string", "Key pattern (default: *)"),
					"count":   prop("integer", "Hint for keys per scan iteration (default: 100)"),
					"cursor":  prop("integer", "Cursor from previous scan (0 to start)"),
				}}},

		// ── Hashes ──────────────────────────────────────────────────────────
		{Name: "redis_hget", Description: "Get a field value from a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "field"},
				Properties: map[string]Property{
					"key":   prop("string", "Hash key"),
					"field": prop("string", "Field name"),
				}}},

		{Name: "redis_hset", Description: "Set one or more fields on a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "fields"},
				Properties: map[string]Property{
					"key":    prop("string", "Hash key"),
					"fields": prop("object", "Object of {field: value} pairs"),
				}}},

		{Name: "redis_hmget", Description: "Get multiple field values from a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "fields"},
				Properties: map[string]Property{
					"key":    prop("string", "Hash key"),
					"fields": arrProp("string", "Field names to retrieve"),
				}}},

		{Name: "redis_hgetall", Description: "Get all field-value pairs from a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Hash key")}}},

		{Name: "redis_hdel", Description: "Delete one or more fields from a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "fields"},
				Properties: map[string]Property{
					"key":    prop("string", "Hash key"),
					"fields": arrProp("string", "Fields to delete"),
				}}},

		{Name: "redis_hkeys", Description: "Get all field names from a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Hash key")}}},

		{Name: "redis_hvals", Description: "Get all values from a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Hash key")}}},

		{Name: "redis_hlen", Description: "Get the number of fields in a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Hash key")}}},

		{Name: "redis_hexists", Description: "Check if a field exists in a hash.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "field"},
				Properties: map[string]Property{
					"key":   prop("string", "Hash key"),
					"field": prop("string", "Field name"),
				}}},

		{Name: "redis_hincrby", Description: "Increment a hash field's integer value.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "field", "increment"},
				Properties: map[string]Property{
					"key":       prop("string", "Hash key"),
					"field":     prop("string", "Field name"),
					"increment": prop("integer", "Amount to increment by"),
				}}},

		// ── Lists ────────────────────────────────────────────────────────────
		{Name: "redis_lpush", Description: "Prepend values to a list (left push). Returns new list length.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "values"},
				Properties: map[string]Property{
					"key":    prop("string", "List key"),
					"values": arrProp("string", "Values to prepend"),
				}}},

		{Name: "redis_rpush", Description: "Append values to a list (right push). Returns new list length.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "values"},
				Properties: map[string]Property{
					"key":    prop("string", "List key"),
					"values": arrProp("string", "Values to append"),
				}}},

		{Name: "redis_lpop", Description: "Remove and return element(s) from the left of a list.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{
					"key":   prop("string", "List key"),
					"count": prop("integer", "Number of elements to pop (default: 1)"),
				}}},

		{Name: "redis_rpop", Description: "Remove and return element(s) from the right of a list.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{
					"key":   prop("string", "List key"),
					"count": prop("integer", "Number of elements to pop (default: 1)"),
				}}},

		{Name: "redis_lrange", Description: "Get a range of elements from a list (0-based, -1 = last).",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "start", "stop"},
				Properties: map[string]Property{
					"key":   prop("string", "List key"),
					"start": prop("integer", "Start index"),
					"stop":  prop("integer", "Stop index (-1 for last element)"),
				}}},

		{Name: "redis_llen", Description: "Get the length of a list.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "List key")}}},

		{Name: "redis_lindex", Description: "Get an element from a list by index.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "index"},
				Properties: map[string]Property{
					"key":   prop("string", "List key"),
					"index": prop("integer", "Element index (0-based, negative counts from end)"),
				}}},

		{Name: "redis_lset", Description: "Set a list element at a given index.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "index", "value"},
				Properties: map[string]Property{
					"key":   prop("string", "List key"),
					"index": prop("integer", "Element index"),
					"value": prop("string", "New value"),
				}}},

		{Name: "redis_lrem", Description: "Remove occurrences of a value from a list.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "count", "value"},
				Properties: map[string]Property{
					"key":   prop("string", "List key"),
					"count": prop("integer", "Number to remove (0=all, >0=from head, <0=from tail)"),
					"value": prop("string", "Value to remove"),
				}}},

		// ── Sets ─────────────────────────────────────────────────────────────
		{Name: "redis_sadd", Description: "Add members to a set. Returns count of new members added.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "members"},
				Properties: map[string]Property{
					"key":     prop("string", "Set key"),
					"members": arrProp("string", "Members to add"),
				}}},

		{Name: "redis_srem", Description: "Remove members from a set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "members"},
				Properties: map[string]Property{
					"key":     prop("string", "Set key"),
					"members": arrProp("string", "Members to remove"),
				}}},

		{Name: "redis_smembers", Description: "Get all members of a set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Set key")}}},

		{Name: "redis_sismember", Description: "Check if a value is a member of a set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "member"},
				Properties: map[string]Property{
					"key":    prop("string", "Set key"),
					"member": prop("string", "Value to check"),
				}}},

		{Name: "redis_scard", Description: "Get the number of members in a set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Set key")}}},

		{Name: "redis_sunion", Description: "Get the union of multiple sets.",
			InputSchema: InputSchema{Type: "object", Required: []string{"keys"},
				Properties: map[string]Property{"keys": arrProp("string", "Set keys to union")}}},

		{Name: "redis_sinter", Description: "Get the intersection of multiple sets.",
			InputSchema: InputSchema{Type: "object", Required: []string{"keys"},
				Properties: map[string]Property{"keys": arrProp("string", "Set keys to intersect")}}},

		{Name: "redis_sdiff", Description: "Get the difference between sets (first key minus all others).",
			InputSchema: InputSchema{Type: "object", Required: []string{"keys"},
				Properties: map[string]Property{"keys": arrProp("string", "Set keys (first minus rest)")}}},

		// ── Sorted Sets ──────────────────────────────────────────────────────
		{Name: "redis_zadd", Description: "Add members with scores to a sorted set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "members"},
				Properties: map[string]Property{
					"key":     prop("string", "Sorted set key"),
					"members": arrProp("object", "Array of {score: float, member: string} objects"),
					"nx":      prop("boolean", "Only add new members, don't update existing"),
					"xx":      prop("boolean", "Only update existing members, don't add new"),
					"gt":      prop("boolean", "Only update if new score > current"),
					"lt":      prop("boolean", "Only update if new score < current"),
				}}},

		{Name: "redis_zrem", Description: "Remove members from a sorted set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "members"},
				Properties: map[string]Property{
					"key":     prop("string", "Sorted set key"),
					"members": arrProp("string", "Members to remove"),
				}}},

		{Name: "redis_zscore", Description: "Get the score of a member in a sorted set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "member"},
				Properties: map[string]Property{
					"key":    prop("string", "Sorted set key"),
					"member": prop("string", "Member name"),
				}}},

		{Name: "redis_zrank", Description: "Get the rank of a member in a sorted set (0-based, ascending).",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "member"},
				Properties: map[string]Property{
					"key":    prop("string", "Sorted set key"),
					"member": prop("string", "Member name"),
					"rev":    prop("boolean", "Use reverse rank (highest score = rank 0)"),
				}}},

		{Name: "redis_zrange", Description: "Get members from a sorted set by rank range with optional scores.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "start", "stop"},
				Properties: map[string]Property{
					"key":        prop("string", "Sorted set key"),
					"start":      prop("integer", "Start rank"),
					"stop":       prop("integer", "Stop rank (-1 for last)"),
					"with_scores": prop("boolean", "Include scores in results"),
					"rev":        prop("boolean", "Reverse order (highest first)"),
				}}},

		{Name: "redis_zrangebyscore", Description: "Get members from a sorted set by score range.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "min", "max"},
				Properties: map[string]Property{
					"key":         prop("string", "Sorted set key"),
					"min":         prop("string", "Min score ('-inf' for unbounded)"),
					"max":         prop("string", "Max score ('+inf' for unbounded)"),
					"with_scores": prop("boolean", "Include scores in results"),
					"limit":       prop("integer", "Max results"),
					"offset":      prop("integer", "Skip this many results"),
				}}},

		{Name: "redis_zcard", Description: "Get the number of members in a sorted set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key"},
				Properties: map[string]Property{"key": prop("string", "Sorted set key")}}},

		{Name: "redis_zincrby", Description: "Increment a member's score in a sorted set.",
			InputSchema: InputSchema{Type: "object", Required: []string{"key", "increment", "member"},
				Properties: map[string]Property{
					"key":       prop("string", "Sorted set key"),
					"increment": prop("number", "Amount to add to current score"),
					"member":    prop("string", "Member name"),
				}}},

		// ── Server ───────────────────────────────────────────────────────────
		{Name: "redis_ping", Description: "Ping the Redis server. Returns PONG if alive.",
			InputSchema: InputSchema{Type: "object", Properties: map[string]Property{}}},

		{Name: "redis_info", Description: "Get Redis server info, optionally filtered to a section.",
			InputSchema: InputSchema{Type: "object",
				Properties: map[string]Property{
					"section": prop("string", "Info section: server, clients, memory, stats, replication, cpu, keyspace (omit for all)"),
				}}},

		{Name: "redis_dbsize", Description: "Get the number of keys in the current database.",
			InputSchema: InputSchema{Type: "object", Properties: map[string]Property{}}},

		{Name: "redis_keys", Description: "Find all keys matching a pattern. Use redis_scan for large keyspaces.",
			InputSchema: InputSchema{Type: "object", Required: []string{"pattern"},
				Properties: map[string]Property{
					"pattern": prop("string", "Key pattern (e.g. 'user:*', '*session*')"),
				}}},

		{Name: "redis_flushdb", Description: "Delete ALL keys from the current database. REQUIRES confirm=true.",
			InputSchema: InputSchema{Type: "object", Required: []string{"confirm"},
				Properties: map[string]Property{
					"confirm": prop("boolean", "Must be true to execute — safety check"),
					"async":   prop("boolean", "Flush asynchronously (default: false)"),
				}}},
	}
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

func strArg(args map[string]any, key string) string {
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func intArg(args map[string]any, key string, def int) int {
	if v, ok := args[key]; ok {
		if f, ok := v.(float64); ok {
			return int(f)
		}
	}
	return def
}

func boolArg(args map[string]any, key string, def bool) bool {
	if v, ok := args[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return def
}

func strSliceArg(args map[string]any, key string) []string {
	v, ok := args[key]
	if !ok {
		return nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func dispatch(name string, args map[string]any) (string, error) {
	switch name {
	case "redis_get":
		return redisGet(strArg(args, "key"))
	case "redis_set":
		return redisSet(args)
	case "redis_del":
		return redisDel(strSliceArg(args, "keys"))
	case "redis_exists":
		return redisExists(strSliceArg(args, "keys"))
	case "redis_expire":
		return redisExpire(strArg(args, "key"), intArg(args, "seconds", 0))
	case "redis_ttl":
		return redisTTL(strArg(args, "key"))
	case "redis_incr":
		return redisIncr(strArg(args, "key"), intArg(args, "by", 1))
	case "redis_decr":
		return redisDecr(strArg(args, "key"), intArg(args, "by", 1))
	case "redis_mget":
		return redisMGet(strSliceArg(args, "keys"))
	case "redis_mset":
		return redisMSet(args)
	case "redis_type":
		return redisType(strArg(args, "key"))
	case "redis_rename":
		return redisRename(strArg(args, "key"), strArg(args, "new_key"))
	case "redis_persist":
		return redisPersist(strArg(args, "key"))
	case "redis_scan":
		return redisScan(strArg(args, "pattern"), intArg(args, "count", 100), uint64(intArg(args, "cursor", 0)))
	case "redis_hget":
		return redisHGet(strArg(args, "key"), strArg(args, "field"))
	case "redis_hset":
		return redisHSet(strArg(args, "key"), args)
	case "redis_hmget":
		return redisHMGet(strArg(args, "key"), strSliceArg(args, "fields"))
	case "redis_hgetall":
		return redisHGetAll(strArg(args, "key"))
	case "redis_hdel":
		return redisHDel(strArg(args, "key"), strSliceArg(args, "fields"))
	case "redis_hkeys":
		return redisHKeys(strArg(args, "key"))
	case "redis_hvals":
		return redisHVals(strArg(args, "key"))
	case "redis_hlen":
		return redisHLen(strArg(args, "key"))
	case "redis_hexists":
		return redisHExists(strArg(args, "key"), strArg(args, "field"))
	case "redis_hincrby":
		return redisHIncrBy(strArg(args, "key"), strArg(args, "field"), int64(intArg(args, "increment", 1)))
	case "redis_lpush":
		return redisLPush(strArg(args, "key"), strSliceArg(args, "values"))
	case "redis_rpush":
		return redisRPush(strArg(args, "key"), strSliceArg(args, "values"))
	case "redis_lpop":
		return redisLPop(strArg(args, "key"), intArg(args, "count", 1))
	case "redis_rpop":
		return redisRPop(strArg(args, "key"), intArg(args, "count", 1))
	case "redis_lrange":
		return redisLRange(strArg(args, "key"), int64(intArg(args, "start", 0)), int64(intArg(args, "stop", -1)))
	case "redis_llen":
		return redisLLen(strArg(args, "key"))
	case "redis_lindex":
		return redisLIndex(strArg(args, "key"), int64(intArg(args, "index", 0)))
	case "redis_lset":
		return redisLSet(strArg(args, "key"), int64(intArg(args, "index", 0)), strArg(args, "value"))
	case "redis_lrem":
		return redisLRem(strArg(args, "key"), int64(intArg(args, "count", 0)), strArg(args, "value"))
	case "redis_sadd":
		return redisSAdd(strArg(args, "key"), strSliceArg(args, "members"))
	case "redis_srem":
		return redisSRem(strArg(args, "key"), strSliceArg(args, "members"))
	case "redis_smembers":
		return redisSMembers(strArg(args, "key"))
	case "redis_sismember":
		return redisSIsMember(strArg(args, "key"), strArg(args, "member"))
	case "redis_scard":
		return redisSCard(strArg(args, "key"))
	case "redis_sunion":
		return redisSUnion(strSliceArg(args, "keys"))
	case "redis_sinter":
		return redisSInter(strSliceArg(args, "keys"))
	case "redis_sdiff":
		return redisSDiff(strSliceArg(args, "keys"))
	case "redis_zadd":
		return redisZAdd(strArg(args, "key"), args)
	case "redis_zrem":
		return redisZRem(strArg(args, "key"), strSliceArg(args, "members"))
	case "redis_zscore":
		return redisZScore(strArg(args, "key"), strArg(args, "member"))
	case "redis_zrank":
		return redisZRank(strArg(args, "key"), strArg(args, "member"), boolArg(args, "rev", false))
	case "redis_zrange":
		return redisZRange(args)
	case "redis_zrangebyscore":
		return redisZRangeByScore(args)
	case "redis_zcard":
		return redisZCard(strArg(args, "key"))
	case "redis_zincrby":
		inc, _ := args["increment"].(float64)
		return redisZIncrBy(strArg(args, "key"), inc, strArg(args, "member"))
	case "redis_ping":
		return redisPing()
	case "redis_info":
		return redisInfo(strArg(args, "section"))
	case "redis_dbsize":
		return redisDBSize()
	case "redis_keys":
		return redisKeys(strArg(args, "pattern"))
	case "redis_flushdb":
		return redisFlushDB(boolArg(args, "confirm", false), boolArg(args, "async", false))
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

// ── Tool implementations ──────────────────────────────────────────────────────

func redisGet(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	val, err := rdb.Get(ctx, key).Result()
	if err == redis.Nil {
		return fmt.Sprintf("Key `%s` does not exist.", key), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` = `%s`", key, val), nil
}

func redisSet(args map[string]any) (string, error) {
	key := strArg(args, "key")
	value := strArg(args, "value")
	if key == "" || value == "" {
		return "", fmt.Errorf("key and value required")
	}
	opts := &redis.SetArgs{Mode: ""}
	if ex := intArg(args, "ex", 0); ex > 0 {
		opts.TTL = time.Duration(ex) * time.Second
	}
	if px := intArg(args, "px", 0); px > 0 {
		opts.TTL = time.Duration(px) * time.Millisecond
	}
	if boolArg(args, "nx", false) {
		opts.Mode = "NX"
	}
	if boolArg(args, "xx", false) {
		opts.Mode = "XX"
	}
	if boolArg(args, "keepttl", false) {
		opts.KeepTTL = true
	}
	err := rdb.SetArgs(ctx, key, value, *opts).Err()
	if err == redis.Nil {
		return fmt.Sprintf("Set skipped (NX/XX condition not met) for `%s`.", key), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ SET `%s` = `%s`", key, value), nil
}

func redisDel(keys []string) (string, error) {
	if len(keys) == 0 {
		return "", fmt.Errorf("keys required")
	}
	n, err := rdb.Del(ctx, keys...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Deleted **%d** key(s).", n), nil
}

func redisExists(keys []string) (string, error) {
	if len(keys) == 0 {
		return "", fmt.Errorf("keys required")
	}
	n, err := rdb.Exists(ctx, keys...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("**%d** of %d key(s) exist.", n, len(keys)), nil
}

func redisExpire(key string, seconds int) (string, error) {
	if key == "" || seconds == 0 {
		return "", fmt.Errorf("key and seconds required")
	}
	ok, err := rdb.Expire(ctx, key, time.Duration(seconds)*time.Second).Result()
	if err != nil {
		return "", err
	}
	if !ok {
		return fmt.Sprintf("Key `%s` does not exist.", key), nil
	}
	return fmt.Sprintf("✅ TTL set on `%s`: %ds", key, seconds), nil
}

func redisTTL(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	ttl, err := rdb.TTL(ctx, key).Result()
	if err != nil {
		return "", err
	}
	switch ttl {
	case -2 * time.Second:
		return fmt.Sprintf("`%s` does not exist.", key), nil
	case -1 * time.Second:
		return fmt.Sprintf("`%s` has no TTL (persistent).", key), nil
	default:
		return fmt.Sprintf("`%s` TTL: **%s** (%.0f seconds)", key, ttl, ttl.Seconds()), nil
	}
}

func redisIncr(key string, by int) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	var n int64
	var err error
	if by == 1 {
		n, err = rdb.Incr(ctx, key).Result()
	} else {
		n, err = rdb.IncrBy(ctx, key, int64(by)).Result()
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` = **%d**", key, n), nil
}

func redisDecr(key string, by int) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	var n int64
	var err error
	if by == 1 {
		n, err = rdb.Decr(ctx, key).Result()
	} else {
		n, err = rdb.DecrBy(ctx, key, int64(by)).Result()
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` = **%d**", key, n), nil
}

func redisMGet(keys []string) (string, error) {
	if len(keys) == 0 {
		return "", fmt.Errorf("keys required")
	}
	vals, err := rdb.MGet(ctx, keys...).Result()
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for i, v := range vals {
		if v == nil {
			sb.WriteString(fmt.Sprintf("`%s` = (nil)\n", keys[i]))
		} else {
			sb.WriteString(fmt.Sprintf("`%s` = `%v`\n", keys[i], v))
		}
	}
	return sb.String(), nil
}

func redisMSet(args map[string]any) (string, error) {
	pairs, ok := args["pairs"].(map[string]any)
	if !ok || len(pairs) == 0 {
		return "", fmt.Errorf("pairs object required")
	}
	kv := make([]any, 0, len(pairs)*2)
	for k, v := range pairs {
		kv = append(kv, k, fmt.Sprintf("%v", v))
	}
	err := rdb.MSet(ctx, kv...).Err()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ MSET — set %d keys.", len(pairs)), nil
}

func redisType(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	t, err := rdb.Type(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` type: **%s**", key, t), nil
}

func redisRename(key, newKey string) (string, error) {
	if key == "" || newKey == "" {
		return "", fmt.Errorf("key and new_key required")
	}
	err := rdb.Rename(ctx, key, newKey).Err()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ Renamed `%s` → `%s`", key, newKey), nil
}

func redisPersist(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	ok, err := rdb.Persist(ctx, key).Result()
	if err != nil {
		return "", err
	}
	if !ok {
		return fmt.Sprintf("`%s` has no TTL or does not exist.", key), nil
	}
	return fmt.Sprintf("✅ TTL removed from `%s` (now persistent).", key), nil
}

func redisScan(pattern string, count int, cursor uint64) (string, error) {
	if pattern == "" {
		pattern = "*"
	}
	keys, nextCursor, err := rdb.Scan(ctx, cursor, pattern, int64(count)).Result()
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Scan — pattern: `%s`, cursor: %d → %d\n\n", pattern, cursor, nextCursor))
	sb.WriteString(fmt.Sprintf("**%d** keys returned\n\n", len(keys)))
	for _, k := range keys {
		sb.WriteString(fmt.Sprintf("- `%s`\n", k))
	}
	if nextCursor != 0 {
		sb.WriteString(fmt.Sprintf("\n> Next cursor: `%d`", nextCursor))
	} else {
		sb.WriteString("\n> Scan complete.")
	}
	return sb.String(), nil
}

// ── Hash implementations ──────────────────────────────────────────────────────

func redisHGet(key, field string) (string, error) {
	if key == "" || field == "" {
		return "", fmt.Errorf("key and field required")
	}
	val, err := rdb.HGet(ctx, key, field).Result()
	if err == redis.Nil {
		return fmt.Sprintf("`%s`.`%s` does not exist.", key, field), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s`.`%s` = `%s`", key, field, val), nil
}

func redisHSet(key string, args map[string]any) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	fields, ok := args["fields"].(map[string]any)
	if !ok || len(fields) == 0 {
		return "", fmt.Errorf("fields object required")
	}
	kv := make([]any, 0, len(fields)*2)
	for f, v := range fields {
		kv = append(kv, f, fmt.Sprintf("%v", v))
	}
	n, err := rdb.HSet(ctx, key, kv...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ HSET `%s` — %d new fields set.", key, n), nil
}

func redisHMGet(key string, fields []string) (string, error) {
	if key == "" || len(fields) == 0 {
		return "", fmt.Errorf("key and fields required")
	}
	vals, err := rdb.HMGet(ctx, key, fields...).Result()
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for i, v := range vals {
		if v == nil {
			sb.WriteString(fmt.Sprintf("`%s` = (nil)\n", fields[i]))
		} else {
			sb.WriteString(fmt.Sprintf("`%s` = `%v`\n", fields[i], v))
		}
	}
	return sb.String(), nil
}

func redisHGetAll(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	m, err := rdb.HGetAll(ctx, key).Result()
	if err != nil {
		return "", err
	}
	if len(m) == 0 {
		return fmt.Sprintf("Hash `%s` is empty or does not exist.", key), nil
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Hash: `%s` (%d fields)\n\n", key, len(m)))
	for f, v := range m {
		sb.WriteString(fmt.Sprintf("- **%s**: `%s`\n", f, v))
	}
	return sb.String(), nil
}

func redisHDel(key string, fields []string) (string, error) {
	if key == "" || len(fields) == 0 {
		return "", fmt.Errorf("key and fields required")
	}
	n, err := rdb.HDel(ctx, key, fields...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Deleted %d field(s) from `%s`.", n, key), nil
}

func redisHKeys(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	keys, err := rdb.HKeys(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## HKEYS `%s`\n\n%s", key, strings.Join(keys, "\n")), nil
}

func redisHVals(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	vals, err := rdb.HVals(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## HVALS `%s`\n\n%s", key, strings.Join(vals, "\n")), nil
}

func redisHLen(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	n, err := rdb.HLen(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` has **%d** fields.", key, n), nil
}

func redisHExists(key, field string) (string, error) {
	if key == "" || field == "" {
		return "", fmt.Errorf("key and field required")
	}
	ok, err := rdb.HExists(ctx, key, field).Result()
	if err != nil {
		return "", err
	}
	if ok {
		return fmt.Sprintf("✅ `%s`.`%s` exists.", key, field), nil
	}
	return fmt.Sprintf("❌ `%s`.`%s` does not exist.", key, field), nil
}

func redisHIncrBy(key, field string, inc int64) (string, error) {
	if key == "" || field == "" {
		return "", fmt.Errorf("key and field required")
	}
	n, err := rdb.HIncrBy(ctx, key, field, inc).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s`.`%s` = **%d**", key, field, n), nil
}

// ── List implementations ──────────────────────────────────────────────────────

func redisLPush(key string, vals []string) (string, error) {
	if key == "" || len(vals) == 0 {
		return "", fmt.Errorf("key and values required")
	}
	ivals := make([]any, len(vals))
	for i, v := range vals {
		ivals[i] = v
	}
	n, err := rdb.LPush(ctx, key, ivals...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ LPUSH `%s` — list length: **%d**", key, n), nil
}

func redisRPush(key string, vals []string) (string, error) {
	if key == "" || len(vals) == 0 {
		return "", fmt.Errorf("key and values required")
	}
	ivals := make([]any, len(vals))
	for i, v := range vals {
		ivals[i] = v
	}
	n, err := rdb.RPush(ctx, key, ivals...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ RPUSH `%s` — list length: **%d**", key, n), nil
}

func redisLPop(key string, count int) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	vals, err := rdb.LPopCount(ctx, key, count).Result()
	if err == redis.Nil {
		return fmt.Sprintf("List `%s` is empty or does not exist.", key), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("LPOP `%s`: %s", key, strings.Join(vals, ", ")), nil
}

func redisRPop(key string, count int) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	vals, err := rdb.RPopCount(ctx, key, count).Result()
	if err == redis.Nil {
		return fmt.Sprintf("List `%s` is empty or does not exist.", key), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("RPOP `%s`: %s", key, strings.Join(vals, ", ")), nil
}

func redisLRange(key string, start, stop int64) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	vals, err := rdb.LRange(ctx, key, start, stop).Result()
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## LRANGE `%s` [%d:%d] — %d elements\n\n", key, start, stop, len(vals)))
	for i, v := range vals {
		sb.WriteString(fmt.Sprintf("%d. `%s`\n", i, v))
	}
	return sb.String(), nil
}

func redisLLen(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	n, err := rdb.LLen(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` list length: **%d**", key, n), nil
}

func redisLIndex(key string, index int64) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	val, err := rdb.LIndex(ctx, key, index).Result()
	if err == redis.Nil {
		return fmt.Sprintf("Index %d out of range for `%s`.", index, key), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s`[%d] = `%s`", key, index, val), nil
}

func redisLSet(key string, index int64, value string) (string, error) {
	if key == "" || value == "" {
		return "", fmt.Errorf("key, index, and value required")
	}
	err := rdb.LSet(ctx, key, index, value).Err()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ LSET `%s`[%d] = `%s`", key, index, value), nil
}

func redisLRem(key string, count int64, value string) (string, error) {
	if key == "" || value == "" {
		return "", fmt.Errorf("key, count, and value required")
	}
	n, err := rdb.LRem(ctx, key, count, value).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Removed **%d** occurrence(s) of `%s` from `%s`.", n, value, key), nil
}

// ── Set implementations ───────────────────────────────────────────────────────

func redisSAdd(key string, members []string) (string, error) {
	if key == "" || len(members) == 0 {
		return "", fmt.Errorf("key and members required")
	}
	ivals := make([]any, len(members))
	for i, m := range members {
		ivals[i] = m
	}
	n, err := rdb.SAdd(ctx, key, ivals...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ SADD `%s` — %d new members added.", key, n), nil
}

func redisSRem(key string, members []string) (string, error) {
	if key == "" || len(members) == 0 {
		return "", fmt.Errorf("key and members required")
	}
	ivals := make([]any, len(members))
	for i, m := range members {
		ivals[i] = m
	}
	n, err := rdb.SRem(ctx, key, ivals...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Removed %d member(s) from `%s`.", n, key), nil
}

func redisSMembers(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	members, err := rdb.SMembers(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## SMEMBERS `%s` (%d)\n\n%s", key, len(members), strings.Join(members, "\n")), nil
}

func redisSIsMember(key, member string) (string, error) {
	if key == "" || member == "" {
		return "", fmt.Errorf("key and member required")
	}
	ok, err := rdb.SIsMember(ctx, key, member).Result()
	if err != nil {
		return "", err
	}
	if ok {
		return fmt.Sprintf("✅ `%s` IS a member of `%s`.", member, key), nil
	}
	return fmt.Sprintf("❌ `%s` is NOT a member of `%s`.", member, key), nil
}

func redisSCard(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	n, err := rdb.SCard(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` set cardinality: **%d**", key, n), nil
}

func redisSUnion(keys []string) (string, error) {
	if len(keys) == 0 {
		return "", fmt.Errorf("keys required")
	}
	members, err := rdb.SUnion(ctx, keys...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## SUNION (%d members)\n\n%s", len(members), strings.Join(members, "\n")), nil
}

func redisSInter(keys []string) (string, error) {
	if len(keys) == 0 {
		return "", fmt.Errorf("keys required")
	}
	members, err := rdb.SInter(ctx, keys...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## SINTER (%d members)\n\n%s", len(members), strings.Join(members, "\n")), nil
}

func redisSDiff(keys []string) (string, error) {
	if len(keys) == 0 {
		return "", fmt.Errorf("keys required")
	}
	members, err := rdb.SDiff(ctx, keys...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## SDIFF (%d members)\n\n%s", len(members), strings.Join(members, "\n")), nil
}

// ── Sorted set implementations ────────────────────────────────────────────────

func redisZAdd(key string, args map[string]any) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	membersRaw, ok := args["members"].([]any)
	if !ok || len(membersRaw) == 0 {
		return "", fmt.Errorf("members array required: [{score: float, member: string}]")
	}

	zargs := redis.ZAddArgs{
		NX: boolArg(args, "nx", false),
		XX: boolArg(args, "xx", false),
		GT: boolArg(args, "gt", false),
		LT: boolArg(args, "lt", false),
	}
	zargs.Members = make([]redis.Z, 0, len(membersRaw))
	for _, raw := range membersRaw {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		score, _ := m["score"].(float64)
		member, _ := m["member"].(string)
		zargs.Members = append(zargs.Members, redis.Z{Score: score, Member: member})
	}

	n, err := rdb.ZAddArgs(ctx, key, zargs).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("✅ ZADD `%s` — %d members added/updated.", key, n), nil
}

func redisZRem(key string, members []string) (string, error) {
	if key == "" || len(members) == 0 {
		return "", fmt.Errorf("key and members required")
	}
	ivals := make([]any, len(members))
	for i, m := range members {
		ivals[i] = m
	}
	n, err := rdb.ZRem(ctx, key, ivals...).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("🗑️ Removed %d member(s) from `%s`.", n, key), nil
}

func redisZScore(key, member string) (string, error) {
	if key == "" || member == "" {
		return "", fmt.Errorf("key and member required")
	}
	score, err := rdb.ZScore(ctx, key, member).Result()
	if err == redis.Nil {
		return fmt.Sprintf("`%s` not in `%s`.", member, key), nil
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s`.`%s` score: **%g**", key, member, score), nil
}

func redisZRank(key, member string, rev bool) (string, error) {
	if key == "" || member == "" {
		return "", fmt.Errorf("key and member required")
	}
	var rank int64
	var err error
	if rev {
		rank, err = rdb.ZRevRank(ctx, key, member).Result()
	} else {
		rank, err = rdb.ZRank(ctx, key, member).Result()
	}
	if err == redis.Nil {
		return fmt.Sprintf("`%s` not in `%s`.", member, key), nil
	}
	if err != nil {
		return "", err
	}
	direction := "asc"
	if rev {
		direction = "desc"
	}
	return fmt.Sprintf("`%s`.`%s` rank (%s): **%d**", key, member, direction, rank), nil
}

func redisZRange(args map[string]any) (string, error) {
	key := strArg(args, "key")
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	start := int64(intArg(args, "start", 0))
	stop := int64(intArg(args, "stop", -1))
	withScores := boolArg(args, "with_scores", false)
	rev := boolArg(args, "rev", false)

	rangeArgs := redis.ZRangeArgs{
		Key:     key,
		Start:   start,
		Stop:    stop,
		Rev:     rev,
	}

	if withScores {
		vals, err := rdb.ZRangeArgsWithScores(ctx, rangeArgs).Result()
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		for _, z := range vals {
			sb.WriteString(fmt.Sprintf("- `%v` (score: %g)\n", z.Member, z.Score))
		}
		return fmt.Sprintf("## ZRANGE `%s` [%d:%d] — %d members\n\n%s", key, start, stop, len(vals), sb.String()), nil
	}

	vals, err := rdb.ZRangeArgs(ctx, rangeArgs).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## ZRANGE `%s` [%d:%d] — %d members\n\n%s", key, start, stop, len(vals), strings.Join(vals, "\n")), nil
}

func redisZRangeByScore(args map[string]any) (string, error) {
	key := strArg(args, "key")
	min := strArg(args, "min")
	max := strArg(args, "max")
	if key == "" || min == "" || max == "" {
		return "", fmt.Errorf("key, min, and max required")
	}
	withScores := boolArg(args, "with_scores", false)
	limit := intArg(args, "limit", 0)
	offset := intArg(args, "offset", 0)

	opt := &redis.ZRangeBy{Min: min, Max: max}
	if limit > 0 {
		opt.Offset = int64(offset)
		opt.Count = int64(limit)
	}

	if withScores {
		vals, err := rdb.ZRangeByScoreWithScores(ctx, key, opt).Result()
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		for _, z := range vals {
			sb.WriteString(fmt.Sprintf("- `%v` (score: %g)\n", z.Member, z.Score))
		}
		return fmt.Sprintf("## ZRANGEBYSCORE `%s` [%s:%s] — %d members\n\n%s", key, min, max, len(vals), sb.String()), nil
	}

	vals, err := rdb.ZRangeByScore(ctx, key, opt).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## ZRANGEBYSCORE `%s` [%s:%s] — %d members\n\n%s", key, min, max, len(vals), strings.Join(vals, "\n")), nil
}

func redisZCard(key string) (string, error) {
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	n, err := rdb.ZCard(ctx, key).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s` sorted set cardinality: **%d**", key, n), nil
}

func redisZIncrBy(key string, inc float64, member string) (string, error) {
	if key == "" || member == "" {
		return "", fmt.Errorf("key and member required")
	}
	score, err := rdb.ZIncrBy(ctx, key, inc, member).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("`%s`.`%s` new score: **%g**", key, member, score), nil
}

// ── Server implementations ────────────────────────────────────────────────────

func redisPing() (string, error) {
	result, err := rdb.Ping(ctx).Result()
	if err != nil {
		return "", fmt.Errorf("Redis unreachable: %w", err)
	}
	return fmt.Sprintf("**%s** — Redis is alive.", result), nil
}

func redisInfo(section string) (string, error) {
	var result string
	var err error
	if section != "" {
		result, err = rdb.Info(ctx, section).Result()
	} else {
		result, err = rdb.Info(ctx).Result()
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("## Redis INFO%s\n\n```\n%s\n```", func() string {
		if section != "" {
			return " — " + section
		}
		return ""
	}(), result), nil
}

func redisDBSize() (string, error) {
	n, err := rdb.DBSize(ctx).Result()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("Database has **%d** keys.", n), nil
}

func redisKeys(pattern string) (string, error) {
	if pattern == "" {
		return "", fmt.Errorf("pattern required (use * for all)")
	}
	keys, err := rdb.Keys(ctx, pattern).Result()
	if err != nil {
		return "", err
	}
	if len(keys) == 0 {
		return fmt.Sprintf("No keys matching `%s`.", pattern), nil
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Keys matching `%s` — **%d** found\n\n", pattern, len(keys)))
	for _, k := range keys {
		sb.WriteString(fmt.Sprintf("- `%s`\n", k))
	}
	return sb.String(), nil
}

func redisFlushDB(confirm, async bool) (string, error) {
	if !confirm {
		return "", fmt.Errorf("set confirm=true to flush the database — this deletes ALL keys")
	}
	var err error
	if async {
		err = rdb.FlushDBAsync(ctx).Err()
	} else {
		err = rdb.FlushDB(ctx).Err()
	}
	if err != nil {
		return "", err
	}
	return "🗑️ Database flushed — all keys deleted.", nil
}

// ── Request handler ───────────────────────────────────────────────────────────

func handleRequest(req JSONRPCRequest) JSONRPCResponse {
	switch req.Method {
	case "initialize":
		return okResponse(req.ID, map[string]any{
			"protocolVersion": "2025-11-25",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": "redis_mcp", "version": version},
		})

	case "tools/list":
		return okResponse(req.ID, map[string]any{"tools": allTools()})

	case "tools/call":
		var params struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &params); err != nil {
			return errResponse(req.ID, -32600, "invalid params")
		}
		result, err := dispatch(params.Name, params.Arguments)
		if err != nil {
			return okResponse(req.ID, map[string]any{
				"content": []map[string]any{{"type": "text", "text": "Error: " + err.Error()}},
				"isError": true,
			})
		}
		return okResponse(req.ID, map[string]any{
			"content": []map[string]any{{"type": "text", "text": result}},
		})

	case "notifications/initialized":
		return JSONRPCResponse{}

	default:
		return errResponse(req.ID, -32601, "method not found: "+req.Method)
	}
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	_ = godotenv.Load()

	addr := envOrDefault("REDIS_ADDR", "localhost:6379")
	password := os.Getenv("REDIS_PASSWORD")
	dbNum, _ := strconv.Atoi(os.Getenv("REDIS_DB"))

	rdb = redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       dbNum,
	})

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	encoder := json.NewEncoder(os.Stdout)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		var req JSONRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			encoder.Encode(errResponse(nil, -32700, "parse error"))
			continue
		}
		resp := handleRequest(req)
		if req.Method == "notifications/initialized" {
			continue
		}
		encoder.Encode(resp)
	}
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
