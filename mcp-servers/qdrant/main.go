/*
qdrant-mcp — Full Qdrant MCP server for Claude Desktop.

Exposes the complete Qdrant REST API as MCP tools across six categories:
  - Collections  (list, get, create, update, delete, exists, aliases)
  - Points       (upsert, get, scroll, count, delete, vectors)
  - Payload      (set, overwrite, delete, clear)
  - Search       (vector search, batch search, recommend, discover, query)
  - Indexes      (list, create, delete field indexes)
  - Snapshots    (list, create, delete)
  - Cluster      (health, telemetry, cluster info)

Usage:
  qdrant-mcp                         # connects via stdio (MCP transport)

Environment:
  QDRANT_URL      default http://localhost:6333
  QDRANT_API_KEY  optional, for secured deployments
*/
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

var version = "1.0.0"

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

// ── Tool definition helpers ───────────────────────────────────────────────────

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

func prop(t, desc string) Property             { return Property{Type: t, Description: desc} }
func arrProp(itemType, desc string) Property   { return Property{Type: "array", Description: desc, Items: &Property{Type: itemType}} }
func enumProp(desc string, vals ...string) Property {
	return Property{Type: "string", Description: desc, Enum: vals}
}

// ── Request handler ───────────────────────────────────────────────────────────

func handleRequest(req JSONRPCRequest) JSONRPCResponse {
	switch req.Method {
	case "initialize":
		return okResponse(req.ID, map[string]any{
			"protocolVersion": "2025-11-25",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": "qdrant_mcp", "version": version},
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

// ── Helpers ───────────────────────────────────────────────────────────────────

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

func mapArg(args map[string]any, key string) map[string]any {
	if v, ok := args[key]; ok {
		if m, ok := v.(map[string]any); ok {
			return m
		}
	}
	return nil
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

func float32SliceArg(args map[string]any, key string) []float32 {
	v, ok := args[key]
	if !ok {
		return nil
	}
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]float32, len(arr))
	for i, item := range arr {
		if f, ok := item.(float64); ok {
			out[i] = float32(f)
		}
	}
	return out
}

func formatJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}
