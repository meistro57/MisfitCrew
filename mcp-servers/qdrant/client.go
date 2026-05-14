package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

var (
	qdrantURL    = envOrDefault("QDRANT_URL", "http://localhost:6333")
	qdrantAPIKey = os.Getenv("QDRANT_API_KEY")
)

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func qdrantReq(method, path string, body any) (map[string]any, error) {
	var bodyReader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(b)
	}

	req, err := http.NewRequest(method, qdrantURL+path, bodyReader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if qdrantAPIKey != "" {
		req.Header.Set("api-key", qdrantAPIKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant unreachable at %s: %w", qdrantURL, err)
	}
	defer resp.Body.Close()

	rb, _ := io.ReadAll(resp.Body)
	var result map[string]any
	if err := json.Unmarshal(rb, &result); err != nil {
		return nil, fmt.Errorf("qdrant response parse error: %w (body: %s)", err, string(rb))
	}
	if resp.StatusCode >= 400 {
		if errMsg := qdrantErrMsg(result); errMsg != "" {
			return nil, fmt.Errorf("qdrant %s %s: %s", method, path, errMsg)
		}
		return nil, fmt.Errorf("qdrant %s %s: HTTP %d", method, path, resp.StatusCode)
	}
	return result, nil
}

func qdrantGet(path string) (map[string]any, error) {
	return qdrantReq("GET", path, nil)
}

func qdrantPost(path string, body any) (map[string]any, error) {
	return qdrantReq("POST", path, body)
}

func qdrantPut(path string, body any) (map[string]any, error) {
	return qdrantReq("PUT", path, body)
}

func qdrantPatch(path string, body any) (map[string]any, error) {
	return qdrantReq("PATCH", path, body)
}

func qdrantDelete(path string, body any) (map[string]any, error) {
	return qdrantReq("DELETE", path, body)
}

func qdrantErrMsg(data map[string]any) string {
	if status, ok := data["status"].(map[string]any); ok {
		if errStr, ok := status["error"].(string); ok && errStr != "" {
			return errStr
		}
	}
	if errStr, ok := data["error"].(string); ok && errStr != "" {
		return errStr
	}
	return ""
}

func getResult(data map[string]any) any {
	if r, ok := data["result"]; ok {
		return r
	}
	return data
}
