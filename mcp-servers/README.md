# mcp-servers

MCP server binaries for Qdrant and Redis.

## Servers

### qdrant-mcp
Qdrant REST coverage includes:
- **Collections**: list/get/create/update/delete/exists + alias operations
- **Points**: upsert/get/get-batch/scroll/count/delete/update-vectors/delete-vectors
- **Payload**: set/overwrite/delete-keys/clear
- **Search**: search/search-batch/recommend/discover/query/query-batch
- **Indexes**: list/create/delete field indexes
- **Snapshots**: list/create/delete
- **Infra**: health/telemetry/cluster info

### redis-mcp
Redis coverage includes:
- **Strings/Keys**: get/set/del/exists/expire/ttl/incr/decr/mget/mset/type/rename/persist/scan
- **Hashes**: hget/hset/hmget/hgetall/hdel/hkeys/hvals/hlen/hexists/hincrby
- **Lists**: lpush/rpush/lpop/rpop/lrange/llen/lindex/lset/lrem
- **Sets**: sadd/srem/smembers/sismember/scard/sunion/sinter/sdiff
- **Sorted Sets**: zadd/zrem/zscore/zrank/zrange/zrangebyscore/zcard/zincrby
- **Server**: ping/info/dbsize/keys/flushdb(confirm=true)/scan

## Build

```bash
cd /home/mark/MisfitCrew/mcp-servers
make all
```

Build individually:

```bash
make qdrant
make redis
```

Clean binaries:

```bash
make clean
```

If local Go is older than `go.mod` requires, run with toolchain auto-download:

```bash
GOTOOLCHAIN=auto make all
```

## Runtime environment

### qdrant-mcp
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### redis-mcp
```env
REDIS_ADDR=localhost:6379
REDIS_PASSWORD=
REDIS_DB=0
```

## Client config example

Add entries to your MCP client config pointing to these binaries:

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "wsl",
      "args": ["-e", "/home/mark/MisfitCrew/mcp-servers/qdrant/qdrant-mcp"]
    },
    "redis": {
      "command": "wsl",
      "args": ["-e", "/home/mark/MisfitCrew/mcp-servers/redis/redis-mcp"]
    }
  }
}
```
