.PHONY: all qdrant redis clean

all: qdrant redis

qdrant:
	cd qdrant && go mod tidy && go build -o qdrant-mcp . && echo "✅ qdrant-mcp built"

redis:
	cd redis && go mod tidy && go build -o redis-mcp . && echo "✅ redis-mcp built"

clean:
	rm -f qdrant/qdrant-mcp redis/redis-mcp
