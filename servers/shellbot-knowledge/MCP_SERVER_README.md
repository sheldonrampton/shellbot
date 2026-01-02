# Shellbot MCP Server

An MCP (Model Context Protocol) server that exposes Sheldon Rampton's knowledge base for use with any LLM client.

## Overview

This MCP server provides access to a knowledge base containing:
- Social media posts (Facebook, Twitter)
- Email messages
- Other writings by Sheldon Rampton

The content is stored in PostgreSQL and indexed in a Pinecone vector database for semantic search.

It uses the social_data.py script from Shellbot 1.0, lightly edited
to remove dependencies on chatbotter.py's write functionality.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORG_ID="your-org-id"  # Optional, defaults to existing value
export OPENAI_PROJECT_ID="your-project-id"  # Optional, defaults to existing value

# Database configuration
export DB_HOST="your-postgres-host"
export DB_PORT="5432"
export SHELLBOT_DB_NAME="your-database-name"
export SHELLBOT_USER="your-database-user"
export SHELLBOT_USER_PASSWORD="your-database-password"

# Pinecone configuration
export PINECONE_API_KEY="your-pinecone-api-key"
```

## Running the Server

### Standalone Mode
```bash
python mcp_server.py
```

### With MCP Client (e.g., Claude Desktop)

Add to your MCP client configuration:

**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
```json
{
  "mcpServers": {
    "shellbot-knowledge": {
      "command": "python",
      "args": ["/path/to/shellbot/mcp_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-api-key",
        "PINECONE_API_KEY": "your-pinecone-key",
        "DB_HOST": "your-db-host",
        "DB_PORT": "5432",
        "SHELLBOT_DB_NAME": "your-db-name",
        "SHELLBOT_USER": "your-db-user",
        "SHELLBOT_USER_PASSWORD": "your-db-password"
      }
    }
  }
}
```

**For other MCP clients**, consult their documentation for adding custom MCP servers.

## Available Tools

### 1. `search_knowledge`
Search the knowledge base using semantic similarity.

**Parameters:**
- `query` (required): The search query
- `top_k` (optional): Number of results to return (1-100, default: 5)

**Example:**
```json
{
  "query": "artificial intelligence projects",
  "top_k": 10
}
```

**Returns:** Formatted results with content, title, date/platform, URL/participants, and relevance scores.

### 2. `get_context`
Get contextual information formatted for LLM prompts.

**Parameters:**
- `query` (required): The question or topic to get context for
- `max_chunks` (optional): Maximum number of context chunks (1-20, default: 5)

**Example:**
```json
{
  "query": "What has Sheldon written about tennis?",
  "max_chunks": 5
}
```

**Returns:** Formatted context chunks suitable for augmenting an LLM prompt.

### 3. `get_knowledge_stats`
Get statistics about the knowledge base.

**Parameters:** None

**Returns:** JSON with total entries, platforms covered, date range, and vector database stats.

## Resources

### `shellbot://knowledge`
Metadata about the knowledge base including:
- Description
- Platforms covered
- Embedding model used
- Database information
- Usage instructions

## Architecture

```
┌─────────────────────┐
│   MCP Client        │
│  (Claude, etc.)     │
└──────────┬──────────┘
           │ MCP Protocol
           │
┌──────────▼──────────┐
│   MCP Server        │
│  (mcp_server.py)    │
└──────────┬──────────┘
           │
           ├─────────────────┐
           │                 │
┌──────────▼──────────┐  ┌──▼─────────────┐
│   PostgreSQL        │  │   Pinecone     │
│  (Knowledge Base)   │  │ (Vector Index) │
└─────────────────────┘  └────────────────┘
```

## Development

### Testing the Server

You can test individual tools using the MCP inspector or by creating a simple test script:

```python
import asyncio
from mcp_server import get_storage

async def test():
    sd = get_storage()
    results = sd.get_pinecone_matches("tennis", top_n=3)
    print(results)

asyncio.run(test())
```

### Extending the Server

To add new tools, follow this pattern in `mcp_server.py`:

1. Add the tool definition in `handle_list_tools()`
2. Add the implementation in `handle_call_tool()`
3. Update this README with documentation

## Troubleshooting

### Connection Issues
- Verify all environment variables are set correctly
- Check database and Pinecone connectivity
- Ensure OpenAI API key has access to embeddings API

### No Results Returned
- The knowledge base may not contain relevant information
- Try different search terms or broader queries
- Use `get_knowledge_stats` to verify the knowledge base contents

### Performance
- Large `top_k` values may slow down searches
- Consider caching frequently accessed content
- Monitor Pinecone usage and rate limits

## License

Same license as the parent shellbot project.

## Contact

For questions or issues, contact the repository maintainer.
