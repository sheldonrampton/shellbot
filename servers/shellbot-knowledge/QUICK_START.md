# Shellbot MCP Server - Quick Start Guide

## What You've Got

I've created an MCP server that exposes your shellbot knowledge base to any LLM that supports the Model Context Protocol (MCP). This means instead of being locked into OpenAI's API, any compatible LLM client can now access your contextual knowledge.

## Files Created

1. **mcp_server.py** - The main MCP server implementation
2. **test_mcp_server.py** - Test script to verify everything works
3. **MCP_SERVER_README.md** - Complete documentation
4. **claude_desktop_config.json.example** - Example configuration for Claude Desktop
5. **requirements.txt** - Updated with MCP dependency

## Quick Test

To verify the server works with your existing database:

```bash
# Install the MCP SDK
pip install mcp

# Run the test script
python test_mcp_server.py
```

This will test:
- ✓ Connection to your PostgreSQL database
- ✓ Connection to Pinecone vector database
- ✓ Search functionality
- ✓ Context retrieval
- ✓ Statistics gathering

## How It Works

### Three Main Tools:

1. **search_knowledge** - Search your knowledge base semantically
   - Returns formatted results with content, dates, platforms, and relevance scores
   - Perfect for exploring what's in the knowledge base

2. **get_context** - Get contextual information for LLM prompts
   - Optimized for providing background information to LLMs
   - Returns formatted text chunks ready to augment prompts

3. **get_knowledge_stats** - Get stats about your knowledge base
   - Total entries, platforms, date ranges, vector database info

### The Flow:

```
User asks question → MCP Client → Your MCP Server → 
  → Searches Pinecone (vector similarity) →
  → Retrieves from PostgreSQL (full content) →
  → Returns to MCP Client → 
  → LLM gets context → Answers with your knowledge
```

## Using with Claude Desktop

1. Edit `claude_desktop_config.json.example` with your actual credentials
2. Copy it to Claude Desktop's config location:
   ```bash
   # macOS
   cp claude_desktop_config.json.example \
     ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
3. Restart Claude Desktop
4. You'll see "shellbot-knowledge" available as a tool

## Using with Other MCP Clients

Any MCP-compatible client can use your server. Popular options:
- Claude Desktop (Anthropic)
- Continue (VS Code extension)
- Custom implementations using the MCP SDK

## Environment Variables Needed

```bash
# OpenAI (for embeddings)
export OPENAI_API_KEY="sk-..."

# Pinecone (for vector search)
export PINECONE_API_KEY="..."

# PostgreSQL (for content storage)
export DB_HOST="your-host"
export DB_PORT="5432"
export SHELLBOT_DB_NAME="shellbot_knowledge"
export SHELLBOT_USER="your-user"
export SHELLBOT_USER_PASSWORD="your-password"
```

## Example Usage

Once connected to an MCP client (like Claude Desktop), you can:

**Ask questions:**
```
"What has Sheldon written about tennis?"
```

The LLM will:
1. Use the `get_context` tool to search your knowledge base
2. Get relevant posts/emails about tennis
3. Answer based on that actual content

**Search explicitly:**
```
"Use the search_knowledge tool to find posts about AI from 2024"
```

**Get stats:**
```
"How many entries are in the knowledge base?"
```

## Advantages Over Your Current Setup

### Before (Flask app with OpenAI):
- ✗ Locked into OpenAI API
- ✗ Requires custom integration for each LLM
- ✗ Context limited to your Flask app

### Now (MCP Server):
- ✓ Works with any MCP-compatible LLM
- ✓ Standard protocol (MCP)
- ✓ Your knowledge accessible everywhere
- ✓ Can use local models, Claude, GPT-4, etc.
- ✓ Client handles conversation, server provides knowledge

## Next Steps

1. **Test it:** Run `python test_mcp_server.py`
2. **Configure a client:** Try with Claude Desktop or another MCP client
3. **Use it:** Ask questions and see your knowledge in action!
4. **Extend it:** Add more tools as needed (see MCP_SERVER_README.md)

## Troubleshooting

**"No module named 'mcp'"**
```bash
pip install mcp
```

**"Connection refused" or database errors**
- Check your environment variables
- Verify PostgreSQL is running
- Test Pinecone API key

**Server starts but no results**
- Verify your `shellbot_knowledge` database has data
- Check Pinecone index name matches ("shellbot-embeddings2")
- Use `get_knowledge_stats` to verify contents

## Architecture Comparison

**Old (Flask RAG app):**
```
User → Flask → OpenAI → Pinecone/PostgreSQL → OpenAI → Response
```

**New (MCP Server):**
```
User → Any LLM Client → MCP Server → Pinecone/PostgreSQL → Client → Response
             ↓
        Claude, GPT-4, Local models, etc.
```

The key difference: Your knowledge is now a **universal resource** accessible by any LLM, not tied to OpenAI.

## Questions?

See `MCP_SERVER_README.md` for complete documentation, or test with:
```bash
python test_mcp_server.py
```
