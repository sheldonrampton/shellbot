#!/usr/bin/env python3
"""
MCP Server for Shellbot Knowledge Base

This server exposes Sheldon Rampton's knowledge base (stored in PostgreSQL and 
indexed in Pinecone) via the Model Context Protocol (MCP), allowing any LLM 
to query the knowledge base for context.

Resources:
- shellbot://knowledge - The full knowledge base description

Tools:
- search_knowledge - Search the knowledge base using semantic similarity
- get_context - Get relevant context for a specific query
"""

import asyncio
import os
from typing import Any, Optional
import json

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from openai import OpenAI
from social_data import SocialData

# Initialize server
server = Server("shellbot-knowledge")

# Global storage for the knowledge base
_sd: Optional[SocialData] = None
_openai_client: Optional[OpenAI] = None


def get_storage() -> SocialData:
    """Get or create the SocialData storage instance."""
    global _sd, _openai_client
    
    if _sd is None:
        # Initialize OpenAI client
        _openai_client = OpenAI(
            organization=os.getenv('OPENAI_ORGANIZATION'),
            project=os.getenv('OPENAI_PROJECT'),
        )
        
        # Initialize SocialData with knowledge base
        _sd = SocialData(
            _openai_client,
            knowledge_db_name='shellbot_knowledge',
            pinecone_index_name="shellbot-embeddings2",
        )
        _sd.setup_pinecone()
    
    return _sd


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources (the knowledge base)."""
    return [
        Resource(
            uri="shellbot://knowledge",
            name="Sheldon Rampton's Knowledge Base",
            description="A searchable knowledge base containing Sheldon Rampton's social media posts, emails, and writings",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "shellbot://knowledge":
        return json.dumps({
            "description": "Knowledge base containing social media posts, emails, and writings by Sheldon Rampton",
            "platforms": ["Facebook", "Twitter", "Email"],
            "embedding_model": "text-embedding-3-small",
            "vector_database": "Pinecone",
            "storage": "PostgreSQL",
            "usage": "Use the search_knowledge or get_context tools to query this knowledge base"
        }, indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_knowledge",
            description=(
                "Search Sheldon Rampton's knowledge base using semantic similarity. "
                "Returns relevant posts, emails, and social media content that match the query. "
                "Each result includes the content, title (with date and platform), URL/participants, "
                "and a relevance score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant content"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 100)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_context",
            description=(
                "Get contextual information from the knowledge base to answer a specific question. "
                "This tool searches for relevant content and formats it in a way optimized for "
                "providing context to an LLM. Returns formatted text chunks that can be used "
                "to augment a prompt with relevant background information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to get context for"
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum number of context chunks to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_knowledge_stats",
            description=(
                "Get statistics about the knowledge base, including total number of entries, "
                "platforms covered, date range, and index information."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution."""
    
    try:
        sd = get_storage()
        
        if name == "search_knowledge":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            
            if not query:
                raise ValueError("query parameter is required")
            
            # Search the knowledge base
            results_df = sd.get_pinecone_matches(query, top_n=top_k)
            
            # Format results
            output = []
            output.append(f"Found {len(results_df)} results for query: '{query}'\n")
            
            for idx, row in results_df.iterrows():
                output.append(f"\n{'='*80}")
                output.append(f"Result #{idx + 1} (Score: {row['score']:.4f})")
                output.append(f"Title: {row['title']}")
                if row['url']:
                    output.append(f"URL/Participants: {row['url']}")
                output.append(f"\nContent:\n{row['content']}")
                output.append(f"{'='*80}\n")
            
            return [TextContent(
                type="text",
                text="\n".join(output)
            )]
        
        elif name == "get_context":
            query = arguments.get("query")
            max_chunks = arguments.get("max_chunks", 5)
            
            if not query:
                raise ValueError("query parameter is required")
            
            # Search the knowledge base
            results_df = sd.get_pinecone_matches(query, top_n=max_chunks)
            
            # Format as context chunks
            context_chunks = []
            context_chunks.append(f"Context from Sheldon Rampton's knowledge base for query: '{query}'\n")
            context_chunks.append(f"The following {len(results_df)} messages/posts were written by Sheldon Rampton:\n")
            
            for idx, row in results_df.iterrows():
                context_chunks.append(f"\n--- {row['title']} ---")
                context_chunks.append(row['content'])
            
            return [TextContent(
                type="text",
                text="\n".join(context_chunks)
            )]
        
        elif name == "get_knowledge_stats":
            # Get index stats from Pinecone
            index_stats = sd.pinecone_index.describe_index_stats()
            
            # Get database stats
            conn, cur = sd.database_connection()
            cur.execute(f"SELECT COUNT(*) FROM {sd.knowledge_db_name}")
            total_entries = cur.fetchone()[0]
            
            cur.execute(f"SELECT DISTINCT platform FROM {sd.knowledge_db_name}")
            platforms = [row[0] for row in cur.fetchall()]
            
            cur.execute(f"""
                SELECT 
                    MIN(unix_timestamp) as earliest,
                    MAX(unix_timestamp) as latest
                FROM {sd.knowledge_db_name}
            """)
            date_range = cur.fetchone()
            conn.close()
            
            from datetime import datetime
            earliest_date = datetime.fromtimestamp(date_range[0]).strftime("%B %d, %Y") if date_range[0] else "Unknown"
            latest_date = datetime.fromtimestamp(date_range[1]).strftime("%B %d, %Y") if date_range[1] else "Unknown"
            
            stats = {
                "total_entries": total_entries,
                "platforms": platforms,
                "date_range": {
                    "earliest": earliest_date,
                    "latest": latest_date
                },
                "pinecone_stats": {
                    "total_vector_count": index_stats.total_vector_count,
                    "dimension": index_stats.dimension,
                    "namespaces": dict(index_stats.namespaces)
                },
                "embedding_model": sd.embedding_model
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="shellbot-knowledge",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
