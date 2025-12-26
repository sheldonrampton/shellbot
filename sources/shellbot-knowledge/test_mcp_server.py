#!/usr/bin/env python3
"""
Test script for the Shellbot MCP Server

This script tests the MCP server's tools locally without requiring
a full MCP client connection.
"""

import asyncio
import sys
from mcp_server import handle_call_tool, get_storage


async def test_search_knowledge():
    """Test the search_knowledge tool."""
    print("\n" + "="*80)
    print("Testing search_knowledge tool")
    print("="*80)
    
    result = await handle_call_tool(
        name="search_knowledge",
        arguments={
            "query": "tennis",
            "top_k": 3
        }
    )
    
    print("\nResults:")
    for item in result:
        print(item.text)


async def test_get_context():
    """Test the get_context tool."""
    print("\n" + "="*80)
    print("Testing get_context tool")
    print("="*80)
    
    result = await handle_call_tool(
        name="get_context",
        arguments={
            "query": "What have you been doing with artificial intelligence?",
            "max_chunks": 3
        }
    )
    
    print("\nContext:")
    for item in result:
        print(item.text)


async def test_get_knowledge_stats():
    """Test the get_knowledge_stats tool."""
    print("\n" + "="*80)
    print("Testing get_knowledge_stats tool")
    print("="*80)
    
    result = await handle_call_tool(
        name="get_knowledge_stats",
        arguments={}
    )
    
    print("\nStats:")
    for item in result:
        print(item.text)


async def test_storage_connection():
    """Test basic storage connection."""
    print("\n" + "="*80)
    print("Testing storage connection")
    print("="*80)
    
    try:
        sd = get_storage()
        print(f"✓ Successfully connected to knowledge base")
        print(f"  - Database: {sd.knowledge_db_name}")
        print(f"  - Pinecone Index: {sd.pinecone_index_name}")
        print(f"  - Embedding Model: {sd.embedding_model}")
        
        # Test a simple query
        results = sd.get_pinecone_matches("test query", top_n=1)
        print(f"✓ Successfully queried Pinecone (got {len(results)} result)")
        
    except Exception as e:
        print(f"✗ Error connecting to storage: {e}")
        sys.exit(1)


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SHELLBOT MCP SERVER TEST SUITE")
    print("="*80)
    
    # Test storage connection first
    await test_storage_connection()
    
    # Test each tool
    await test_search_knowledge()
    await test_get_context()
    await test_get_knowledge_stats()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
