from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
from mem0 import Memory
import asyncio
import json
import os

from utils import get_mem0_client

load_dotenv()

# Default user ID for memory operations. This can be overridden by providing
# a ``user_id`` argument to the tools or by setting the ``DEFAULT_USER_ID``
# environment variable.
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "user")

# Create a dataclass for our application context
@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server."""
    memory_client: Memory
    notes_client: Memory

@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """
    Manages the Mem0 client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Mem0Context: The context containing the Mem0 client
    """
    # Create Memory clients for main conversations and notes
    memory_client = get_mem0_client("mem0_memories")
    notes_client = get_mem0_client("mem0_notes")
    
    try:
        yield Mem0Context(memory_client=memory_client, notes_client=notes_client)
    finally:
        # No explicit cleanup needed for the Mem0 client
        pass

# Initialize FastMCP server with the Mem0 client as context
mcp = FastMCP(
    "mcp-mem0",
    description="MCP server for long term memory storage and retrieval with Mem0",
    lifespan=mem0_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8050")
)        

@mcp.tool()
async def save_memory(
    ctx: Context,
    text: str,
    memory_type: str = "memory",
    user_id: str = DEFAULT_USER_ID,
    message_number: int | None = None,
    date: str | None = None,
) -> str:
    """Save information to your long-term memory.

    This tool is designed to store any type of information that might be useful in the future.
    The content will be processed and indexed for later retrieval through semantic search.

    Args:
        ctx: The MCP server provided context which includes the Mem0 clients
        text: The content to store in memory or notes
        memory_type: Either "memory" or "notes" to select the collection
        user_id: Identifier for the user whose memory should be updated
        message_number: The message number associated with this text
        date: The date the message was sent
    """
    try:
        if memory_type == "notes":
            mem0_client = ctx.request_context.lifespan_context.notes_client
        else:
            mem0_client = ctx.request_context.lifespan_context.memory_client

        metadata = {
            "id": user_id,
            "message_number": message_number,
            "date": date,
        }

        messages = [
            {
                "role": "user",
                "content": text,
                "metadata": metadata,
            }
        ]

        mem0_client.add(messages, user_id=user_id, metadatas=[metadata])
        return f"Successfully saved memory: {text[:100]}..." if len(text) > 100 else f"Successfully saved memory: {text}"
    except Exception as e:
        return f"Error saving memory: {str(e)}"

@mcp.tool()
async def get_all_memories(
    ctx: Context,
    memory_type: str = "memory",
    user_id: str = DEFAULT_USER_ID,
) -> str:
    """Get all stored memories for the user.
    
    Call this tool when you need complete context of all previously memories.

    Args:
        ctx: The MCP server provided context which includes the Mem0 clients
        memory_type: Either "memory" or "notes" to select the collection
        user_id: Identifier for the user whose memories should be returned

    Returns a JSON formatted list of all stored memories, including when they were created
    and their content. Results are paginated with a default of 50 items per page.
    """
    try:
        if memory_type == "notes":
            mem0_client = ctx.request_context.lifespan_context.notes_client
        else:
            mem0_client = ctx.request_context.lifespan_context.memory_client
        memories = mem0_client.get_all(user_id=user_id)
        if isinstance(memories, dict) and "results" in memories:
            formatted = []
            for memory in memories["results"]:
                entry = {
                    "memory": memory.get("memory"),
                    "metadata": memory.get("metadata"),
                }
                formatted.append(entry)
        else:
            formatted = memories
        return json.dumps(formatted, indent=2)
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"

@mcp.tool()
async def search_memories(
    ctx: Context,
    query: str,
    limit: int = 3,
    memory_type: str = "memory",
    user_id: str = DEFAULT_USER_ID,
) -> str:
    """Search memories using semantic search.

    This tool should be called to find relevant information from your memory. Results are ranked by relevance.
    Always search your memories before making decisions to ensure you leverage your existing knowledge.

    Args:
        ctx: The MCP server provided context which includes the Mem0 clients
        query: Search query string describing what you're looking for. Can be natural language.
        limit: Maximum number of results to return (default: 3)
        memory_type: Either "memory" or "notes" to select the collection
        user_id: Identifier for the user whose memories should be searched
    """
    try:
        if memory_type == "notes":
            mem0_client = ctx.request_context.lifespan_context.notes_client
        else:
            mem0_client = ctx.request_context.lifespan_context.memory_client
        memories = mem0_client.search(query, user_id=user_id, limit=limit)
        if isinstance(memories, dict) and "results" in memories:
            formatted = []
            for memory in memories["results"]:
                entry = {
                    "memory": memory.get("memory"),
                    "metadata": memory.get("metadata"),
                }
                formatted.append(entry)
        else:
            formatted = memories
        return json.dumps(formatted, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())
