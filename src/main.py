"""MCP server that connects to the hosted Mem0 Cloud API."""

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import json
import logging
import os

from mem0 import AsyncMemoryClient
from utils import get_mem0_client, close_mem0_client

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default user ID for memory operations. This can be overridden by providing
# a ``user_id`` argument to the tools or by setting the ``DEFAULT_USER_ID``
# environment variable.
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "user")

def get_mem0_client_safe() -> AsyncMemoryClient | None:
    """Create a Mem0 client if possible, returning ``None`` on failure."""
    try:
        return get_mem0_client()
    except Exception as exc:
        logger.warning("Failed to create Mem0 client: %s", exc)
        return None

# Create a dataclass for our application context
@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server."""
    memory_client: AsyncMemoryClient | None

@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """
    Manages the Mem0 client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Mem0Context: The context containing the Mem0 client
    """
    # Create a single Mem0 client lazily with graceful failure handling
    memory_client = get_mem0_client_safe()

    try:
        yield Mem0Context(memory_client=memory_client)
    finally:
        if memory_client is not None:
            await close_mem0_client(memory_client)

# Initialize FastMCP server with the Mem0 client as context
mcp = FastMCP(
    "mcp-mem0",
    description="MCP server for long term memory storage and retrieval with Mem0",
    lifespan=mem0_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8050")
)

@mcp.tool(description="Store information for future reference in Mem0")
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
        ctx: The MCP server provided context which includes the Mem0 client
        text: The content to store in memory or as a note
        memory_type: Either "memory" or "note" to select the category
        user_id: Identifier for the user whose memory should be updated
        message_number: The message number associated with this text
        date: The date the message was sent
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.memory_client
        if mem0_client is None:
            return "Error: Memory service is currently unavailable. Please try again later."

        metadata = {
            "id": user_id,
            "message_number": message_number,
            "date": date,
        }

        messages = [{"role": "user", "content": text}]

        await mem0_client.add(
            messages,
            user_id=user_id,
            category=memory_type,
            metadata=metadata,
        )
        return (
            f"Successfully saved memory: {text[:100]}..."
            if len(text) > 100
            else f"Successfully saved memory: {text}"
        )
    except Exception as e:
        logger.exception("Error saving memory")
        return f"Error saving memory: {str(e)}"

@mcp.tool(description="Retrieve all memories stored for a user")
async def get_all_memories(
    ctx: Context,
    memory_type: str = "memory",
    user_id: str = DEFAULT_USER_ID,
) -> str:
    """Get all stored memories for the user.
    
    Call this tool when you need complete context of all previously memories.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        memory_type: Either "memory" or "note" to select the category
        user_id: Identifier for the user whose memories should be returned

    Returns a JSON formatted list of all stored memories, including when they were created
    and their content. Results are paginated with a default of 50 items per page.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.memory_client
        if mem0_client is None:
            return "Error: Memory service is currently unavailable. Please try again later."
        memories = await mem0_client.get_all(user_id=user_id, category=memory_type)
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
        logger.exception("Error retrieving memories")
        return f"Error retrieving memories: {str(e)}"

@mcp.tool(description="Search stored memories for relevant information")
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
        memory_type: Either "memory" or "note" to select the category
        user_id: Identifier for the user whose memories should be searched
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.memory_client
        if mem0_client is None:
            return "Error: Memory service is currently unavailable. Please try again later."
        memories = await mem0_client.search(
            query,
            user_id=user_id,
            category=memory_type,
            limit=limit,
        )
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
        logger.exception("Error searching memories")
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
