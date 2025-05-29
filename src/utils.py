"""Utility functions for working with the Mem0 Cloud API."""

from mem0 import AsyncMemoryClient
import os


def get_mem0_client() -> AsyncMemoryClient:
    """Instantiate an async Mem0 Cloud client from environment variables."""

    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("MEM0_API_KEY environment variable is required")
    base_url = os.getenv("MEM0_BASE_URL", "https://api.mem0.ai")

    return AsyncMemoryClient(api_key=api_key, base_url=base_url)


async def close_mem0_client(client: AsyncMemoryClient) -> None:
    """Close the HTTP client used by :class:`AsyncMemoryClient`."""
    if client is not None:
        await client.aclose()
