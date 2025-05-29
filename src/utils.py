"""Utility functions for working with the Mem0 Cloud API."""

from dataclasses import dataclass
import os
import httpx


@dataclass
class AsyncMemoryClient:
    """Minimal async client for Mem0 Cloud."""

    api_key: str
    base_url: str = "https://api.mem0.ai"

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def add(
        self,
        messages: list[dict],
        *,
        user_id: str,
        category: str = "memory",
        metadata: dict | None = None,
    ) -> dict:
        payload = {
            "user_id": user_id,
            "category": category,
            "messages": messages,
            "metadata": metadata,
        }
        resp = await self._client.post("/memory", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_all(
        self,
        *,
        user_id: str,
        category: str = "memory",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        params = {
            "user_id": user_id,
            "category": category,
            "limit": limit,
            "offset": offset,
        }
        resp = await self._client.get("/memory", params=params)
        resp.raise_for_status()
        return resp.json()

    async def search(
        self,
        query: str,
        *,
        user_id: str,
        category: str = "memory",
        limit: int = 3,
    ) -> dict:
        params = {
            "query": query,
            "user_id": user_id,
            "category": category,
            "limit": limit,
        }
        resp = await self._client.get("/memory/search", params=params)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()


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
