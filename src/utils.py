"""Utility functions for working with the Mem0 Cloud API."""

from mem0 import AsyncMemoryClient
import os
import logging


logger = logging.getLogger(__name__)


def get_mem0_client() -> AsyncMemoryClient:
    """Instantiate an async Mem0 Cloud client from environment variables.

    This function supports both older and newer versions of ``mem0ai``. Some
    versions of :class:`~mem0.AsyncMemoryClient` do not accept a ``base_url``
    parameter, so we attempt to provide it only if supported. If passing the
    parameter fails, we fall back to instantiating the client with only the API
    key.
    """

    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        raise ValueError("MEM0_API_KEY environment variable is required")
    base_url = os.getenv("MEM0_BASE_URL")

    if base_url:
        try:
            return AsyncMemoryClient(api_key=api_key, base_url=base_url)
        except TypeError as exc:
            logger.warning(
                "AsyncMemoryClient does not accept 'base_url': %s; "
                "falling back to default", exc
            )

    return AsyncMemoryClient(api_key=api_key)


async def close_mem0_client(client: AsyncMemoryClient) -> None:
    """Close the HTTP client used by :class:`AsyncMemoryClient`."""
    if client is not None:
        await client.aclose()
