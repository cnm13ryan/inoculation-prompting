"""Thin wrappers over the OpenAI SDK for the Batch API.

Isolated from the rest of the package so the SDK is only required when
actually hitting OpenAI. Tests and dry runs do not import this module.
"""
import time
from pathlib import Path
from typing import Callable, Optional


def make_openai_client():
    """Return a configured OpenAI client. Reads OPENAI_API_KEY from env."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai SDK not installed. `pip install openai`."
        ) from e
    return OpenAI()


def upload_batch_file(client, path: str) -> str:
    """Upload a local JSONL file as a Batch input. Returns the file id."""
    with open(path, "rb") as f:
        result = client.files.create(file=f, purpose="batch")
    return result.id


def create_batch(
    client,
    input_file_id: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: Optional[dict] = None,
) -> dict:
    """Create a batch from an already-uploaded input file."""
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )
    return _to_dict(batch)


def retrieve_batch(client, batch_id: str) -> dict:
    return _to_dict(client.batches.retrieve(batch_id))


def cancel_batch(client, batch_id: str) -> dict:
    return _to_dict(client.batches.cancel(batch_id))


def download_file(client, file_id: str, dest_path: str) -> None:
    """Download a file (output_file_id or error_file_id) to a local path."""
    response = client.files.content(file_id)
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    # The SDK's response object has a .read() method; fall back to .content/.text.
    if hasattr(response, "read"):
        content = response.read()
    elif hasattr(response, "content"):
        content = response.content
    else:
        content = response.text  # type: ignore[attr-defined]
    if isinstance(content, str):
        content = content.encode("utf-8")
    Path(dest_path).write_bytes(content)


TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


def wait_for_batch(
    client,
    batch_id: str,
    poll_interval: int = 60,
    timeout_seconds: Optional[int] = None,
    on_status: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Poll until the batch reaches a terminal state. Returns the final batch dict.

    Raises TimeoutError if `timeout_seconds` elapses before termination.
    """
    started = time.time()
    while True:
        batch = retrieve_batch(client, batch_id)
        if on_status is not None:
            on_status(batch)
        if batch.get("status") in TERMINAL_STATUSES:
            return batch
        if timeout_seconds is not None and (time.time() - started) > timeout_seconds:
            raise TimeoutError(
                f"batch {batch_id} did not reach a terminal state within "
                f"{timeout_seconds}s (last status={batch.get('status')})"
            )
        time.sleep(poll_interval)


def _to_dict(obj) -> dict:
    """SDK objects are pydantic models in newer versions, dict-like in older."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return dict(obj)
