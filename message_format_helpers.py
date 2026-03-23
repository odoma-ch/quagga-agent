import re
import time
import uuid
import json

from agent_constants import MODEL_ID


def _make_chat_completion(content: str, finish_reason: str = "stop") -> dict:
    """Build an OpenAI-style ChatCompletion response body."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _make_openai_error(message: str, err_type: str = "api_error") -> dict:
    """Build an OpenAI-style error payload."""
    return {"error": {"message": message, "type": err_type}}


def _make_chat_completion_chunk(
    content: str,
    finish_reason: str | None = None,
    chunk_id: str | None = None,
    role: str | None = None,
    chunk_type: str | None = None,
) -> str:
    """Build one SSE chunk in OpenAI streaming format."""
    delta = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content

    chunk = {
        "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }

    chunk["x_chunk_type"] = chunk_type
    return f"data: {json.dumps(chunk)}\n\n"


def _message_content_to_text(content) -> str:
    """Normalize LangChain message/chunk content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_messages(body: dict) -> list[tuple[str, str]]:
    """Convert OpenAI-style messages list into LangGraph (role, content) tuples."""
    messages = []
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        messages.append((role, content))
    return messages


def _pretty_sparql(query: str) -> str:
    q = re.sub(r"\s+", " ", query.strip())  # normalize spaces first

    # Protect IRIs and quoted literals so formatting does not split dots inside them.
    protected_parts: list[str] = []

    def _protect(match: re.Match) -> str:
        protected_parts.append(match.group(0))
        return f"__SPARQL_PROTECTED_{len(protected_parts)-1}__"

    q = re.sub(r"<[^>]*>|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'", _protect, q)

    for kw in [
        "PREFIX", "SELECT", "CONSTRUCT", "ASK", "DESCRIBE",
        "WHERE", "OPTIONAL", "FILTER", "VALUES",
        "GROUP BY", "ORDER BY", "LIMIT", "OFFSET"
    ]:
        q = re.sub(rf"\b{kw}\b", f"\n{kw}", q, flags=re.IGNORECASE)
    q = re.sub(r"\{\s*", "{\n  ", q)
    q = re.sub(r"\s*\}", "\n}", q)
    # Only break on triple terminators, not every dot character.
    q = re.sub(r"\s+\.(?=\s|})", " .\n  ", q)

    # Restore protected IRIs and literals.
    for idx, original in enumerate(protected_parts):
        q = q.replace(f"__SPARQL_PROTECTED_{idx}__", original)

    return q.strip()
