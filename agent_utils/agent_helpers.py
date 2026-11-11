import os
import uuid
import asyncio
import threading
import contextvars

from agent_utils.agent_runtime_state import runtime_state
from agent_utils.agent_config import MAX_AGENT_STEPS, SPARQL_ENDPOINT
from helpers.message_format_helpers import (
    _message_content_to_text,
    _make_chat_completion_chunk,
    _make_progress_chunk,
)
from helpers.tool_helpers import (
    _extract_terminal_tool_answer,
    _format_tool_call,
    _format_tool_result,
)
from models import State, ContextSchema
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError


from dotenv import load_dotenv

load_dotenv()

from agent_utils.mcp_client import set_current_thread_id, initialize_mcp_session

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

_progress_lock = threading.Lock()
_progress_streams: dict[str, tuple[asyncio.Queue, asyncio.AbstractEventLoop, str]] = {}
_progress_token_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "progress_token", default=""
)


def emit_progress(message: str):
    token = _progress_token_ctx.get()
    if not token:
        return
    with _progress_lock:
        stream_data = _progress_streams.get(token)
    if stream_data is None:
        return
    queue, loop, stream_id = stream_data
    if queue is None or loop is None or stream_id is None:
        return
    chunk = _make_progress_chunk(message, chunk_id=stream_id)
    loop.call_soon_threadsafe(queue.put_nowait, chunk)


def _load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, "r") as f:
        return f.read().strip()


def extract_question_answer_pairs(state, keep_last_n=5):
    """Extract question-answer pairs from the state."""
    messages = state["messages"]

    pairs = []
    current_question = None

    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            current_question = msg.content
        elif isinstance(msg, ToolMessage) and msg.name == "validate_sparql_tool":
            if current_question is not None:
                artifact = getattr(msg, "artifact", None)
                pairs.append(
                    {
                        "question": current_question,
                        "answer": msg.content,
                        "sparql_query": artifact.get("query") if artifact else None,
                        "results": artifact.get("rows") if artifact else None,
                    }
                )
            current_question = None

    return pairs[-keep_last_n:]


def _build_agent_context(thread_id: str):
    """Build the shared context passed to graph.stream()."""
    session = runtime_state.get_or_create_session(thread_id)
    prefixes_str = session.shex_prefixes or ""
    sparql_endpoint = session.sparql_endpoint or SPARQL_ENDPOINT

    return ContextSchema(
        thread_id=thread_id,
        selected_kg=session.selected_kg,
        sparql_endpoint=sparql_endpoint,
        shex_prefixes=prefixes_str,
        chunk_titles=session.chunk_titles,
        retrieved_chunks=set(),
    )


async def _run_agent(messages: list[tuple[str, str]], thread_id: str) -> str:
    """Invoke the LangGraph agent and collect the full response including thinking."""
    set_current_thread_id(thread_id)
    await initialize_mcp_session()

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_AGENT_STEPS,
    }
    context = _build_agent_context(thread_id)

    # Only send the last user message; the checkpointer provides prior history.
    last_message = messages[-1:]

    parts: list[str] = []
    try:

        def _collect_events():
            set_current_thread_id(thread_id)
            return list(
                runtime_state.graph.stream(
                    {"messages": last_message}, config, context=context
                )
            )

        for event in await asyncio.to_thread(_collect_events):
            for value in event.values():
                if value.get("messages"):
                    latest = value["messages"][-1]
                    # Tool-call message (LLM decided to call a tool)
                    if hasattr(latest, "tool_calls") and len(latest.tool_calls) > 0:
                        parts.append(_format_tool_call(latest))
                        continue
                    # Tool-result message
                    if hasattr(latest, "type") and latest.type == "tool":
                        terminal_answer = _extract_terminal_tool_answer(latest)
                        if terminal_answer:
                            parts.append(terminal_answer)
                            continue
                        parts.append(_format_tool_result(latest))
                        continue
                    # Final assistant response
                    text = _message_content_to_text(latest.content)
                    if text:
                        parts.append(text)
    except GraphRecursionError:
        print("[DEBUG] GraphRecursionError caught in _run_agent", flush=True)
        parts.append(
            "I've reached the conversation depth limit for this part of our discussion. Could we take a step back or try a different approach?"
        )

    return "\n".join(parts)


async def _stream_agent(messages: list[tuple[str, str]], thread_id: str):
    """Yield SSE chunks as tool steps + tokenized assistant output."""
    set_current_thread_id(thread_id)
    await initialize_mcp_session()

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_AGENT_STEPS,
    }
    context = _build_agent_context(thread_id)

    # Only send the last user message; the checkpointer provides prior history.
    last_message = messages[-1:]
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    progress_token = f"progress-{uuid.uuid4().hex[:12]}"
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

    with _progress_lock:
        _progress_streams[progress_token] = (queue, loop, stream_id)

    # Emit initial role delta for OpenAI-compatible clients.
    role_chunk = _make_chat_completion_chunk(
        "", chunk_id=stream_id, role="assistant", chunk_type="role"
    )
    yield role_chunk

    def _token_chunks(text: str):
        """Split text into stream-friendly chunks that feel token-like."""
        if not text:
            return
        words = text.split(" ")
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word

    def _producer():
        token = _progress_token_ctx.set(progress_token)
        set_current_thread_id(thread_id)
        try:
            for event in runtime_state.graph.stream(
                {"messages": last_message}, config, context=context
            ):
                for value in event.values():
                    if not value.get("messages"):
                        continue
                    latest = value["messages"][-1]
                    if hasattr(latest, "tool_calls") and len(latest.tool_calls) > 0:
                        tool_call_chunk = _make_chat_completion_chunk(
                            _format_tool_call(latest) + "\n\n",
                            chunk_id=stream_id,
                            chunk_type="tool_call",
                        )
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            tool_call_chunk,
                        )
                        continue
                    if hasattr(latest, "type") and latest.type == "tool":
                        terminal_answer = _extract_terminal_tool_answer(latest)
                        if terminal_answer:
                            for chunk in _token_chunks(terminal_answer):
                                text_chunk = _make_chat_completion_chunk(
                                    chunk, chunk_id=stream_id, chunk_type="llm_text"
                                )
                                loop.call_soon_threadsafe(
                                    queue.put_nowait,
                                    text_chunk,
                                )
                            continue
                        tool_result_chunk = _make_chat_completion_chunk(
                            _format_tool_result(latest),
                            chunk_id=stream_id,
                            chunk_type="tool_result",
                        )
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            tool_result_chunk,
                        )
                        continue

                    text = _message_content_to_text(getattr(latest, "content", ""))
                    for chunk in _token_chunks(text):
                        text_chunk = _make_chat_completion_chunk(
                            chunk, chunk_id=stream_id, chunk_type="llm_text"
                        )
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            text_chunk,
                        )
        except GraphRecursionError:
            print(
                "[DEBUG] GraphRecursionError caught in _stream_agent/_producer",
                flush=True,
            )
            error_chunk = _make_chat_completion_chunk(
                "\n\nI've reached the conversation depth limit for this part of our discussion. Could we take a step back or try a different approach?",
                chunk_id=stream_id,
                chunk_type="error",
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                error_chunk,
            )
        except Exception as exc:
            error_chunk = _make_chat_completion_chunk(
                f"\n\n**Error:** {exc}", chunk_id=stream_id, chunk_type="error"
            )
            loop.call_soon_threadsafe(
                queue.put_nowait,
                error_chunk,
            )
        finally:
            _progress_token_ctx.reset(token)
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    asyncio.get_running_loop().run_in_executor(None, _producer)

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
    finally:
        with _progress_lock:
            _progress_streams.pop(progress_token, None)

    final_chunk = _make_chat_completion_chunk(
        "", finish_reason="stop", chunk_id=stream_id, chunk_type="final"
    )
    yield final_chunk
    done_chunk = "data: [DONE]\n\n"
    yield done_chunk
