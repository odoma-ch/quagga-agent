import os
import uuid
import asyncio

from agent_utils.agent_runtime_state import runtime_state
from agent_utils.agent_config import MAX_AGENT_STEPS, SPARQL_ENDPOINT
from helpers.message_format_helpers import (
    _message_content_to_text,
    _make_chat_completion_chunk,
)
from helpers.tool_helpers import (
    _extract_terminal_tool_answer,
    _format_tool_call,
    _format_tool_result,
)
from models import State, ContextSchema
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError


from dotenv import load_dotenv
load_dotenv()


PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, "r") as f:
        return f.read().strip()


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
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    context = _build_agent_context(thread_id)

    # Only send the last user message; the checkpointer provides prior history.
    last_message = messages[-1:]

    parts: list[str] = []
    try:
        for event in await asyncio.to_thread(
            lambda: list(runtime_state.graph.stream({"messages": last_message}, config, context=context))
        ):
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
        parts.append("I've reached the conversation depth limit for this part of our discussion. Could we take a step back or try a different approach?")

    return "\n".join(parts)


async def _stream_agent(messages: list[tuple[str, str]], thread_id: str):
    """Yield SSE chunks as tool steps + tokenized assistant output."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    context = _build_agent_context(thread_id)

    # Only send the last user message; the checkpointer provides prior history.
    last_message = messages[-1:]
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

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
        try:
            for event in runtime_state.graph.stream({"messages": last_message}, config, context=context):
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
            error_chunk = _make_chat_completion_chunk(
                "\n\nI've reached the conversation depth limit for this part of our discussion. Could we take a step back or try a different approach?",
                chunk_id=stream_id,
                chunk_type="error"
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
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    asyncio.get_running_loop().run_in_executor(None, _producer)

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        yield item

    final_chunk = _make_chat_completion_chunk(
        "", finish_reason="stop", chunk_id=stream_id, chunk_type="final"
    )
    yield final_chunk
    done_chunk = "data: [DONE]\n\n"
    yield done_chunk
