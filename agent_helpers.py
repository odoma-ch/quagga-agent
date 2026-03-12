import os
import uuid
import json
import openai
import asyncio
import requests

from agent_runtime_state import runtime_state
from agent_config import MAX_AGENT_STEPS, SPARQL_ENDPOINT
from message_format_helpers import (
    _message_content_to_text,
    _make_chat_completion_chunk,
)
from tool_helpers import (
    _extract_terminal_tool_answer,
    _format_tool_call,
    _format_tool_result,
)
from agent_constants import (
    GRAPHS_INVOLVED,
    GRAPH_KG_DESCRIPTIONS_URL,
    GRAPH_RELEVANCE_BATCH_RESPONSE_SCHEMA,
)
from models import State, GraphRelevanceBatchResponse
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage


from dotenv import load_dotenv
load_dotenv()


openai_client = openai.OpenAI(
	api_key=os.getenv("PSNC_OKD_API_KEY"),
	base_url=os.getenv("PSNC_OKD_BASE_URL")
)


def _build_agent_context() -> dict:
    """Build the shared context dict passed to graph.stream()."""
    prefixes_str = (
        "You might need to use these prefixes in the SPARQL query: " + runtime_state.shex_prefixes
        if runtime_state.shex_prefixes
        else ""
    )
    return {
        "sparql_endpoint": SPARQL_ENDPOINT,
        "shex_prefixes": prefixes_str,
        "chunk_titles": runtime_state.chunk_titles,
        "retrieved_chunks": set(),
        "benchmark_mode": False,
        "benchmark_method_description": "ShEX",
    }


async def _run_agent(messages: list[tuple[str, str]], thread_id: str) -> str:
    """Invoke the LangGraph agent and collect the full response including thinking."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    context = _build_agent_context()

    parts: list[str] = []
    # Run synchronous graph.stream in a thread so we don't block the event loop.
    for event in await asyncio.to_thread(
        lambda: list(runtime_state.graph.stream({"messages": messages}, config, context=context))
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

    return "\n".join(parts)


async def _stream_agent(messages: list[tuple[str, str]], thread_id: str):
    """Yield SSE chunks as tool steps + tokenized assistant output."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    context = _build_agent_context()
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
            for event in runtime_state.graph.stream({"messages": messages}, config, context=context):
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


def _scope_checker(state: State, config: RunnableConfig):
    graph_kg_descriptions = []
    for kg_name in GRAPHS_INVOLVED:
        response = requests.get(
            GRAPH_KG_DESCRIPTIONS_URL, params={"kg_name": kg_name}, headers={"accept": "application/json"}).json()
        graph_kg_descriptions.append(response)

    def get_relevance_scores(query: str, passages: list[dict]) -> dict:
        passages_content = "\n\n".join(
            f"Passage {i+1} ({p['kg_name']}): {p['description']}" 
            for i, p in enumerate(passages)
        )

        response = openai_client.chat.completions.parse(
            model=os.getenv("PSNC_OKD_MEDIUM_MODEL"),
            max_tokens=2048,
            response_format=GraphRelevanceBatchResponse,
            messages=[
                {
                    "role": "system",
                    "content": "You are a relevance scoring assistant. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": f"""Given a user query A and multiple knowledge graph descriptions B, determine whether the query is related to each knowledge graph.
A: {query}
B:
{passages_content}
For EACH passage, provide a decision of 'Yes' or 'No', a confidence score between 0 and 1, and a one sentence reasoning. Use the passage kg_name to identify each result.
Respond with ONLY valid JSON matching this schema, nothing else:
{GRAPH_RELEVANCE_BATCH_RESPONSE_SCHEMA}"""
                }
            ]
        )
        result = json.loads(response.choices[0].message.content)
        kg_with_confidence_scores = {r["kg_name"]: r for r in result["results"]}
        return kg_with_confidence_scores

    # build query from user/tool messages for both tuple and LangChain message object formats.
    query_parts: list[str] = []
    for msg in state["messages"]:
        if isinstance(msg, (tuple, list)) and len(msg) >= 2:
            if msg[0] in {"user", "tool"}:
                text = msg[1] if isinstance(msg[1], str) else str(msg[1])
                if text:
                    query_parts.append(text)
            continue

        msg_type = getattr(msg, "type", "")
        if msg_type in {"human", "tool"}:
            text = _message_content_to_text(getattr(msg, "content", ""))
            if text:
                query_parts.append(text)

    query = "\n".join(query_parts)
    results = get_relevance_scores(query, graph_kg_descriptions)
    scores = {
        r["kg_name"]: r["confidence"] if r["decision"] == "Yes" else -r["confidence"] for r in results.values()
    }
    # if the highest score is relatively high, then the user input is relevant to the knowledge graph
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_key, top_score = sorted_items[0]
    second_key, second_score = sorted_items[1]
    high_confidence_prompt = f"User messages are aligned with the knowledge graph: {top_key}"
    low_confidence_prompt = f"Which graph would you want me to query: {top_key} or {second_key}?"
    # return a proper tool message with tool_call_id [making the subgrapgh act like a tool]
    # sometimes substracting two scores can be very small, so we add a small epsilon to the threshold
    scope_prompt = (
        high_confidence_prompt if (top_score - second_score) >= (0.1 - 1e-9) else low_confidence_prompt
    )
    print(f"Scope prompt: {scope_prompt} with top score: {top_score} and second score: {second_score}")
    return {
        "messages": [
            ToolMessage(
                content=scope_prompt,
                tool_call_id=f"scope_detection_{uuid.uuid4().hex[:12]}",
                name="Knowledge graph identification",
            )
        ]
    }
