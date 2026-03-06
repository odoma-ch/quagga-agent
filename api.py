"""
OpenAI-compatible API for the Quagga LangGraph Agent (ShEx retriever only).

Exposes /v1/models and /v1/chat/completions so that OpenWebUI (or any
OpenAI-compatible client) can talk to the agent out of the box.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

    # Then in OpenWebUI → Admin → Connections → OpenAI API:
    #   URL:  http://<host>:8000/v1
    #   Key:  (any non-empty string, e.g. "sk-dummy")
"""

import os
import json
import time
import uuid
import httpx
import logging
import asyncio
from contextlib import asynccontextmanager

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from models import ContextSchema, State
from helper_functions import load_shex_file, chunk_shex_file, get_embeddings_for_chunks
from tools import CustomShexRetrieverTool, validate_sparql_tool

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration – read from environment variables
# ---------------------------------------------------------------------------
USE_OLLAMA = os.getenv("LOCAL_MODE", "False") == "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"

PATH_TO_SHEX = os.getenv("PATH_TO_SHEX", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))

MODEL_ID = "Graphia SPARQL Agent"
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "20"))
SSE_DEBUG = os.getenv("SSE_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger("graphia-sparql-agent-api")
logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Module-level state (populated during lifespan startup)
# ---------------------------------------------------------------------------
llm_with_tools = None
graph = None
shex_prefixes: str = ""
chunk_titles: list[str] = []
# Per-session checkpointer living in memory (thread_id → conversation)
checkpointer = InMemorySaver()


# ---------------------------------------------------------------------------
# RAG environment setup (mirrors run_agent.py, ShEx only)
# ---------------------------------------------------------------------------
def setup_rag_environment(path_to_shex: str, collection_name: str, k: int = 3):
    """Return (retriever_tool, shex_prefixes, chunk_titles)."""
    shex_collection_name = collection_name + "_shex"

    if not os.path.exists("chroma_db"):
        shex_content = load_shex_file(path_to_shex)
        shex_prefixes_local, shex_chunks, chunk_titles_local = chunk_shex_file(shex_content)
        shex_vector_store = get_embeddings_for_chunks(shex_collection_name, shex_chunks, use_chonkie=False)
        retriever_tool = CustomShexRetrieverTool(vectorstore=shex_vector_store, k=k)
        return retriever_tool, shex_prefixes_local, chunk_titles_local
    else:
        client = chromadb.PersistentClient(path="chroma_db")
        embedding_model = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-0.6B",
            openai_api_base="https://qwen-embedding-route-graphia-app1-staging.apps.bst2.paas.psnc.pl/v1",
            openai_api_key="EMPTY",
        )
        available_collections = [c.name for c in client.list_collections()]
        if shex_collection_name not in available_collections:
            raise RuntimeError(f"Collection {shex_collection_name} not found in chroma_db")

        retriever_tool = CustomShexRetrieverTool(
            vectorstore=Chroma(
                collection_name=shex_collection_name,
                persist_directory="chroma_db",
                embedding_function=embedding_model,
            ),
            k=k,
        )

        shex_content = load_shex_file(path_to_shex)
        shex_prefixes_local, _, chunk_titles_local = chunk_shex_file(shex_content)
        return retriever_tool, shex_prefixes_local, chunk_titles_local


def sparql_chat(state: State, config: RunnableConfig):
    runtime = config["configurable"]["__pregel_runtime"]
    _shex_prefixes = runtime.context["shex_prefixes"]
    endpoint = runtime.context["sparql_endpoint"]

    shex_already_added = any(
        hasattr(msg, "type") and msg.type == "system" and msg.content == _shex_prefixes
        for msg in state["messages"]
    )

    if not shex_already_added and _shex_prefixes:
        state["messages"].insert(0, ("system", f"You are a helpful assistant for SPARQL queries. If the user is just greeting you or making small talk, respond normally without generating a query."))
        state["messages"].insert(0, ("system", _shex_prefixes))
        state["messages"].insert(
            0,
            (
                "system",
                f"""Use the SHEX retriever tool to retrieve the relevant schemas for the question.
            After generating the SPARQL query, always use the validate_sparql_tool to validate the query.""",
            ),
        )
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the LangGraph agent once at startup."""
    global llm_with_tools, graph, shex_prefixes, chunk_titles

    if not PATH_TO_SHEX or not COLLECTION_NAME:
        raise RuntimeError(
            "Set PATH_TO_SHEX, COLLECTION_NAME (and optionally SPARQL_ENDPOINT, RETRIEVER_K) "
            "as environment variables before starting the API."
        )

    retriever_tool, shex_prefixes, chunk_titles = setup_rag_environment(
        PATH_TO_SHEX, COLLECTION_NAME, k=RETRIEVER_K
    )

    if USE_OLLAMA:
        llm = ChatOllama(model="qwen3:30b-a3b", temperature=0.7)
    else:
        base_url = os.getenv("PSNC_OKD_BASE_URL")
        model = os.getenv("PSNC_OKD_MEDIUM_MODEL")
        api_key = os.getenv("PSNC_OKD_API_KEY")
        http_client = httpx.Client(verify=False)
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0.1,
            top_p=0.9,
            http_client=http_client,
            max_tokens=None,
        )

    tools = [retriever_tool, validate_sparql_tool]
    llm_with_tools = llm.bind_tools(tools)

    graph_builder = StateGraph(State)
    graph_builder.add_node("sparql_chat_agent", sparql_chat)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("sparql_chat_agent", tools_condition)
    graph_builder.add_edge("tools", "sparql_chat_agent")
    graph_builder.set_entry_point("sparql_chat_agent")

    graph = graph_builder.compile(checkpointer=checkpointer)

    logger.info("Quagga SPARQL agent is ready")
    yield  # application is running
    logger.info("Shutting down Quagga SPARQL Agent API")


app = FastAPI(
    title="Quagga SPARQL Agent API",
    description="OpenAI-compatible API wrapping the LangGraph ShEx-based SPARQL agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------------------------------------------------------------------------
# Friendly labels for tool names shown to the user
# ---------------------------------------------------------------------------
_TOOL_DISPLAY_NAMES = {
    "shex_retriever": "Searching ShEx schemas",
    "validate_sparql_tool": "Generating and validating SPARQL query",
}


def _format_tool_call(message) -> str:
    """Render an LLM tool-call message as a readable markdown block."""
    parts = []
    for tc in message.tool_calls:
        name = tc.get("name", "unknown_tool")
        args = tc.get("args", {})
        display = _TOOL_DISPLAY_NAMES.get(name, name)
        header = f"**{display}**"

        # TODO: check if we need to show args
        # if args:
        #     # Show the arguments compactly
        #     args_str = ", ".join(f"`{k}`: {v}" for k, v in args.items())
        #     parts.append(f"{header}\n> {args_str}\n")
        # else:
        parts.append(f"{header}\n")
    return "\n".join(parts)


def _format_tool_result(message) -> str:
    """Render a tool result message as a collapsible markdown details block."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    tool_name = getattr(message, "name", "tool")
    display = _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    # Truncate very long results so the chat stays readable
    preview = content[:500] + ("…" if len(content) > 500 else "")
    return (
        f"<details>\n<summary>Result from {display}</summary>\n\n"
        f"```\n{preview}\n```\n</details>\n\n"
    )


def _build_agent_context() -> dict:
    """Build the shared context dict passed to graph.stream()."""
    prefixes_str = (
        "You might need to use these prefixes in the SPARQL query: " + shex_prefixes
        if shex_prefixes
        else ""
    )
    return {
        "sparql_endpoint": SPARQL_ENDPOINT,
        "shex_prefixes": prefixes_str,
        "chunk_titles": chunk_titles,
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
        lambda: list(graph.stream({"messages": messages}, config, context=context))
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
    print("role", role_chunk)
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
            for event in graph.stream({"messages": messages}, config, context=context):
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
                        print("tool_call", tool_call_chunk)
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            tool_call_chunk,
                        )
                        continue
                    if hasattr(latest, "type") and latest.type == "tool":
                        tool_result_chunk = _make_chat_completion_chunk(
                            _format_tool_result(latest),
                            chunk_id=stream_id,
                            chunk_type="tool_result",
                        )
                        print("tool_result", tool_result_chunk)
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
                        print("text", text_chunk)
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            text_chunk,
                        )
        except Exception as exc:
            error_chunk = _make_chat_completion_chunk(
                f"\n\n**Error:** {exc}", chunk_id=stream_id, chunk_type="error"
            )
            print("error", error_chunk)
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
    print("final", final_chunk)
    yield final_chunk
    done_chunk = "data: [DONE]\n\n"
    print("done", done_chunk)
    yield done_chunk


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    """Return the list of available models (single agent model)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Odoma",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Handle a chat completion request.

    Supports both streaming (``stream: true``) and non-streaming modes,
    matching the OpenAI Chat Completions API contract expected by OpenWebUI.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=_make_openai_error("Invalid JSON body.", "invalid_request_error"),
        )

    try:
        messages = _extract_messages(body)
        stream = body.get("stream", False)

        # Use a stable thread_id per conversation when provided, else create one.
        thread_id = body.get("thread_id", str(uuid.uuid4()))

        if stream:
            return StreamingResponse(
                _stream_agent(messages, thread_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        content = await _run_agent(messages, thread_id)
        return JSONResponse(_make_chat_completion(content))
    except Exception as exc:
        logger.exception("chat_completions failed")
        return JSONResponse(
            status_code=500,
            content=_make_openai_error(str(exc), "internal_server_error"),
        )


@app.get("/health")
async def health():
    return {"status": "Agent is functioning"}
