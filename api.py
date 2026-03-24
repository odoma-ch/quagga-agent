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
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from models import ContextSchema, State
from tools import (
    ShexRetrieverTool, validate_sparql_tool, describe_knowledge_graphs_tool, scope_checker_tool,
)

from agent_constants import MODEL_ID
from agent_config import SPARQL_ENDPOINT, MAX_AGENT_STEPS
from agent_runtime_state import runtime_state
from message_format_helpers import (
    _extract_messages,
    _make_chat_completion,
    _make_openai_error,
    _message_content_to_text,
)
from agent_helpers import _run_agent, _stream_agent
from tool_helpers import _after_tool

load_dotenv()


# disable this for now as we are not enabling the local model mode for the agent API
# USE_OLLAMA = os.getenv("LOCAL_MODE", "False") == "True"
MERGE_SYSTEM_MESSAGES = os.getenv("MERGE_SYSTEM_MESSAGES") == "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"

logger = logging.getLogger("quagga-sparql-agent-api")
logging.basicConfig(level=logging.INFO)


def is_openwebui_internal(message: str) -> bool:
    return message.strip().startswith("### Task:")


def sparql_chat(state: State, config: RunnableConfig):
    runtime = config["configurable"]["__pregel_runtime"]
    _shex_prefixes = runtime.context["shex_prefixes"]
    prefix_anchor = "You might need to use these prefixes in the SPARQL query:"
    clean_prefixes = (_shex_prefixes or "").replace(prefix_anchor, "").strip()

    # Seed system messages are never stored in state; always injected fresh so they
    # remain at the very start of every LLM call and never appear mid-conversation.
    added_system_messages: list[SystemMessage] = [
            SystemMessage(
                content="You are a helpful assistant called Quagga to generate SPARQL queries. If the user is just greeting you or making small talk, respond normally without generating a query."
            ),
            SystemMessage(
            content="""
If the user asks which knowledge graphs, KGs, or datasets are available to query, or wants to know details such as 
endpoint URLs, always call the `describe_knowledge_graphs_tool` tool immediately before responding. Do not answer from memory.

Rules for `scope_checker_tool`:
1. Call it AT MOST ONCE per user turn.
2. Only call it when ALL of the following are true:
   a. No knowledge graph has been confirmed yet in this conversation, OR the user explicitly asks to switch to a different KG.
   b. You have NOT already called `scope_checker_tool` in this turn.
3. A KG is only considered "confirmed" when `scope_checker_tool` has returned a single, unambiguous KG in a previous turn. If the previous turn ended with you asking the user to choose between multiple KGs, the KG is NOT yet confirmed — you MUST call `scope_checker_tool` with the user's reply to formally confirm their selection.
4. Once `scope_checker_tool` returns a result (whether it identifies a KG or says the query is out of scope), treat that result as final for this turn. Do NOT call `scope_checker_tool` again in the same turn regardless of the outcome.
5. If the KG was confirmed in a previous turn (per rule 3) and the user has not requested a change, skip `scope_checker_tool` entirely and proceed directly to schema retrieval.
6. If `scope_checker_tool` returns an out-of-scope or ambiguous result, inform the user and wait for clarification. Do NOT retry the tool."""
            ),
            SystemMessage(
                content="""
Rules for `shex_retriever`:
1. Call it only AFTER the target knowledge graph is known — either just confirmed by `scope_checker_tool` in this turn, or already established in a prior turn and not changed since.
2. If the KG is already established and the user has not switched, skip `scope_checker_tool` and call `shex_retriever` directly with the user's input.
3. Pass the user's input as-is to the tool.
4. If the tool returns no relevant shapes, inform the user that the requested concept may not exist in the selected KG. Do NOT guess a schema.
5. Use the returned ShEx shapes as the sole basis for building triple patterns, following the ShEx interpretation rules provided separately."""
            ),
            SystemMessage(
                content="""
Rules for `validate_sparql_tool`:
1. Always validate the generated SPARQL query before presenting it to the user.
2. If validation fails, you may use either or both of the following recovery strategies in any order:
   a. Revise the query based on the error message and re-validate.
   b. Call `shex_retriever` again with the user's input to check if alternative or additional shapes are available, then regenerate and re-validate.
3. If the query still fails after multiple retries, present the best attempt to the user along with the validation error and ask for guidance."""
            ),
            SystemMessage(
                content="""
A ShEx (Shape Expressions) file defines the expected structure of RDF nodes: which predicates they must have, what value types are allowed, and how many times each predicate can appear (using *, +, ?). Each constraint maps directly to a SPARQL triple pattern, with optional cardinalities (* or ?) becoming OPTIONAL {} clauses and required cardinalities (+ or no qualifier) becoming bare triple patterns. The node type is declared by the shape name itself — the opening pattern `a [ <Class> ]* ; a . + ;` is a ShEx convention and should be collapsed into a single required triple ?node rdf:type <Class> . in the WHERE clause. When a predicate allows multiple value types via OR (e.g. xsd:date OR IRI), wrap the alternatives in a UNION inside an OPTIONAL {} block. Shape references (@ShapeName OR IRI) generate a single optional triple to the linked node without recursing into the referenced shape. When multiple predicates lead to semantically similar targets, prefer the most direct path to the actual entity class — intermediate or wrapper classes that represent metadata about an entity rather than the entity itself should not be substituted for the actual target class unless explicitly requested."""
            ),
        ]
    inject_prefix_for_llm = bool(clean_prefixes)

    # Strip all system messages from state history (seeds + any stale prefix messages).
    # Seeds are re-injected fresh above; the current prefix is appended below.
    # This guarantees all SystemMessages appear before any human/AI/tool messages.
    filtered_state_messages = []
    for msg in state["messages"]:
        if isinstance(msg, (tuple, list)) and len(msg) >= 2 and msg[0] == "system":
            continue
        if getattr(msg, "type", "") == "system":
            continue
        filtered_state_messages.append(msg)

    llm_system_messages = list(added_system_messages)
    if inject_prefix_for_llm:
        llm_system_messages.append(
            SystemMessage(content=f"{prefix_anchor}\n{clean_prefixes}")
        )

    # Merge all system messages into one — vLLM (and many small models) only
    # accept a single system message at position 0. Controlled via MERGE_SYSTEM_MESSAGES.
    if MERGE_SYSTEM_MESSAGES and len(llm_system_messages) > 1:
        merged_system_content = "\n\n".join(
            m.content for m in llm_system_messages if isinstance(m.content, str)
        )
        llm_system_messages = [SystemMessage(content=merged_system_content)]

    llm_input = llm_system_messages + filtered_state_messages
    llm_message = runtime_state.llm_with_tools.invoke(llm_input)
    # Only store the LLM response in state — system messages are always rebuilt fresh.
    return {"messages": [llm_message]}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the LangGraph agent once at startup."""

    from agent_config import RETRIEVER_K
    
    # Initialize ChromaDB client and embedding model
    chroma_path = "/data/chroma_db" if os.getenv("LOCAL_MODE", "False") != "True" else "chroma_db"
    runtime_state.chroma_client = chromadb.PersistentClient(path=chroma_path)
    runtime_state.embedding_model = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        openai_api_base="https://qwen-embedding-route-graphia-app1-staging.apps.bst2.paas.psnc.pl/v1",
        openai_api_key="EMPTY"
    )

    # if USE_OLLAMA:
    #     # llm = ChatOllama(model="qwen3:30b-a3b", temperature=0.7)
    # else:
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
    )

    # Initialize dynamic retriever tool
    retriever_tool = ShexRetrieverTool(k=RETRIEVER_K)
    
    tools = [retriever_tool, validate_sparql_tool, describe_knowledge_graphs_tool, scope_checker_tool]
    runtime_state.llm = llm
    runtime_state.llm_with_tools = llm.bind_tools(tools)
    # runtime_state.scope_graph = scope_subgraph()

    graph_builder = StateGraph(State)
    # graph_builder.add_node("scope_subgraph", scope_subgraph())
    graph_builder.add_node("sparql_chat_agent", sparql_chat)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # graph_builder.add_edge(START, "scope_subgraph")
    # graph_builder.add_edge("scope_subgraph", "sparql_chat_agent")
    graph_builder.add_conditional_edges("sparql_chat_agent", tools_condition)
    graph_builder.add_conditional_edges("tools", _after_tool)

    graph_builder.set_entry_point("sparql_chat_agent")
    runtime_state.graph = graph_builder.compile(checkpointer=runtime_state.checkpointer)

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
        last_message = messages[-1][1] if messages else ""
        internal_task = isinstance(last_message, str) and is_openwebui_internal(last_message)

        # Use a stable thread_id per conversation when provided, else throw an error as it should come from OpenWebUI.
        thread_id = body.get("chat_id", None)
        if not internal_task and thread_id is None:
            raise ValueError("chat_id is required")

        # OpenWebUI sends internal metadata requests (title/tags/follow-ups)
        # as regular chat completions. Handle them directly to avoid running scope/tools.
        if internal_task and not stream and runtime_state.llm is not None:
            sys_msgs = [m for m in messages if isinstance(m, (tuple, list)) and len(m) >= 2 and m[0] == "system"]
            other_msgs = [m for m in messages if not (isinstance(m, (tuple, list)) and len(m) >= 2 and m[0] == "system")]
            ordered_messages = sys_msgs + other_msgs
            response = await asyncio.to_thread(lambda: runtime_state.llm.invoke(ordered_messages))
            content = _message_content_to_text(getattr(response, "content", ""))
            return JSONResponse(_make_chat_completion(content))

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
