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
from helper_functions import load_shex_file, chunk_shex_file, get_embeddings_for_chunks
from tools import (
    CustomShexRetrieverTool, validate_sparql_tool, describe_knowledge_graphs_tool, scope_checker_tool,
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


USE_OLLAMA = os.getenv("LOCAL_MODE", "False") == "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"

PATH_TO_SHEX = os.getenv("PATH_TO_SHEX", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))

logger = logging.getLogger("graphia-sparql-agent-api")
logging.basicConfig(level=logging.INFO)


def is_openwebui_internal(message: str) -> bool:
    return message.strip().startswith("### Task:")


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
    system_anchor = "You are a helpful assistant for SPARQL queries."
    last_msg = state["messages"][-1] if state["messages"] else None
    # last_msg_is_user = (
    #     (isinstance(last_msg, (tuple, list)) and len(last_msg) >= 1 and last_msg[0] == "user")
    #     or (getattr(last_msg, "type", "") in {"human", "user"})
    # )

    # if last_msg_is_user and runtime_state.scope_graph is not None:
    #     print("Running scope detection")
    #     scope_result = runtime_state.scope_graph.invoke({"messages": state["messages"]}, config)
    #     scope_messages = scope_result.get("messages", [])
    #     if scope_messages:
    #         state["messages"] = state["messages"] + scope_messages

    has_seed_system_messages = any(
        (
            isinstance(msg, (tuple, list))
            and len(msg) >= 2
            and msg[0] == "system"
            and isinstance(msg[1], str)
            and system_anchor in msg[1]
        )
        or (
            getattr(msg, "type", "") == "system"
            and isinstance(getattr(msg, "content", None), str)
            and system_anchor in msg.content
        )
        for msg in state["messages"]
    )
    print(has_seed_system_messages)

    added_system_messages: list[SystemMessage] = []
    if not has_seed_system_messages and _shex_prefixes:
        added_system_messages = [
            SystemMessage(
                content="You are a helpful assistant for SPARQL queries. If the user is just greeting you or making small talk, respond normally without generating a query."
            ),
            SystemMessage(
                content="""
If the user asks which knowledge graphs, KGs, or datasets are available to query, or wants to know details such as 
endpoint URLs, always call the `describe_knowledge_graphs_tool` tool immediately before responding. Do not answer from memory. 
If the user input implies to query a knowledge graph --- and to check which KG it is associated with, 
always call the `scope_checker_tool` tool to determine which knowledge graph is relevant."""
            ),
            SystemMessage(
                content="""Use the SHEX retriever tool to retrieve the relevant schemas for the question.
            After generating the SPARQL query, always use the validate_sparql_tool to validate the query."""
            ),
            SystemMessage(content="""
A ShEx (Shape Expressions) file defines the expected structure of RDF nodes: which predicates 
they must have, what value types are allowed, and how many times each predicate can appear (using *, +, ?). 
Each constraint maps directly to a SPARQL triple pattern, with optional cardinalities becoming OPTIONAL {} clauses. 
The node type declared with a [ <Class> ] maps to ?node rdf:type <Class> in the WHERE clause."""),
            SystemMessage(content=f"You might need to use these prefixes in the SPARQL query: {_shex_prefixes}"),
        ]
    llm_input = added_system_messages + state["messages"] if added_system_messages else state["messages"]
    llm_message = runtime_state.llm_with_tools.invoke(llm_input)
    return {"messages": added_system_messages + [llm_message]}


# def scope_subgraph():
#     """
#     Initialise the scope subgraph.
#     Ideally, it should be triggered whenever user input is received.
#     """
#     scope_subgraph_builder = StateGraph(State)
#     scope_subgraph_builder.add_node("scope_detection", _scope_checker)
#     scope_subgraph_builder.add_edge(START, "scope_detection")
#     scope_subgraph_builder.add_edge("scope_detection", END)

#     scope_subgraph = scope_subgraph_builder.compile()
#     return scope_subgraph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the LangGraph agent once at startup."""

    if not PATH_TO_SHEX or not COLLECTION_NAME:
        raise RuntimeError(
            "Set PATH_TO_SHEX, COLLECTION_NAME (and optionally SPARQL_ENDPOINT, RETRIEVER_K) "
            "as environment variables before starting the API."
        )

    retriever_tool, runtime_state.shex_prefixes, runtime_state.chunk_titles = setup_rag_environment(
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

    logger.info("Graphia SPARQL agent is ready")
    yield  # application is running
    logger.info("Shutting down Graphia SPARQL Agent API")


app = FastAPI(
    title="Graphia SPARQL Agent API",
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
            response = await asyncio.to_thread(lambda: runtime_state.llm.invoke(messages))
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
