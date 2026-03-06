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

from models import ContextSchema, State
from helper_functions import load_shex_file, chunk_shex_file, get_embeddings_for_chunks
from tools import CustomShexRetrieverTool, validate_sparql_tool

from agent_constants import MODEL_ID
from agent_config import SPARQL_ENDPOINT, MAX_AGENT_STEPS
from agent_runtime_state import runtime_state
from message_format_helpers import _extract_messages, _make_chat_completion, _make_openai_error
from agent_helpers import _run_agent, _stream_agent

load_dotenv()


USE_OLLAMA = os.getenv("LOCAL_MODE", "False") == "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"

PATH_TO_SHEX = os.getenv("PATH_TO_SHEX", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))

logger = logging.getLogger("graphia-sparql-agent-api")
logging.basicConfig(level=logging.INFO)


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
    return {"messages": [runtime_state.llm_with_tools.invoke(state["messages"])]}



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

    tools = [retriever_tool, validate_sparql_tool]
    runtime_state.llm_with_tools = llm.bind_tools(tools)

    graph_builder = StateGraph(State)
    graph_builder.add_node("sparql_chat_agent", sparql_chat)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("sparql_chat_agent", tools_condition)
    graph_builder.add_edge("tools", "sparql_chat_agent")
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
