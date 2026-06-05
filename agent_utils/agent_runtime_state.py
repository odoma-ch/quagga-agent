# from langgraph.checkpoint.memory import InMemorySaver
import os
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from dotenv import load_dotenv
load_dotenv()


class SessionRuntimeState:
    """Per-thread mutable runtime values."""

    def __init__(self):
        self.selected_kg: str = ""
        self.shex_prefixes: str = ""
        self.chunk_titles: list[str] = []
        self.sparql_endpoint: str = ""
        self.mcp_session_id: str | None = None
        self.mcp_initialized: bool = False


class AgentRuntimeState:
    """Mutable runtime state shared between API and helper modules."""

    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        self.tool_openai_client = None
        self.tool_model = None
        self.graph = None
        self.scope_graph = None
        # Per-session checkpointer living in memory (thread_id -> conversation)
        # self.checkpointer = InMemorySaver()
        postgres_db = os.getenv("POSTGRES_DB")
        postgres_user = os.getenv("POSTGRES_USER")
        postgres_host = os.getenv("POSTGRES_HOST")
        postgres_password = os.getenv("POSTGRES_PASSWORD")
        postgres_conn_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}/{postgres_db}"
        self.pool = AsyncConnectionPool(
            conninfo=postgres_conn_string,
            open=False,
            min_size=2,
            max_size=8,
            max_lifetime=300,
            max_idle=60,
            timeout=30.0,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self.checkpointer = AsyncPostgresSaver(self.pool)
        # cached KG metadata: {kg_name: {sparql_endpoint, shex_content, retriever_tool}}
        self.kg_metadata_cache: dict = {}
        # per-thread runtime values (thread_id -> SessionRuntimeState)
        self.session_state_by_thread: dict[str, SessionRuntimeState] = {}
        self.chroma_client = None
        self.embedding_model = None
        # current thread ID for MCP session lookup (set at start of request)
        self.current_thread_id: str = ""

    def get_or_create_session(self, thread_id: str | None) -> SessionRuntimeState:
        """Return the mutable runtime state for the given thread."""
        key = thread_id or "__default__"
        if key not in self.session_state_by_thread:
            self.session_state_by_thread[key] = SessionRuntimeState()
        return self.session_state_by_thread[key]


runtime_state = AgentRuntimeState()
