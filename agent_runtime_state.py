from langgraph.checkpoint.memory import InMemorySaver


class SessionRuntimeState:
    """Per-thread mutable runtime values."""

    def __init__(self):
        self.selected_kg: str = ""
        self.shex_prefixes: str = ""
        self.chunk_titles: list[str] = []
        self.sparql_endpoint: str = ""


class AgentRuntimeState:
    """Mutable runtime state shared between API and helper modules."""

    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        self.graph = None
        self.scope_graph = None
        # Per-session checkpointer living in memory (thread_id -> conversation)
        self.checkpointer = InMemorySaver()
        # cached KG metadata: {kg_name: {sparql_endpoint, shex_content, retriever_tool}}
        self.kg_metadata_cache: dict = {}
        # per-thread runtime values (thread_id -> SessionRuntimeState)
        self.session_state_by_thread: dict[str, SessionRuntimeState] = {}
        self.chroma_client = None
        self.embedding_model = None

    def get_or_create_session(self, thread_id: str | None) -> SessionRuntimeState:
        """Return the mutable runtime state for the given thread."""
        key = thread_id or "__default__"
        if key not in self.session_state_by_thread:
            self.session_state_by_thread[key] = SessionRuntimeState()
        return self.session_state_by_thread[key]


runtime_state = AgentRuntimeState()
