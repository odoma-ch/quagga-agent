from langgraph.checkpoint.memory import InMemorySaver


class AgentRuntimeState:
    """Mutable runtime state shared between API and helper modules."""

    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        self.graph = None
        self.scope_graph = None
        self.shex_prefixes: str = ""
        self.chunk_titles: list[str] = []
        # Per-session checkpointer living in memory (thread_id -> conversation)
        self.checkpointer = InMemorySaver()


runtime_state = AgentRuntimeState()
