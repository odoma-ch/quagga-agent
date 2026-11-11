import os

SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "")
MAX_AGENT_STEPS = int(
    os.getenv("MAX_AGENT_STEPS", "60")
)  # 40 for the subagents, 20 for the main agent
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "20"))
