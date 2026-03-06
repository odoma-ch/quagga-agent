import os


SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "")
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "20"))
