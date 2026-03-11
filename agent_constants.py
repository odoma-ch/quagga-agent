from pydantic import BaseModel, Field
from models import GraphRelevanceBatchResponse

MODEL_ID = "Graphia SPARQL Agent"
GRAPHS_INVOLVED = ["OpenCitations Meta", "GESIS KG", "EHRI - KG"]
GRAPH_KG_DESCRIPTIONS_URL = "https://quagga.graphia-ssh.eu/get_kg_info"
GRAPH_RELEVANCE_BATCH_RESPONSE_SCHEMA = GraphRelevanceBatchResponse.model_json_schema()