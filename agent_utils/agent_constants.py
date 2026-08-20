from typing import TypedDict
from pydantic import BaseModel, Field


class GraphRelevanceResult(BaseModel):
    kg_name: str = Field(description="Knowledge graph name")
    decision: str = Field(description="Relevance decision: Yes or No")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score in [0, 1]")
    reasoning: str = Field(description="One sentence explanation")


class GraphRelevanceBatchResponse(BaseModel):
    results: list[GraphRelevanceResult]


MODEL_ID = "Graphia SPARQL Agent"
GRAPHS_INVOLVED = ["GESIS KG", "EHRI - KG", "DNB KB", "Wikidata", "GoTriple KG"]
GRAPH_METADATA_URL = "https://exygen.graphia-ssh.eu/get_kg_metadata"
GRAPH_KG_DESCRIPTIONS_URL = "https://quagga.graphia-ssh.eu/get_kg_info"
GRAPH_RELEVANCE_BATCH_RESPONSE_SCHEMA = GraphRelevanceBatchResponse.model_json_schema()
