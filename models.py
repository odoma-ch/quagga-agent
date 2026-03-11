
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from typing import Annotated, Literal
from langgraph.graph.message import add_messages


@dataclass
class ContextSchema:
    sparql_endpoint: str
    shex_prefixes: str


class ShexRetrieverInput(BaseModel):
    question: str = Field(description="The question to search for in the vectorstore")


class SPARQLValidationInput(BaseModel):
    query: str = Field(description="The sparql query to validate")


class State(TypedDict):
    messages: Annotated[list, add_messages]


class GraphRelevanceResult(BaseModel):
    kg_name: str = Field(description="Knowledge graph name")
    decision: str = Field(description="Relevance decision: Yes or No")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score in [0, 1]")
    reasoning: str = Field(description="One sentence explanation")


class GraphRelevanceBatchResponse(BaseModel):
    results: list[GraphRelevanceResult]
