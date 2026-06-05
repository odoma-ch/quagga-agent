
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages


@dataclass
class ContextSchema:
    sparql_endpoint: str
    shex_prefixes: str
    thread_id: str = ""
    selected_kg: str = ""
    chunk_titles: list[str] = field(default_factory=list)
    retrieved_chunks: set = field(default_factory=set)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


class ShexRetrieverInput(BaseModel):
    question: str = Field(description="The question to search for in the vectorstore")


class SPARQLValidationInput(BaseModel):
    summary: str = Field(..., description="The full response text returned by the subagent")


class SPARQLExtractionResult(BaseModel):
    query: str = Field(description="The SPARQL query exactly as written, preserving all whitespace, prefixes, and formatting. Null if not present.")
    description: str = Field(description="The human-readable description exactly as written, excluding the SPARQL query block. Null if not present.")


class State(TypedDict):
    messages: Annotated[list, add_messages]


class GraphRelevanceResult(BaseModel):
    kg_name: str = Field(description="Knowledge graph name")
    decision: str = Field(description="Relevance decision: Yes or No")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score in [0, 1]")
    reasoning: str = Field(description="One sentence explanation")


class GraphRelevanceBatchResponse(BaseModel):
    results: list[GraphRelevanceResult]
