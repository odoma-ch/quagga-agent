from typing import Annotated
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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
