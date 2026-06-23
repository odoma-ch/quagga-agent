import os
import json
import re
import openai
import logging
import requests
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Type, Annotated, Optional

# from langchain.tools import tool
from langchain_core.tools import tool
from rich.console import Console
from langchain_chroma import Chroma
from langgraph.prebuilt import InjectedState

# from langchain.tools.base import BaseTool
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

import httpx
from langchain_openai import ChatOpenAI
from helpers.tool_helpers import _is_sparql_endpoint_up
from helpers.message_format_helpers import _message_content_to_text, _pretty_sparql
from agent_utils.provider_router import create_openai_client
from agent_utils.agent_constants import (
    GRAPHS_INVOLVED,
    GRAPH_KG_DESCRIPTIONS_URL,
    GRAPH_METADATA_URL,
    GRAPH_RELEVANCE_BATCH_RESPONSE_SCHEMA,
)
from agent_utils.agent_helpers import extract_question_answer_pairs
from models import (
    ContextSchema,
    ShexRetrieverInput,
    SPARQLValidationInput,
    SPARQLExtractionResult,
    GraphRelevanceBatchResponse,
    State,
)

from rdflib import Graph
from rdflib.plugins.stores import sparqlstore

from rdflib.term import Variable
from rdflib import Literal, URIRef
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.plugins.sparql import algebra as sparql_algebra

from dotenv import load_dotenv

load_dotenv()

# Disable LangChain verbose output
os.environ["LANGCHAIN_VERBOSE"] = "false"

def _as_bool(v: str | None) -> bool:
  return (v or "").strip().lower() in {"1", "true", "yes", "on", "True"}

console = Console(quiet=_as_bool(os.getenv("DISABLE_CONSOLE_PRINT")))
logging.getLogger().setLevel(logging.INFO)


def _get_tool_client_and_model() -> tuple[openai.OpenAI, str]:
    """Get the current health-routed tool client/model from runtime state."""
    from agent_utils.agent_runtime_state import runtime_state

    model = runtime_state.tool_model or os.getenv("PSNC_SMALL_TOOL_MODEL")
    if runtime_state.tool_openai_client is None:
        runtime_state.tool_openai_client = create_openai_client(
            base_url=os.getenv("PSNC_SMALL_TOOL_BASE_URL"),
            api_key=os.getenv("PSNC_SMALL_TOOL_API_KEY"),
        )
    return runtime_state.tool_openai_client, model


def _get_thread_context():
    """Return (thread_id, runtime_context) when called inside a graph run."""
    try:
        runtime = get_runtime(ContextSchema)
        context = runtime.context
        return getattr(context, "thread_id", "") or "", context
    except Exception:
        return "", None


def _resolve_session_values():
    """Resolve endpoint/chunk titles for current thread first, then fallback to defaults."""
    from agent_utils.agent_runtime_state import runtime_state

    thread_id, context = _get_thread_context()
    endpoint = ""
    chunk_titles: list[str] = []
    selected_kg = ""

    if context is not None:
        endpoint = getattr(context, "sparql_endpoint", "") or ""
        chunk_titles = list(getattr(context, "chunk_titles", []) or [])
        selected_kg = getattr(context, "selected_kg", "") or ""

    if thread_id:
        session = runtime_state.get_or_create_session(thread_id)
        endpoint = endpoint or session.sparql_endpoint
        chunk_titles = chunk_titles or list(session.chunk_titles or [])
        selected_kg = selected_kg or session.selected_kg

    return thread_id, selected_kg, endpoint, chunk_titles


def _get_scope_relevance_scores(query: str, passages: list[dict]) -> dict:
    tool_client, tool_model = _get_tool_client_and_model()
    passages_content = "\n\n".join(
        f"Passage {i+1} ({p['kg_name']}): {p['description']}"
        for i, p in enumerate(passages)
    )
    response = tool_client.chat.completions.parse(
        model=tool_model,
        max_tokens=4096,
        response_format=GraphRelevanceBatchResponse,
        messages=[
            {
                "role": "system",
                "content": "You are a relevance scoring assistant. Always respond with valid JSON only.",
            },
            {
                "role": "user",
                "content": f"""Given user messages A and multiple knowledge graph descriptions B, determine whether the messages are related to each knowledge graph.
A: {query}
B:
{passages_content}
For EACH passage, provide a decision of 'Yes' or 'No', a confidence score between 0 and 1, and a one sentence reasoning. Use the passage kg_name to identify each result.
For all the user messages, give high priority to the most recent messages [last ones in the list] and give low priority to the older messages.
Respond with ONLY valid JSON matching this schema, nothing else:
{GRAPH_RELEVANCE_BATCH_RESPONSE_SCHEMA}""",
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    return {r["kg_name"]: r for r in result["results"]}


@tool
def scope_checker_tool(state: Annotated[State, InjectedState]) -> str:
    """Classify input user messages and tool messages against known KGs, load the selected KG metadata into runtime, and return confirmation."""
    from helpers.kg_metadata_helper import setup_kg_rag_environment
    from agent_utils.agent_runtime_state import runtime_state
    
    retriever_k = int(os.getenv("RETRIEVER_K", "10"))
    
    graph_kg_descriptions = []
    for kg_name in GRAPHS_INVOLVED:
        try:
            response = requests.get(
                GRAPH_KG_DESCRIPTIONS_URL,
                params={"kg_name": kg_name},
                headers={"accept": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                graph_kg_descriptions.append(payload)
        except Exception:
            continue

    if not graph_kg_descriptions:
        return "Currently cannot load knowledge graph descriptions. Please ask the user to provide the knowledge graph name."

    query_parts: list[str] = []
    for msg in state["messages"]:
        if isinstance(msg, (tuple, list)) and len(msg) >= 2:
            if msg[0] in {"user"}:
                text = msg[1] if isinstance(msg[1], str) else str(msg[1])
                if text:
                    query_parts.append(text)
            continue

        msg_type = getattr(msg, "type", "") 
        if msg_type in {"human"}:
            text = _message_content_to_text(getattr(msg, "content", ""))
            if text:
                query_parts.append(text)

    # take only the last user message [this can be changed thats why the above code is there]
    query = "\n".join(query_parts[-1:]).strip()

    results = _get_scope_relevance_scores(query, graph_kg_descriptions)
    scores = {
        r["kg_name"]: r.get("confidence", 0.0) if r.get("decision") == "Yes" else -r.get("confidence", 0.0)
        for r in results.values()
        if isinstance(r, dict) and r.get("kg_name")
    }
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    top_key, top_score = sorted_items[0]
    second_key, second_score = sorted_items[1] if len(sorted_items) > 1 else (None, 0)
    
    if (top_score - second_score) >= (0.1 - 1e-9):
        selected_kg = top_key
        
        if selected_kg not in runtime_state.kg_metadata_cache:
            kg_endpoint = next(
                (
                    kg.get("endpoint_url")
                    for kg in graph_kg_descriptions
                    if isinstance(kg, dict) and kg.get("kg_name") == selected_kg
                ),
                None,
            )
            if kg_endpoint:
                try:
                    retriever_tool, shex_prefixes, chunk_titles, sparql_endpoint = setup_kg_rag_environment(
                        selected_kg,
                        kg_endpoint,
                        GRAPH_METADATA_URL,
                        runtime_state.chroma_client,
                        runtime_state.embedding_model,
                        k=retriever_k
                    )
                    runtime_state.kg_metadata_cache[selected_kg] = {
                        "retriever_tool": retriever_tool,
                        "shex_prefixes": shex_prefixes,
                        "chunk_titles": chunk_titles,
                        "sparql_endpoint": sparql_endpoint
                    }
                except Exception:
                    return (
                        f"I identified `{selected_kg}` as the best match, but failed to initialize its metadata. "
                        "Please try again or choose another knowledge graph."
                    )
        
        if selected_kg not in runtime_state.kg_metadata_cache:
            return (
                f"I identified `{selected_kg}` as the best match, but could not load its metadata. "
                "Please try again or choose another knowledge graph."
            )

        thread_id, _ = _get_thread_context()
        session = runtime_state.get_or_create_session(thread_id)
        session.selected_kg = selected_kg
        session.shex_prefixes = runtime_state.kg_metadata_cache[selected_kg]["shex_prefixes"]
        session.chunk_titles = runtime_state.kg_metadata_cache[selected_kg]["chunk_titles"]
        session.sparql_endpoint = runtime_state.kg_metadata_cache[selected_kg]["sparql_endpoint"]
        
        # Update runtime context with SPARQL endpoint
        try:
            current_runtime = get_runtime(ContextSchema)
            current_runtime.context.selected_kg = session.selected_kg
            current_runtime.context.sparql_endpoint = session.sparql_endpoint
            current_runtime.context.chunk_titles = session.chunk_titles
            current_runtime.context.shex_prefixes = session.shex_prefixes
        except:
            pass
        
        console.print(f"Loaded metadata for KG: {selected_kg}", style="bold green")
        return f"User messages are aligned with the knowledge graph: {selected_kg}"
    else:
        low_confidence_prompt = f"Ask the user which knowledge graph would they want me to query: {top_key} or {second_key}?"
        return low_confidence_prompt


class ShexRetrieverTool(BaseTool):
    name: str = "shex_retriever"
    description: str = "Retrieve relevant schemas from vector store based on a question"
    args_schema: Type[BaseModel] = ShexRetrieverInput

    k: int = 3
    standalone_mode: bool = False
    model_name: str = None
    retrieved_chunks: set = Field(default_factory=set)
    vectorstore: Chroma = None

    def __init__(self, vectorstore=None, k=3, standalone_mode=False, **kwargs):
        kwargs["k"] = k
        kwargs["standalone_mode"] = standalone_mode
        super().__init__(**kwargs)
        self.vectorstore = vectorstore

    def _run(self, question: str) -> str:
        """Execute the tool"""
        try:
            # Determine which vectorstore to use
            if self.vectorstore is not None:
                # Use the provided vectorstore (KG-specific mode)
                console.print(
                    f"Finding the best classes from SHEX schema with k={self.k}, standalone_mode={self.standalone_mode}",
                    style="bold blue",
                )
                total_chunks = self.vectorstore._collection.count()
                retriever = self.vectorstore.as_retriever
            else:
                # Get vectorstore from runtime state (dynamic mode)
                from agent_utils.agent_runtime_state import runtime_state
                _, selected_kg, _, _ = _resolve_session_values()
                if not selected_kg:
                    return "No knowledge graph has been selected. Please call scope_checker_tool first to determine which KG to query."
                retriever_tool = (
                    runtime_state.kg_metadata_cache.get(selected_kg, {}).get("retriever_tool")
                )
                if retriever_tool is None:
                    return "No knowledge graph has been selected. Please call scope_checker_tool first to determine which KG to query."
                
                console.print(
                    f"Finding the best classes from SHEX schema with k={self.k}, standalone_mode={self.standalone_mode}",
                    style="bold blue",
                )
                total_chunks = retriever_tool.vectorstore._collection.count()
                retriever = retriever_tool.vectorstore.as_retriever

            updated_search_kwargs = {"k": self.k}
            runtime = None
            context_retrieved_chunks = set()

            if not self.standalone_mode:
                try:
                    runtime = get_runtime(ContextSchema)
                    retrieved_attr = getattr(runtime.context, "retrieved_chunks", None)
                    context_retrieved_chunks = set(retrieved_attr) if retrieved_attr else set()
                except Exception:
                    runtime = None
                    context_retrieved_chunks = set()

            all_retrieved_chunks = self.retrieved_chunks.union(context_retrieved_chunks)

            if 0 < len(all_retrieved_chunks) < total_chunks:
                updated_search_kwargs["filter"] = {
                    "chunk_index": {"$nin": list(all_retrieved_chunks)}
                }

            retriever_result = retriever(
                search_kwargs=updated_search_kwargs, verbose=False
            ).invoke(question)
            console.print(
                f"Retriever returned {retriever_result} chunks", style="bold green"
            )

            new_chunk_indices = set()
            for doc in retriever_result:
                chunk_index = doc.metadata.get("chunk_index")
                if chunk_index is None:
                    continue
                if isinstance(chunk_index, str) and chunk_index.isdigit():
                    chunk_index = int(chunk_index)
                new_chunk_indices.add(chunk_index)

            if not self.standalone_mode:
                self.retrieved_chunks = all_retrieved_chunks.union(new_chunk_indices)
                console.print(
                    f"Retrieved chunks: {sorted(self.retrieved_chunks)}",
                    style="bold green",
                )
                if runtime is not None:
                    runtime.context.retrieved_chunks = self.retrieved_chunks
            else:
                self.retrieved_chunks.update(new_chunk_indices)

            max_model_tokens = None
            if self.standalone_mode:
                from data_functions import MAX_MODEL_TOKENS
                max_model_tokens = MAX_MODEL_TOKENS[self.model_name]

            if max_model_tokens is not None:
                retriever_result = get_fitting_chunks(retriever_result, max_model_tokens)
            result = "\n\n".join([doc.page_content for doc in retriever_result])
            logging.info(
                f"Chunks retrieved for the natural language query = {question}: {result}"
            )
            return result
        except Exception as e:
            console.print(e, style="bold red")
            return f"Error retrieving documents: {str(e)}"

    async def _arun(self, question: str) -> str:
        """Async version - optional"""
        return self._run(question)


class CustomSampleRetrieverTool(BaseTool):
    name: str = "sample_retriever"
    description: str = "Retrieve relevant samples from vector store based on a question"
    args_schema: Type[BaseModel] = ShexRetrieverInput

    vectorstore: Chroma = None
    bm25_retriever: BM25Retriever = None
    k: int = 5

    def __init__(self, vectorstore, bm25_retriever, k=5, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.k = k

    def _run(self, question: str) -> str:
        console.print(f"Finding the best samples from the KG", style="bold blue")
        updated_search_kwargs = {"k": self.k}

        embedding_retriever = self.vectorstore.as_retriever(
            search_kwargs=updated_search_kwargs, verbose=False
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, embedding_retriever], weights=[0.7, 0.3]
        )
        results = ensemble_retriever.invoke(question)
        result = "\n\n".join([doc.page_content for doc in results])
        logging.info(
            f"Samples retrieved for the natural language query = {question}: {result}"
        )
        return result

    async def _arun(self, question: str) -> str:
        """Async version - optional"""
        return self._run(question)


class CustomVoIDRetrieverTool(BaseTool):
    name: str = "void_retriever"
    description: str = (
        "Retrieve relevant VOID descriptions from vector store based on a question"
    )
    args_schema: Type[BaseModel] = ShexRetrieverInput

    vectorstore: Chroma = None
    k: int = 5
    model_name: str = None

    def __init__(self, vectorstore, k=5, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.k = k

    def _run(self, question: str) -> str:
        console.print(
            f"Finding the best VOID descriptions from the KG", style="bold blue"
        )
        updated_search_kwargs = {"k": self.k}
        retriever = self.vectorstore.as_retriever(
            search_kwargs=updated_search_kwargs, verbose=False
        )
        retriever_result = retriever.invoke(question)
        # drop the msgs with the max tokens until the length
        from data_functions import MAX_MODEL_TOKENS

        max_model_tokens = MAX_MODEL_TOKENS[self.model_name]
        retriever_result = get_fitting_chunks(retriever_result, max_model_tokens)
        result = "\n\n".join([doc.page_content for doc in retriever_result])
        logging.info(
            f"VOID descriptions for model: {self.model_name} retrieved for the natural language query = {question}: {result}"
        )
        return result

    async def _arun(self, question: str) -> str:
        """Async version - optional"""
        return self._run(question)


@tool
def describe_knowledge_graphs_tool():
    """Tool to describe the knowledge graphs which a user can ask questions against"""
    graph_kg_descriptions = []
    for kg_name in GRAPHS_INVOLVED:
        response = requests.get(
            GRAPH_KG_DESCRIPTIONS_URL, params={"kg_name": kg_name}, headers={"accept": "application/json"}).json()
        graph_kg_descriptions.append(response)

    return json.dumps(graph_kg_descriptions)


@tool
def run_sparql_subagent_tool(
    question: str,
    state: Annotated[State, InjectedState],
    config: RunnableConfig,
) -> str:
    """Run the SPARQL exploration subagent for the currently selected knowledge graph.
    Returns a JSON string with 'query' (the final SPARQL query, or null if none was produced)
    and 'summary' (the prose answer)."""
    from subagents.run_sparql_subagent import run_pipeline as _run_pipeline

    qa_pairs = extract_question_answer_pairs(state)

    _, selected_kg, endpoint, _ = _resolve_session_values()

    if not endpoint or not selected_kg:
        return "No knowledge graph selected. Call scope_checker_tool first."

    return _run_pipeline(endpoint, selected_kg, question, qa_pairs)


class RunSPARQLToolInput(BaseModel):
    query: str = Field(description="The SPARQL query to execute")

class RunSPARQLTool(BaseTool):
    """Tool to run SPARQL queries for exploration (lightweight execution)."""
    name: str = "run_sparql_tool"
    description: str = (
        "Execute SPARQL queries against the currently selected SPARQL endpoint for knowledge graph exploration. "
        "Use this to query triples, discover schema, and test patterns with small result sets."
    )
    args_schema: type[BaseModel] = RunSPARQLToolInput
    endpoint: str = ""

    def _run(self, query: str) -> str:
        """Execute SPARQL query and return limited results."""
        endpoint = self.endpoint
        if not endpoint:
            return "No SPARQL endpoint configured. Please call scope_checker_tool first to select a knowledge graph."
        try:
            # Prepend loaded prefixes if the query has no PREFIX declarations of its own.
            # if shex_prefixes and not query.lstrip().upper().startswith("PREFIX"):
            #     query = shex_prefixes.strip() + "\n" + query

            console.print(
                f"Executing SPARQL query against {endpoint}",
                style="bold blue"
            )

            sparql = SPARQLWrapper(endpoint)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(30)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: sparql.query().convert())
                try:
                    results = future.result(timeout=30)
                except TimeoutError:
                    return f"Query timed out after 30s. Try a simpler query with a smaller LIMIT."

            if "results" in results and "bindings" in results["results"]:
                bindings = results["results"]["bindings"]
                vars_list = results.get("head", {}).get("vars", [])

                if bindings:
                    display_bindings = bindings[:50]

                    table = ["| " + " | ".join(vars_list) + " |"]
                    table.append("| " + " | ".join("---" for _ in vars_list) + " |")

                    for binding in display_bindings:
                        row = []
                        for var in vars_list:
                            val = binding.get(var, {})
                            if "value" in val:
                                row.append(str(val["value"]))
                            else:
                                row.append("")
                        table.append("| " + " | ".join(row) + " |")

                    result_str = "\n".join(table)
                    return (
                        f"```sparql\n{query}\n```\n"
                        f"Results (showing {len(display_bindings)}/{len(bindings)} rows):\n{result_str}"
                    )
                else:
                    return f"```sparql\n{query}\n```\nNo results returned."
            else:
                return f"```sparql\n{query}\n```\nQuery executed successfully."

        except Exception as e:
            return f"Error executing SPARQL query: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)


@tool
def validate_sparql_tool(
    summary: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Extract the SPARQL query (if any) from the subagent's response, re-run it, and
    format the results as markdown. Returns both the prose description and the query results. 
    This tool is always called by the main agent after the `run_sparql_subagent_tool` has run."""
    try:
        # Prefer full subagent result from state — avoids LLM truncation when building tool args.
        # Keep incoming `summary` as fallback.
        extracted_summary = ""
        if isinstance(state, dict):
            for msg in reversed(state.get("messages", [])):
                if (
                    getattr(msg, "type", "") == "tool"
                    and getattr(msg, "name", None) == "run_sparql_subagent_tool"
                ):
                    extracted_summary = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                    break

        if extracted_summary:
            summary = extracted_summary

        # Never forward chain-of-thought style sections to downstream extraction.
        summary = re.sub(
            r"<think>.*?</think>",
            "",
            summary,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        if not summary:
            return "No subagent result found to validate."

        console.print("Extracting SPARQL query from subagent response", style="bold blue")

        tool_client, tool_model = _get_tool_client_and_model()
        extracted_query: str | None = None
        extracted_description: str = summary

        try:
            extraction_response = tool_client.chat.completions.parse(
                model=tool_model,
                response_format=SPARQLExtractionResult,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract two things from the user's message:\n"
                            "1. The SPARQL query exactly as written, preserving all whitespace, prefixes, and formatting.\n"
                            "2. The human-readable description exactly as written, excluding the SPARQL query block.\n"
                            "Preserve everything precisely — numeric values, counts, totals, percentages, and all other specifics "
                            "must not be modified, paraphrased, or removed. Set a field to null if not present."
                        ),
                    },
                    {"role": "user", "content": summary},
                ],
            )
            extracted: SPARQLExtractionResult | None = extraction_response.choices[0].message.parsed
            if extracted is not None:
                extracted_query = extracted.query
                extracted_description = extracted.description
        except Exception as extraction_error:
            logging.warning(
                "Structured SPARQL extraction failed, using regex fallback: %s",
                extraction_error,
            )

        if not extracted_query:
            sparql_match = re.search(r"```sparql\s*\n(.*?)\n```", summary, re.DOTALL | re.IGNORECASE)
            if sparql_match:
                extracted_query = sparql_match.group(1).strip()
            else:
                prefix_match = re.search(r"(PREFIX\s+\w+:\s+<[^>]+>\s*)+", summary, re.IGNORECASE)
                if prefix_match:
                    extracted_query = prefix_match.group(0)
                else:
                    select_match = re.search(r"(SELECT\s+.*?WHERE\s*\{.*?\})", summary, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        extracted_query = select_match.group(1)

        if isinstance(extracted_query, str):
            normalized_query = extracted_query.strip()
            fenced_match = re.fullmatch(r"```(?:sparql)?\s*(.*?)\s*```", normalized_query, re.DOTALL | re.IGNORECASE)
            if fenced_match:
                normalized_query = fenced_match.group(1).strip()
            if normalized_query.lower() in {"", "null", "none"}:
                extracted_query = None
            else:
                extracted_query = normalized_query

        _, _, endpoint, _ = _resolve_session_values()
        if not endpoint or not _is_sparql_endpoint_up(endpoint):
            if extracted_query:
                pretty_query = _pretty_sparql(extracted_query)
                return (
                    f"{extracted_description}\n\n"
                    "The SPARQL has been executed:\n"
                    "```sparql\n"
                    f"{pretty_query}\n"
                    "```\n"
                )
            return f"The SPARQL has been executed:\n{extracted_description}"

        if not extracted_query:
            return extracted_description

        pretty_query = _pretty_sparql(extracted_query)
        return (
            f"{extracted_description}\n\n"
            "The SPARQL has been executed:\n"
            "```sparql\n"
            f"{pretty_query}\n"
            "```\n"
        )
    except Exception as e:
        logging.exception("validate_sparql_tool failed")
        return f"The SPARQL has been executed:\nValidation step failed: {str(e) or 'unknown error'}"



def validate_by_ask_sparql_tool(query: str):
    """Tool to validate a query by asking the SPARQL endpoint incrementally"""
    _, _, endpoint, _ = _resolve_session_values()
    if not endpoint:
        return "No SPARQL endpoint configured. Please call scope_checker_tool first to select a knowledge graph."
    
    parsed = parseQuery(query)
    prologue = sparql_algebra.translatePrologue(parsed[0], base=None)
    algebra_query = sparql_algebra.translateQuery(parsed)
    
    prefixes = dict(prologue.namespace_manager.namespaces())
    
    prefix_str = "\n".join(
        [f"PREFIX {k or ''}: <{v}>" for k, v in prefixes.items() if k is not None]
    )
    
    triples = []
    all_filters = []

    def walk(node):
        if node is None:
            return
        if hasattr(node, "name"):
            if node.name == "BGP" and hasattr(node, "triples"):
                triples.extend(node.triples)
            elif node.name == "Filter":
                all_filters.append(node)
        for attr in ("p", "p1", "p2"):
            walk(getattr(node, attr, None))

    walk(algebra_query.algebra)

    if not triples:
        return "Could not find triple patterns in the query."

    def term(t) -> str:
        if isinstance(t, Variable):
            return f"?{t}"
        if isinstance(t, URIRef):
            t_str = str(t)
            for prefix, ns in prefixes.items():
                if prefix is None:
                    continue
                ns_str = str(ns)
                if t_str.startswith(ns_str):
                    return f"{prefix}:{t_str[len(ns_str):]}" if prefix else f":{t_str[len(ns_str):]}"
            return f"<{t}>"
        if isinstance(t, Literal):
            return t.n3()
        return str(t)

    store = SPARQLStore(query_endpoint=endpoint)
    graph = Graph(store=store)
    results = []
    accumulated_triples = []
    
    for idx, pattern in enumerate(triples):
        accumulated_triples.append(pattern)
        
        triples_str = "\n".join([
            f"  {term(s)} {term(p)} {term(o)} ."
            for s, p, o in accumulated_triples
        ])
        
        filter_str = ""
        for f in all_filters:
            filter_expr = term(f.expr) if hasattr(f, "expr") else str(f)
            filter_str += f"  FILTER({filter_expr})\n"
        
        ask_query = f"""{prefix_str}
ASK {{
{triples_str}
{filter_str}}}"""
        
        try:
            ask_result = graph.query(ask_query)
            result = bool(getattr(ask_result, "askAnswer", False))
        except Exception:
            result = False
        
        results.append({
            "step": idx + 1,
            "pattern": f"{term(pattern[0])} {term(pattern[1])} {term(pattern[2])}",
            "valid": result
        })
        
        if not result:
            break
    
    return [result for result in results if not result["valid"]]


def generate_hypothetical_description_tool(question: str, chunk_titles: list[str]):
    """Tool to align the user question with the knowledge graph ontology"""
    console.print(
        f"Making the query suitable for the required schema", style="bold blue"
    )
    http_client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url=os.getenv("PSNC_HPC_BASE_URL"),
        model="GLM-4.7",
        api_key=os.getenv("PSNC_HPC_API_KEY"),
        temperature=0.1,
        top_p=1,
        http_client=http_client,
        max_tokens=None,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    result = llm.invoke(f"""
    Reformulate and elaborate this question for better understanding and return only the reformulated question: {question}
    Integrate the following titles into the reformulated question only if they are relevant to the question: {chunk_titles}
    """)
    console.print(f"The hypothetical description: {result.content}", style="bold green")
    return result.content


def get_fitting_chunks(docs, max_tokens):
    """Drop the chunks until the total tokens are less than the max tokens"""
    doc_tokens = [(doc, len(tokenizer.encode(doc.page_content))) for doc in docs]

    total = sum(t for _, t in doc_tokens)

    # drop the largest chunk repeatedly until we fit
    while total > max_tokens and doc_tokens:
        # find the index of the largest chunk
        largest_idx = max(range(len(doc_tokens)), key=lambda i: doc_tokens[i][1])
        total -= doc_tokens[largest_idx][1]
        doc_tokens.pop(largest_idx)
    # return docs in original order
    return [doc for doc, _ in doc_tokens]
