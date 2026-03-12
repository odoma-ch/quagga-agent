import os
import json
import logging
from typing import Type
from datetime import datetime
from pydantic import BaseModel

# from langchain.tools import tool
from langchain_core.tools import tool
from rich.console import Console
from langchain_chroma import Chroma

# from langchain.tools.base import BaseTool
from langchain_core.tools import BaseTool
from langgraph.runtime import get_runtime
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

import httpx
from langchain_openai import ChatOpenAI
from tool_helpers import _is_sparql_endpoint_up
from models import ContextSchema, ShexRetrieverInput, SPARQLValidationInput


from rdflib import Graph
from rdflib.plugins.stores import sparqlstore

from dotenv import load_dotenv

load_dotenv()

# Disable LangChain verbose output
os.environ["LANGCHAIN_VERBOSE"] = "false"


console = Console()
date = datetime.now().strftime("%Y%m%dT%H:%M:%S")
filename = f"conversation_logs_{date}.log"
file_handler = logging.FileHandler(filename)
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)


class CustomShexRetrieverTool(BaseTool):
    name: str = "shex_retriever"
    description: str = "Retrieve relevant schemas from vector store based on a question"
    args_schema: Type[BaseModel] = ShexRetrieverInput

    vectorstore: Chroma = None
    k: int = 3
    retrieved_chunks: set = set()
    standalone_mode: bool = False
    model_name: str = None

    def __init__(self, vectorstore, k=3, standalone_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.k = k
        self.retrieved_chunks = set()
        self.standalone_mode = standalone_mode

    def _run(self, question: str) -> str:
        """Execute the tool"""
        try:
            console.print(
                f"Finding the best classes from SHEX schema with k={self.k}, standalone_mode={self.standalone_mode}",
                style="bold blue",
            )
            total_chunks = self.vectorstore._collection.count()

            updated_search_kwargs = {"k": self.k}
            runtime = None
            context_retrieved_chunks = set()

            # Get previously retrieved chunks from context if available
            if not self.standalone_mode:
                try:
                    runtime = get_runtime(ContextSchema)
                    context_retrieved_chunks = set(
                        runtime.context.get("retrieved_chunks", set())
                    )
                except Exception:
                    # In non-graph execution contexts, runtime may be unavailable.
                    runtime = None
                    context_retrieved_chunks = set()

            # Merge with instance retrieved_chunks
            all_retrieved_chunks = self.retrieved_chunks.union(context_retrieved_chunks)

            # parameter: take total_chunks/2 instead (another option)
            if 0 < len(all_retrieved_chunks) < total_chunks:
                updated_search_kwargs["filter"] = {
                    "chunk_index": {"$nin": list(all_retrieved_chunks)}
                }

            retriever = self.vectorstore.as_retriever(
                search_kwargs=updated_search_kwargs, verbose=False
            )
            retriever_result = retriever.invoke(question)
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

                # Update context with newly retrieved chunks (when runtime exists)
                if runtime is not None:
                    runtime.context["retrieved_chunks"] = self.retrieved_chunks
            else:
                self.retrieved_chunks.update(new_chunk_indices)

            max_model_tokens = None
            if self.standalone_mode:
                from data_functions import MAX_MODEL_TOKENS

                max_model_tokens = MAX_MODEL_TOKENS[self.model_name]

            if max_model_tokens is not None:
                retriever_result = get_fitting_chunks(
                    retriever_result, max_model_tokens
                )
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


@tool(args_schema=SPARQLValidationInput)
def validate_sparql_tool(query: str):
    """Tool to execute a sparql query and validate the result against a remote endpoint"""
    try:
        runtime = get_runtime(ContextSchema)
        endpoint = runtime.context.get("sparql_endpoint")
        console.print(
            f"Running the query against the SPARQL endpoint", style="bold blue"
        )
    
        results = []
        console.print(
            f"Querying the SPARQL endpoint: {endpoint}", style="bold blue"
        )

        if not _is_sparql_endpoint_up(endpoint):
            return f"The SPARQL endpoint is not reachable. Currently the knowledge graph is not available."

        store = sparqlstore.SPARQLStore(query_endpoint=endpoint)
        g = Graph(store=store)

        logging.info(f"Query to be validated: {query}")
        results = g.query(query)
        console.print(f"Results: {results}", style="bold green")
        console.print(f"""```sparql\n{query}\n```""")
    except Exception as e:
        console.print(
            f"Error running the query with the following exception: {e}",
            style="bold red",
        )
        return f"Rewrite the query given the context and tools and try again, this is the error: {e}"

    results_list = list(results)
    logging.info(
        f"Results retrieved for the query: {len(results_list)} with results: {results_list}"
    )
    if len(results_list) == 0:
        chunk_titles = runtime.context.get("chunk_titles")
        new_text_query = generate_hypothetical_description_tool(query, chunk_titles)
        console.print(
            f"New formulated query to the KG schema: {new_text_query}",
            style="bold green",
        )
        return f"""
        No results were found for the previous query
        A new textual query has been generated: {new_text_query}. 
        Use always the shex_retriever_tool again to look at other classes and properties and generate a new SPARQL query.
        """
    # Return markdown without leading indentation so the SPARQL block renders cleanly.
    return (
        "The SPARQL has been executed:\n"
        "```sparql\n"
        f"{query.strip()}\n"
        "```\n"
        f"Query returned some results, lets return this query as the end result if there exists a result: {results_list[0][:5]}"
    )


def generate_hypothetical_description_tool(question: str, chunk_titles: list[str]):
    """Tool to generate a hypothetical description to a query user has asked for"""
    console.print(
        f"Making the query suitable for the required schema", style="bold blue"
    )
    http_client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url=os.getenv("PSNC_HPC_BASE_URL"),
        model="DeepSeek-V3.1-vLLM",
        api_key=os.getenv("PSNC_HPC_API_KEY"),
        temperature=0.1,
        top_p=1,
        http_client=http_client,
        max_tokens=None,
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
