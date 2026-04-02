import os
import json
import uuid
import httpx
import shutil
import logging
import argparse

import chromadb
from langchain_chroma import Chroma

from rich.prompt import Prompt
from rich.console import Console
from langchain_openai import OpenAIEmbeddings

from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from langchain_core.runnables import RunnableConfig

from langgraph.runtime import Runtime
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from models import ContextSchema, State
from helpers.helper_functions import get_embeddings_for_chunks, load_shex_file, chunk_shex_file, chunk_sample_file
from tools import (
    CustomShexRetrieverTool, validate_sparql_tool, 
    generate_hypothetical_description_tool, CustomSampleRetrieverTool
)

USE_OLLAMA = True if os.getenv("LOCAL_MODE") == "True" else False
# set the tokenizers parallelism to be false to avoid the warning at the start
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable LangChain verbose output
os.environ["LANGCHAIN_VERBOSE"] = "false"
console = Console()


def setup_rag_environment(path_to_shex, collection_name, k=3):
    shex_collection_name = collection_name + "_shex"
    sample_collection_name = collection_name + "_sample"
    if not os.path.exists("chroma_db"):
        # shutil.rmtree("chroma_db", ignore_errors=True)

        shex_content = load_shex_file(path_to_shex)
        shex_prefixes, shex_chunks, chunk_titles = chunk_shex_file(shex_content)
        shex_vector_store = get_embeddings_for_chunks(shex_collection_name, shex_chunks, use_chonkie=False)

        # sample_content = load_shex_file(path_to_sample)
        # sample_prefixes, sample_chunks = chunk_sample_file(sample_content)
        # sample_vector_store = get_embeddings_for_chunks(sample_collection_name, sample_chunks, use_chonkie=True)

        retriever_tool = CustomShexRetrieverTool(vectorstore=shex_vector_store, k=k)
        # sample_retriever_tool = CustomSampleRetrieverTool(vectorstore=sample_vector_store, k=k)
        return retriever_tool, shex_prefixes, chunk_titles # , sample_retriever_tool
    else:
        client = chromadb.PersistentClient(path="chroma_db")
        embedding_model = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-0.6B",
            openai_api_base="https://qwen-embedding-route-graphia-app1-staging.apps.bst2.paas.psnc.pl/v1",
            openai_api_key="EMPTY"
        )

        available_collections = [_.name for _ in client.list_collections()]

        if not shex_collection_name in available_collections:
            raise Exception(f"Collection {shex_collection_name} not found")
            
        # if not sample_collection_name in available_collections:
        #     raise Exception(f"Collection {sample_collection_name} not found")
        
        retriever_tool = CustomShexRetrieverTool(
            vectorstore=Chroma(
                collection_name=shex_collection_name, persist_directory="chroma_db",
                embedding_function=embedding_model,
            ), 
            k=k
        )
        # sample_retriever_tool = CustomSampleRetrieverTool(
        #     vectorstore=Chroma(
        #         collection_name=sample_collection_name, persist_directory="chroma_db", embedding_function=embedding_model
        #     ), 
        #     k=k
        # )

        shex_content = load_shex_file(path_to_shex)
        shex_prefixes, _, chunk_titles = chunk_shex_file(shex_content)

        return retriever_tool, shex_prefixes, chunk_titles # , sample_retriever_tool


def sparql_chat(state: State, config: RunnableConfig):
    runtime = config["configurable"]["__pregel_runtime"]
    shex_prefixes = runtime.context["shex_prefixes"]
    endpoint = runtime.context["sparql_endpoint"]
    benchmark_method_description = runtime.context["benchmark_method_description"]
    shex_already_added = any(
        hasattr(msg, "type") and msg.type == "system" and msg.content == shex_prefixes 
        for msg in state["messages"]
    )

    if not shex_already_added and shex_prefixes:
        state["messages"].insert(0, ("system", shex_prefixes))
        state["messages"].insert(0, (
            "system", f"""
            You are a helpful assistant that can help with SPARQL queries.

            Generate the SPARQL query for a given user input for the endpoint: {endpoint}

            Use the {benchmark_method_description} retriever tool to retrieve the relevant schemas for the question.
            After generating the SPARQL query, always use the validate_sparql_tool to validate the query.
            """
        ))
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    # Example endpoint URL: "https://sparql.opencitations.net/meta"
    parser.add_argument("--endpoint-url", type=str, required=True)
    # Example path: "../test/artifacts/shex/shex_2025-08-11T17-01-11_oc_meta.ttl"
    parser.add_argument("--path-to-shex", type=str, required=True)
    # Example path: "../generate_sample/samples/opencitations_meta_2_20_TOPN=20_PP=0.5.ttl"
    # parser.add_argument("--path-to-sample", type=str, required=True)
    # Example collection name: "opencitations_meta"
    parser.add_argument("--collection-name", type=str, required=True)
    parser.add_argument("--benchmark-mode", action="store_true", default=False)
    parser.add_argument("--benchmark-model", type=str, default="M", choices=["S", "M", "L"])
    parser.add_argument("--benchmark-method", type=str, default="3", choices=["2", "3", "4", "5", "6", "7"])
    args = parser.parse_args()

    retriever_tool, shex_prefixes, chunk_titles = setup_rag_environment(
        args.path_to_shex, args.collection_name, k=args.k
    )

    if USE_OLLAMA:
        llm = ChatOllama(
            model="qwen3:30b-a3b",
            temperature=0.7,
        )
        tools = [retriever_tool, validate_sparql_tool]
        llm_with_tools = llm.bind_tools(tools)
    else:
        base_url, model, api_key = os.getenv("PSNC_OKD_BASE_URL"), os.getenv("PSNC_OKD_MODEL"), os.getenv("PSNC_OKD_API_KEY")
        if args.benchmark_mode:
            if args.benchmark_model == "S":
                base_url, api_key, model = (
                    os.getenv("SMALL_MODEL_BASE_URL"), os.getenv("SMALL_MODEL_API_KEY"), "Qwen/Qwen2.5-Coder-7B-Instruct"
                )
            elif args.benchmark_model == "M":
                base_url, api_key, model = (
                    os.getenv("PSNC_OKD_BASE_URL"), os.getenv("PSNC_OKD_API_KEY"), "Qwen3-Coder-30B-A3B-Instruct-Q8_0"
                )
            elif args.benchmark_model == "L":
                base_url, api_key, model = (
                    os.getenv("PSNC_HPC_BASE_URL"), os.getenv("PSNC_HPC_API_KEY"), "DeepSeek-V3.1-vLLM"
                )

        http_client = httpx.Client(verify=False)
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=0.1,
            top_p=0.9,
            http_client=http_client,
            max_tokens=None
        )
        tools = [retriever_tool, validate_sparql_tool]
        llm_with_tools = llm.bind_tools(tools)

    graph_builder = StateGraph(State)
    graph_builder.add_node("sparql_chat_agent", sparql_chat)

    tool_node = ToolNode(tools=[retriever_tool, validate_sparql_tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("sparql_chat_agent", tools_condition)
    graph_builder.add_edge("tools", "sparql_chat_agent")
    graph_builder.set_entry_point("sparql_chat_agent")

    checkpointer = InMemorySaver()
    # todo: use postgres db as checkpointer as later
    graph = graph_builder.compile(checkpointer=checkpointer)
    # print(graph.get_graph().draw_ascii())

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # normal mode for the agent to interact with the user
    if not args.benchmark_mode:
        while True:
            user_input = Prompt.ask("[bold red]\nUser[/bold red]")
            if user_input.lower() in ["/bye", "/quit", "/exit", "bye", "quit", "exit"]:
                console.print("Goodbye!", style="bold blue")
                break

            prefixes_str = "You might need to use these prefixes in the SPARQL query: " + shex_prefixes

            for event in graph.stream(
                {"messages": [("user", user_input)]},
                config, context={"sparql_endpoint": args.endpoint_url, 
                "shex_prefixes": prefixes_str, "chunk_titles": chunk_titles, "retrieved_chunks": set()}):
                for value in event.values():
                    if value["messages"]:
                        latest_message = value["messages"][-1]
                        # tool call message
                        if hasattr(latest_message, "tool_calls") and len(latest_message.tool_calls) > 0:
                            tool_name = latest_message.tool_calls[0]["name"]
                            # if tool_name == "shex_retriever":
                            continue
                        # tool result message
                        elif hasattr(latest_message, "type") and latest_message.type == "tool":
                            continue
                        console.print(f"\nAssistant:\n{latest_message.content}", style="bold green")

    # run only when the benchamrk mode is enabled
    if args.benchmark_mode:
        home_dir = os.getenv("HOME")
        with open(f"{home_dir}/quagga-agent/benchmark/dataset/orkg_sciqa_rows.json", "r") as f:
            sciqa_test_rows = json.load(f)

            for row in sciqa_test_rows:
                ground_truth_id = row["id"]
                user_input = row["question"]["string"]
                ground_truth_query = row["query"]["sparql"]

                # only used when the RAG involves VoID and ShEX files
                prefixes_str = "You might need to use these prefixes in the SPARQL query: " + shex_prefixes

                context = {
                    "sparql_endpoint": args.endpoint_url, "shex_prefixes": prefixes_str, "chunk_titles": chunk_titles, "retrieved_chunks": set()
                }
                # benchmark model: S, M and L 
                # benchmark prompt: from SCiQA test set
                # benchmark method: 2, 3, 4, 5, 6, 7
                if args.benchmark_mode:
                    benckmark_method_mapping = {"2": "VoID", "3": "ShEX", "4": "Sample"}
                    added_context = {
                        "benchmark_method": args.benchmark_method, 
                        "benchmark_prompt": user_input, 
                        "benchmark_model": args.benchmark_model,
                        "benchmark_method_description": benckmark_method_mapping[args.benchmark_method],
                        "benchmark_mode": args.benchmark_mode,
                        "ground_truth_query": ground_truth_query,
                        "ground_truth_id": ground_truth_id,

                    }
                    context.update(added_context)

                console.print(f"Using the following context: {context}", style="bold blue")

                for event in graph.stream(
                    {"messages": [("user", user_input)]}, config, context=context):
                    for value in event.values():
                        if value["messages"]:
                            latest_message = value["messages"][-1]
                            # tool call message
                            if hasattr(latest_message, "tool_calls") and len(latest_message.tool_calls) > 0:
                                tool_name = latest_message.tool_calls[0]["name"]
                                # if tool_name == "shex_retriever":
                                continue
                            # tool result message
                            elif hasattr(latest_message, "type") and latest_message.type == "tool":
                                continue
                            console.print(f"\nAssistant:\n{latest_message.content}", style="bold green")
