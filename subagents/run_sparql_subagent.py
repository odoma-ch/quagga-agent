import os
import re
import sys
import json
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from string import Template
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

from tools import RunSPARQLTool

logger = logging.getLogger(__name__)
GRAPH_DESCRIPTIONS_DIR = Path(__file__).parent / "graph_descriptions"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Matches XML-style tool call syntax that some models emit as plain text
# instead of using the structured tool_calls field.
_TEXT_TOOL_CALL_RE = re.compile(r"<invoke\s|<tool_call\b|<function_calls\b", re.IGNORECASE)
MAX_TEXT_TOOL_CALL_RETRIES = 2


def _looks_like_text_tool_call(content: str) -> bool:
    return bool(_TEXT_TOOL_CALL_RE.search(content))


def _load_prompt(name: str, **kwargs) -> str:
    return Template((PROMPTS_DIR / name).read_text()).substitute(**kwargs)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class PipelineState(TypedDict):
    kg_name: str
    question: str
    final_summary: str


llm = ChatOpenAI(
    base_url=os.getenv("PSNC_OKD_BASE_URL"),
    model=os.getenv("PSNC_OKD_LARGE_MODEL"),
    api_key=os.getenv("PSNC_OKD_API_KEY"),
    temperature=0,
    extra_body={
        "reasoning": {"enabled": True}
    }
)
logger.info(f"Using model: {os.getenv('PSNC_OKD_LARGE_MODEL')}")
logger.info(f"Using endpoint: {os.getenv('PSNC_OKD_BASE_URL')}")


def build_subagent(endpoint: str):
    """Build and compile a SPARQL subagent bound to the given endpoint."""
    tools = [RunSPARQLTool(endpoint=endpoint)]
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState) -> AgentState:
        messages = list(state["messages"])
        last = messages[-1] if messages else None
        extra: list = []

        # If the model previously emitted a tool call as plain text, inject a
        # corrective message so it retries using the proper structured format.
        if (
            isinstance(last, AIMessage)
            and not getattr(last, "tool_calls", None)
            and isinstance(last.content, str)
            and _looks_like_text_tool_call(last.content)
        ):
            correction = HumanMessage(
                content="Please call the tool using the proper JSON function-calling format, not XML tags. Try again."
            )
            messages = messages + [correction]
            extra = [correction]

        response = llm_with_tools.invoke(messages)
        return {"messages": extra + [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"

        content = last.content if isinstance(last, AIMessage) and isinstance(last.content, str) else ""
        if _looks_like_text_tool_call(content):
            # Count consecutive text-format AI messages before this one to enforce the cap.
            retries = 0
            for msg in reversed(state["messages"][:-1]):
                if isinstance(msg, HumanMessage):
                    continue  # skip corrective messages injected by agent node
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                    if _looks_like_text_tool_call(msg.content if isinstance(msg.content, str) else ""):
                        retries += 1
                    else:
                        break
                else:
                    break
            if retries < MAX_TEXT_TOOL_CALL_RETRIES:
                return "agent"

        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "agent": "agent", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


def _extract_query_from_tool_call(tool_call: dict) -> str:
    """Return a readable query string from a tool call."""
    args = tool_call.get("args", {}) or {}
    for key in ("query", "sparql_query", "sparql", "q"):
        value = args.get(key)
        if value:
            return str(value)
    return json.dumps(args, ensure_ascii=True, indent=2)


def _stream_subagent(question: str, subagent, thread_id: str) -> str:
    """Send a message to the subagent, stream and print all events, return the final AI response."""
    print(f"\n🔍 {question}\n")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    final_response = ""

    for event in subagent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates",
    ):
        for _, update in event.items():
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tool_call in msg.tool_calls:
                        query_text = _extract_query_from_tool_call(tool_call)
                        print(f"[Tool Query]\n{query_text}\n")
                elif isinstance(msg, ToolMessage):
                    print(f"[Tool Response]\n{msg.content}\n")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"[AI Response]\n{msg.content}\n")
                    content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if not _looks_like_text_tool_call(content_str):
                        final_response = content_str

    return final_response


def run_pipeline(endpoint: str, kg_name: str, question: str) -> str:
    """Run the two-node pipeline: explore KG (with caching) then answer the user question."""
    from langgraph.errors import GraphRecursionError as _GRE
    print(f"[DEBUG] run_pipeline called: kg={kg_name!r}, question={question!r}", flush=True)
    subagent = build_subagent(endpoint)
    thread_id = f"{kg_name}_session"
    desc_path = GRAPH_DESCRIPTIONS_DIR / f"{kg_name}"

    def explore_kg_node(state: PipelineState) -> PipelineState:
        """Node 1: load cached KG description or run full SPARQL exploration."""
        if desc_path.exists():
            print(f"[Explorer] Cached description found for \'{kg_name}\', loading into context...")
            cached = desc_path.read_text()
            config = {"configurable": {"thread_id": thread_id}}                                                          
            subagent.update_state(config, {                                                                              
                "messages": [                                                                                            
                    HumanMessage(content=f"Here is the previously cached ontology report for the knowledge graph {kg_name}:\n\n{cached}"),                                                                                                         
                    AIMessage(content="Understood. I have the KG description loaded and will use it to answer your question."),                                                                                                                     
                ]                                                                                                        
            }) 
        else:
            print(f"[Explorer] No cache found for \'{kg_name}\', detecting graph structure...")
            graphs_question = f"""For the {kg_name} endpoint: {endpoint}
Run this ONE query to detect if this knowledge graph uses named graphs:
SELECT DISTINCT ?g WHERE {{ GRAPH ?g {{ ?s ?p ?o }} }}

Return ONLY "MULTI_GRAPH" if you find one or more proper URIs (starting with http:// or https://), or "SINGLE_GRAPH" if you find no named graphs or only blank nodes/non-URI graph names."""
            
            print(f"[Detector] Checking graph structure...\n")
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
            graph_type = "UNKNOWN"

            for event in subagent.stream(
                {"messages": [HumanMessage(content=graphs_question)]},
                config=config,
                stream_mode="updates",
            ):
                for _, update in event.items():
                    for msg in update.get("messages", []):
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            for tool_call in msg.tool_calls:
                                query_text = _extract_query_from_tool_call(tool_call)
                                print(f"[Detector Query]\n{query_text}\n")
                        elif isinstance(msg, ToolMessage):
                            print(f"[Detector Response]\n{msg.content}\n")
                        elif isinstance(msg, AIMessage) and msg.content:
                            content_upper = msg.content.upper()
                            if "MULTI_GRAPH" in content_upper:
                                graph_type = "MULTI"
                                print(f"[Detector] Detected multiple named graphs\n")
                            elif "SINGLE_GRAPH" in content_upper:
                                graph_type = "SINGLE"
                                print(f"[Detector] Detected single unified graph\n")
            
            prompt_file = "exploration.txt" if graph_type == "MULTI" else "exploration_single_graph.txt"
            print(f"[Explorer] Using {prompt_file} strategy for full exploration...")
            exploration_prompt = _load_prompt(prompt_file, kg_name=kg_name, endpoint=endpoint)
            report = _stream_subagent(exploration_prompt, subagent, thread_id)
            desc_path.write_text(report)
            print(f"[Explorer] Saved exploration report to {desc_path}")

        return state

    def answer_question_node(state: PipelineState) -> PipelineState:
        """Node 2: answer the user question using the exploration context already in the thread."""
        MAX_QUERIES = 80
        prompt = _load_prompt("answer_question.txt", kg_name=kg_name, endpoint=endpoint, question=state["question"])
        print(f"[Answer Question] Prompt:\n{prompt}\n")

        from langgraph.errors import GraphRecursionError as _SubagentRecursionError
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
        queries_made: list[str] = []
        final_summary = ""
        limit_hit = False

        try:
            for event in subagent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="updates",
            ):
                for _, update in event.items():
                    for msg in update.get("messages", []):
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            for tool_call in msg.tool_calls:
                                query_text = _extract_query_from_tool_call(tool_call)
                                print(f"[Tool Query]\n{query_text}\n")
                                queries_made.append(query_text)
                        elif isinstance(msg, ToolMessage):
                            print(f"[Tool Response]\n{msg.content}\n")
                        elif isinstance(msg, AIMessage) and msg.content:
                            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                            print(f"[AI Response]\n{content_str}\n")
                            if not _looks_like_text_tool_call(content_str):
                                final_summary = content_str

                if len(queries_made) >= MAX_QUERIES:
                    print(f"[Limit] Reached max_queries={MAX_QUERIES}. Stopping early.\n")
                    limit_hit = True
                    break
        except _SubagentRecursionError:
            print(f"[Limit] Subagent hit recursion limit after {len(queries_made)} queries. Using accumulated summary.\n")
            limit_hit = True

        if limit_hit:
            queries_summary = "\n".join(
                f"{i + 1}. {q}" for i, q in enumerate(queries_made)
            )
            summary_prompt = _load_prompt(
                "query_limit_summary.txt",
                num_queries=len(queries_made),
                question=state["question"],
                kg_name=kg_name,
                endpoint=endpoint,
                queries_summary=queries_summary,
            )
            final_summary = _stream_subagent(summary_prompt, subagent, thread_id)

        return {**state, "final_summary": final_summary}

    pipeline = StateGraph(PipelineState)
    pipeline.add_node("explore_kg", explore_kg_node)
    pipeline.add_node("answer_question", answer_question_node)
    pipeline.add_edge(START, "explore_kg")
    pipeline.add_edge("explore_kg", "answer_question")
    pipeline.add_edge("answer_question", END)

    try:
        result = pipeline.compile().invoke({
            "kg_name": kg_name,
            "question": question,
            "final_summary": "",
        })
    except _GRE as e:
        print(f"[DEBUG] GraphRecursionError raised INSIDE run_pipeline (subagent hit its own recursion limit): {e}", flush=True)
        return "I was unable to fully answer the question — the subagent exhausted its step budget during exploration. Try a more specific question."
    print(f"[DEBUG] run_pipeline finished successfully, final_summary length={len(result['final_summary'])}", flush=True)
    return result["final_summary"]


if __name__ == "__main__":
    # opencitation_questions = [
    #     "What is the title and publication date of publication with DOI 10.1162/qss_a_00023?",
    #     "How many publications are in the OpenCitations dataset?",
    #     "What is the most cited article in OpenCitations?", # should fail
    #     "How many journal articles contained in Open Citations Meta were published in 2024?",
    #     "I want to query OpenCitations. How many publications are indexed in there? break them down by publication typology",
    #     "Who is the most prolific Humanities author in terms of number of publications?", # should fail
    #     "What are the latest 10 articles published in the journal with ISSN 2641-3337 (Quantitative Science Studies)?",
    #     "What are the publications by the author with ORCID 0000-0003-0530-4305 published in 2024?",
    #     "Which publications contain the words 'open science' in the title and were published in 2024?",
    #     "Which book chapters with 'digital humanities' in their title were published in 2023?",
    #     "Which articles about 'COVID-19' published in 2023 have more than 5 authors?",
    # ]
    # question = opencitation_questions[6]
    # print(f"Running pipeline for question: {question}")
    # run_pipeline(
    #     endpoint="https://sparql.opencitations.net/meta",
    #     kg_name="OpenCitations Meta",
    #     question=question,
    # )

    ehri_questions = [
        "Which archival institutions in the EHRI Knowledge Graph are located in Argentina, and what are their names?",
        "How many archival descriptions does the EHRI KG contain?",
        "Get records with the subject 'Newspapers' in EHRI KG",
        "Which countries are listed in the EHRI Knowledge Graph, and what are their names?",
        "What are the summaries in Dutch for the archival records held by Kazerne Dossin?"
    ]
    question = ehri_questions[1]
    print(f"Running pipeline for question: {question}")
    print(run_pipeline(
        endpoint="https://lod.ehri-project-test.eu/sparql",
        kg_name="EHRI - KG",
        question=question,
    ))

    # gesis_questions = [
    #     "how many articles are in the GESIS KG?"
    # ]
    # question = gesis_questions[0]
    # print(f"Running pipeline for question: {question}")
    # run_pipeline(
    #     endpoint="https://data.gesis.org/gesiskg/sparql",
    #     kg_name="GESIS KG",
    #     question=question,
    # )