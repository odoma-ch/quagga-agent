import os
import re
import sys
import json
import uuid
import asyncio
import logging
import threading
from datetime import datetime
from langchain_core.messages.utils import count_tokens_approximately

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from string import Template
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv
load_dotenv()

from tools import RunSPARQLTool
from agent_utils.agent_runtime_state import runtime_state
from agent_utils.provider_router import build_llm_from_health, periodic_health_check, update_runtime_llm

logger = logging.getLogger(__name__)
GRAPH_DESCRIPTIONS_DIR = Path(__file__).parent / "graph_descriptions"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Matches XML-style tool call syntax that some models emit as plain text
# instead of using the structured tool_calls field.
_TEXT_TOOL_CALL_RE = re.compile(r"<invoke\s|<tool_call\b|<function_calls\b", re.IGNORECASE)
MAX_TEXT_TOOL_CALL_RETRIES = 2
_STEP_HEADER_RE = re.compile(r"^\s*Step\s+\d+\s*:\s*.*$", re.IGNORECASE)



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


_HEALTH_MONITOR_STARTED = False
_HEALTH_MONITOR_LOCK = threading.Lock()


async def _setup_routed_llm() -> None:
    base_url = os.getenv("PSNC_OKD_BASE_URL")
    model = os.getenv("PSNC_OKD_LARGE_MODEL")
    api_key = os.getenv("PSNC_OKD_API_KEY")
    fallback_base_url = os.getenv("GRAPHCHAT_OPENROUTER_BASE_URL")
    fallback_api_key = os.getenv("GRAPHCHAT_OPENROUTER_API_KEY")
    fallback_model = os.getenv("GRAPHCHAT_OPENROUTER_MODEL")

    startup_llm, is_healthy, reason = await build_llm_from_health(
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        fallback_base_url=fallback_base_url,
        fallback_api_key=fallback_api_key,
        fallback_model_name=fallback_model,
        temperature=0.0,
        extra_body={"reasoning": {"enabled": True}},
    )
    update_runtime_llm(runtime_state, startup_llm)
    if is_healthy:
        logger.info(f"Using model: {model}")
        logger.info(f"Using endpoint: {base_url}")
    else:
        logger.warning(f"Primary subagent model unhealthy at startup ({reason}), using fallback")
        logger.info(f"Using model: {fallback_model}")
        logger.info(f"Using endpoint: {fallback_base_url}")


def _start_routed_llm_health_monitor() -> None:
    global _HEALTH_MONITOR_STARTED
    with _HEALTH_MONITOR_LOCK:
        if _HEALTH_MONITOR_STARTED:
            return

        def _runner() -> None:
            asyncio.run(
                periodic_health_check(
                    base_url=os.getenv("PSNC_OKD_BASE_URL"),
                    api_key=os.getenv("PSNC_OKD_API_KEY"),
                    model_name=os.getenv("PSNC_OKD_LARGE_MODEL"),
                    fallback_base_url=os.getenv("GRAPHCHAT_OPENROUTER_BASE_URL"),
                    fallback_api_key=os.getenv("GRAPHCHAT_OPENROUTER_API_KEY"),
                    fallback_model_name=os.getenv("GRAPHCHAT_OPENROUTER_MODEL"),
                    temperature=1.0,
                    top_p=0.9,
                    extra_body={"reasoning": {"enabled": True}},
                    runtime_state=runtime_state,
                    llm_tools=None,
                    interval=120,
                )
            )

        threading.Thread(target=_runner, daemon=True, name="sparql-subagent-llm-health").start()
        _HEALTH_MONITOR_STARTED = True


def _ensure_routed_llm_ready() -> None:
    if runtime_state.llm is None:
        asyncio.run(_setup_routed_llm())
    _start_routed_llm_health_monitor()


def build_subagent(endpoint: str):
    """Build and compile a SPARQL subagent bound to the given endpoint."""
    _ensure_routed_llm_ready()
    tools = [RunSPARQLTool(endpoint=endpoint)]
    # initial_soft_trim_applied = False

    def agent(state: AgentState) -> AgentState:
        # nonlocal initial_soft_trim_applied
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

        current_llm = runtime_state.llm
        if current_llm is None:
            _ensure_routed_llm_ready()
            current_llm = runtime_state.llm

        logger.info(
            f"Messages token count for LLM: {count_tokens_approximately(messages)}"
        )
        response = current_llm.bind_tools(tools).invoke(messages)
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

    postgres_db = "sparql_traces"
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_conn_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}/{postgres_db}"

    pool = ConnectionPool(
        conninfo=postgres_conn_string,
        open=False,
        min_size=2,
        max_size=8,
        max_lifetime=300,
        max_idle=60,
        timeout=30.0,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )

    pool.open(wait=True)
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return graph.compile(checkpointer=checkpointer)


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


def _build_step_prompts(exploration_prompt: str) -> list[tuple[str, str]]:
    """Split a strategy prompt into per-step prompts while preserving shared context."""
    lines = exploration_prompt.splitlines()
    step_indices = [idx for idx, line in enumerate(lines) if _STEP_HEADER_RE.match(line)]
    if not step_indices:
        return []

    preamble = "\n".join(lines[: step_indices[0]]).strip()
    step_prompts: list[tuple[str, str]] = []

    for pos, start_idx in enumerate(step_indices):
        end_idx = step_indices[pos + 1] if pos + 1 < len(step_indices) else len(lines)
        step_label = lines[start_idx].strip()
        step_block = "\n".join(lines[start_idx:end_idx]).strip()

        prompt_parts = [
            preamble,
            "",
            "Execute ONLY this single strategy step now.",
            "Use context already established in this thread from previous steps.",
            "Do not execute future steps in this turn.",
            "",
            step_block,
        ]
        step_prompt = "\n".join(part for part in prompt_parts if part != "").strip()
        step_prompts.append((step_label, step_prompt))

    return step_prompts


def run_pipeline(endpoint: str, kg_name: str, question: str) -> str:
    """Run the two-node pipeline: explore KG (with caching) then answer the user question."""
    from langgraph.errors import GraphRecursionError as _GRE
    start_time = datetime.now()
    print(f"[DEBUG] run_pipeline called at {start_time}: kg={kg_name!r}, question={question!r}", flush=True)
    subagent = build_subagent(endpoint)
    thread_id = f"{kg_name}_{uuid.uuid4().hex}"
    print(f"[DEBUG] Thread ID: {thread_id}")
    desc_path = GRAPH_DESCRIPTIONS_DIR / f"{kg_name}"

    def explore_kg_node(state: PipelineState) -> PipelineState:
        """Node 1: load cached KG description or run full SPARQL exploration."""
        if desc_path.exists():
            print(f"[Explorer] Cached description found for \'{kg_name}\'. Skipping exploration run...")
        else:
            print(f"[Explorer] No cache found for \'{kg_name}\', detecting graph structure...")
            graphs_question = f"""For the {kg_name} endpoint: {endpoint}
Run this ONE query to detect if this knowledge graph uses named graphs:
SELECT DISTINCT ?g WHERE {{ GRAPH ?g {{ ?s ?p ?o }} }}

After looking at the result, your final reply MUST end with EXACTLY ONE of these tokens on its own final line, and nothing after it:
MULTI_GRAPH   — if the result contains one or more proper URIs (starting with http:// or https://).
SINGLE_GRAPH  — if no results are returned, returns no rows, or contains only blank nodes / non-URI graph names.
Do not include both tokens. Do not add quotes, punctuation, or any text after the token."""
            
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
            step_prompts = _build_step_prompts(exploration_prompt)

            if not step_prompts:
                report = _stream_subagent(exploration_prompt, subagent, thread_id)
                desc_path.write_text(report)
            else:
                desc_path.write_text("", encoding="utf-8")
                total_steps = len(step_prompts)
                for idx, (step_label, step_prompt) in enumerate(step_prompts, start=1):
                    print(f"[Explorer] Running {step_label} ({idx}/{total_steps})...")
                    step_result = _stream_subagent(step_prompt, subagent, thread_id)
                    step_result = re.sub(
                        r"<think>.*?</think>", "", step_result, flags=re.IGNORECASE | re.DOTALL).strip()
                    with desc_path.open("a", encoding="utf-8") as f:
                        f.write(f"## {step_label}\n\n{step_result}\n\n")

            print(f"[Explorer] Saved exploration report to {desc_path}")

        return state

    def answer_question_node(state: PipelineState) -> PipelineState:
        """Node 2: answer the user question using the exploration context already in the thread."""
        MAX_QUERIES = 80
        from datetime import date
        current_date = date.today().strftime("%a %b %d %Y")
        prompt = _load_prompt("answer_question.txt", kg_name=kg_name, endpoint=endpoint, question=state["question"], current_date=current_date)
        if desc_path.exists():
            cached = desc_path.read_text()
            prompt = (
                f"Cached ontology report for knowledge graph {kg_name}:\n\n{cached}\n\n"
                "----\n\n"
                f"{prompt}"
            )
        print(f"[Answer Question] Prompt:\n{prompt}\n")

        from langgraph.errors import GraphRecursionError as _SubagentRecursionError
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
        queries_made: list[str] = []
        final_summary = ""
        limit_hit = False

        try:
            for event in subagent.stream(
                {"messages": [SystemMessage(content=prompt)]},
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
    end_time = datetime.now()
    print(f"[DEBUG] run_pipeline finished successfully and took: {(end_time - start_time).total_seconds()} seconds: final_summary length={len(result['final_summary'])}", flush=True)
    return result["final_summary"]


if __name__ == "__main__":
    opencitation_questions = [
        "What is the title and publication date of publication with DOI 10.1162/qss_a_00023?",
        "How many publications are in the OpenCitations dataset?",
        "What is the most cited article in OpenCitations?", # should fail
        "How many journal articles contained in Open Citations Meta were published in 2024?",
        "I want to query OpenCitations. How many publications are indexed in there? break them down by publication typology",
        "Who is the most prolific Humanities author in terms of number of publications?", # should fail
        "What are the latest 10 articles published in the journal with ISSN 2641-3337 (Quantitative Science Studies)?",
        "What are the publications by the author with ORCID 0000-0003-0530-4305 published in 2024?",
        "Which publications contain the words 'open science' in the title and were published in 2024?",
        "Which book chapters with 'digital humanities' in their title were published in 2023?",
        "Which articles about 'COVID-19' published in 2023 have more than 5 authors?",
    ]
    question = opencitation_questions[6]
    print(f"Running pipeline for question: {question}")
    run_pipeline(
        endpoint="https://sparql.opencitations.net/meta",
        kg_name="OpenCitations Meta",
        question=question,
    )

    # ehri_questions = [
    #     "Which archival institutions in the EHRI Knowledge Graph are located in Argentina, and what are their names?",
    #     "How many archival descriptions does the EHRI KG contain?",
    #     "Get records with the subject 'Newspapers' in EHRI KG",
    #     "Which countries are listed in the EHRI Knowledge Graph, and what are their names?",
    #     "What are the summaries in Dutch for the archival records held by Kazerne Dossin?"
    # ]
    # question = ehri_questions[1]
    # print(f"Running pipeline for question: {question}")
    # print(run_pipeline(
    #     endpoint="https://lod.ehri-project-test.eu/sparql",
    #     kg_name="EHRI - KG",
    #     question=question,
    # ))

    # gesis_questions = [
    #     "how many articles are in the GESIS KG?",
    #     "Which variables have been used in the study ZA5282?",
    #     "Which organizations have the most studies archived at GESIS?"
    # ]
    # question = gesis_questions[0]
    # print(f"Running pipeline for question: {question}")
    # run_pipeline(
    #     endpoint="https://data.gesis.org/gesiskg/sparql",
    #     kg_name="GESIS KG",
    #     question=question,
    # )
