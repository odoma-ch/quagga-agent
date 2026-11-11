import argparse
import json
import os
import re
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

FAILED_QUERY_MARKERS = (
    "Error executing SPARQL query",
    "Query timed out after",
    "A bad request has been sent to the endpoint",
)
QUESTION_TO_ANSWER_RE = re.compile(
    r"## Question to Answer\s*(.+?)(?:\n## |\Z)",
    re.DOTALL,
)
INTERNAL_RESULT_MARKERS = (
    "http://www.openlinksw.com/schemas/virtrdf#",
    "http://www.w3.org/ns/ldp#",
)


def _msg_value(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _content_as_text(msg: Any) -> str:
    content = _msg_value(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = next(
                    (
                        block.get(key)
                        for key in ("text", "content", "reasoning", "output")
                        if block.get(key)
                    ),
                    None,
                )
                parts.append(str(text) if text else str(block))
            else:
                parts.append(str(block))
        joined = "\n".join(part for part in parts if part).strip()
        if joined:
            return joined
    return str(content)


def _is_tool_message(msg: Any) -> bool:
    return _msg_value(msg, "type") == "tool"


def _is_ai_message(msg: Any) -> bool:
    return _msg_value(msg, "type") == "ai"


def _is_human_message(msg: Any) -> bool:
    return _msg_value(msg, "type") == "human"


def _is_failed_tool_message(msg: Any) -> bool:
    if not _is_tool_message(msg):
        return False
    if _msg_value(msg, "name", "") != "run_sparql_tool":
        return False
    text = _content_as_text(msg)
    return any(marker in text for marker in FAILED_QUERY_MARKERS)


def _serialize_message(msg: Any, index: int) -> dict[str, Any]:
    payload = (
        msg.model_dump(mode="json")
        if hasattr(msg, "model_dump")
        else {
            "type": _msg_value(msg, "type"),
            "content": _msg_value(msg, "content"),
        }
    )
    payload["index"] = index
    payload["content_text"] = _content_as_text(msg)
    return payload


def _extract_tool_calls(msg: Any) -> list[dict[str, Any]]:
    tool_calls = _msg_value(msg, "tool_calls", []) or []
    extracted: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            extracted.append(tool_call)

    additional_kwargs = _msg_value(msg, "additional_kwargs", {}) or {}
    raw_tool_calls = additional_kwargs.get("tool_calls", [])
    for tool_call in raw_tool_calls:
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {}) or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments}
            extracted.append(
                {
                    "id": tool_call.get("id"),
                    "name": function.get("name"),
                    "args": (
                        arguments
                        if isinstance(arguments, dict)
                        else {"raw_arguments": arguments}
                    ),
                    "type": tool_call.get("type"),
                }
            )
    return extracted


def _extract_query_from_tool_call(tool_call: dict[str, Any]) -> str:
    args = tool_call.get("args", {}) or {}
    if isinstance(args, str):
        return args
    if not isinstance(args, dict):
        return str(args)
    for key in ("query", "sparql_query", "sparql", "q", "raw_arguments"):
        value = args.get(key)
        if value:
            return str(value)
    return json.dumps(args, ensure_ascii=False)


def _extract_user_question(msg: Any) -> str | None:
    if not _is_human_message(msg):
        return None
    text = _content_as_text(msg).strip()
    if not text:
        return None
    if text.startswith("Here is the previously cached ontology report"):
        return None
    match = QUESTION_TO_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _looks_like_good_result(query: str, result_text: str) -> bool:
    if not result_text.strip():
        return False
    if any(marker in result_text for marker in FAILED_QUERY_MARKERS):
        return False
    if "Results (showing 0/0 rows)" in result_text or "No results" in result_text:
        return False
    if "?s ?p ?o" in query.replace("\n", " "):
        return False
    if "Results (showing" not in result_text:
        return False

    data_lines = [
        line.strip()
        for line in result_text.splitlines()
        if line.strip().startswith("| http")
    ]
    if data_lines and all(
        any(marker in line for marker in INTERNAL_RESULT_MARKERS) for line in data_lines
    ):
        return False
    return True


def _load_raw_checkpoint_messages(thread_id: str) -> list[Any]:
    with PostgresSaver.from_conn_string(_build_postgres_conn_string()) as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        latest = checkpointer.get(config)
        if latest:
            return latest["channel_values"].get("messages", [])
    return []


def _collect_successful_query_messages(messages: Sequence[Any]) -> list[Any]:
    successful_tool_ids: set[str] = set()
    failed_tool_ids: set[str] = set()

    for msg in messages:
        if not _is_tool_message(msg):
            continue
        tool_call_id = str(_msg_value(msg, "tool_call_id", ""))
        if not tool_call_id:
            continue
        if _is_failed_tool_message(msg):
            failed_tool_ids.add(tool_call_id)
        else:
            successful_tool_ids.add(tool_call_id)

    successful_tool_ids -= failed_tool_ids
    if not successful_tool_ids:
        return []

    selected_indexes: set[int] = set()
    last_human_idx: int | None = None

    for idx, msg in enumerate(messages):
        if _is_human_message(msg):
            last_human_idx = idx
            continue

        if _is_ai_message(msg):
            tool_calls = _msg_value(msg, "tool_calls", []) or []
            matching_ids = {
                str(tool_call.get("id", ""))
                for tool_call in tool_calls
                if isinstance(tool_call, dict)
                and str(tool_call.get("id", "")) in successful_tool_ids
            }
            if matching_ids:
                selected_indexes.add(idx)
                if last_human_idx is not None:
                    selected_indexes.add(last_human_idx)
                continue

        if _is_tool_message(msg):
            tool_call_id = str(_msg_value(msg, "tool_call_id", ""))
            if tool_call_id in successful_tool_ids:
                selected_indexes.add(idx)
                if last_human_idx is not None:
                    selected_indexes.add(last_human_idx)

    expanded_indexes = set(selected_indexes)
    for idx in sorted(selected_indexes):
        next_idx = idx + 1
        if (
            next_idx < len(messages)
            and _is_ai_message(messages[next_idx])
            and not (_msg_value(messages[next_idx], "tool_calls", []) or [])
        ):
            expanded_indexes.add(next_idx)

    return [messages[idx] for idx in sorted(expanded_indexes)]


def get_successful_query_messages(thread_id: str) -> list[dict]:
    raw_messages = _load_raw_checkpoint_messages(thread_id)
    successful_messages = _collect_successful_query_messages(raw_messages)
    successful_messages_serialized = [
        _serialize_message(msg, idx) for idx, msg in enumerate(successful_messages)
    ]
    with open("successful_messages.json", "w", encoding="utf-8") as f:
        json.dump(successful_messages_serialized, f, ensure_ascii=False, indent=2)
    return successful_messages_serialized


def get_useful_queries_by_question(thread_id: str) -> list[dict[str, Any]]:
    raw_messages = _load_raw_checkpoint_messages(thread_id)
    if not raw_messages:
        return []

    grouped_results: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    pending_queries: dict[str, dict[str, Any]] = {}

    def ensure_group(question: str) -> dict[str, Any]:
        nonlocal current_group
        if current_group is None or current_group["user_question"] != question:
            current_group = {"user_question": question, "queries": []}
            grouped_results.append(current_group)
        return current_group

    for msg in raw_messages:
        extracted_question = _extract_user_question(msg)
        if extracted_question:
            ensure_group(extracted_question)
            pending_queries = {}
            continue

        if current_group is None:
            continue

        if _is_ai_message(msg):
            for tool_call in _extract_tool_calls(msg):
                tool_call_id = str(tool_call.get("id", "") or "")
                query = _extract_query_from_tool_call(tool_call).strip()
                if tool_call_id and query:
                    pending_queries[tool_call_id] = {
                        "sparql_query": query,
                        "ai_message": _content_as_text(msg).strip(),
                    }
            continue

        if _is_tool_message(msg):
            tool_call_id = str(_msg_value(msg, "tool_call_id", "") or "")
            if not tool_call_id or tool_call_id not in pending_queries:
                continue
            query_entry = pending_queries.pop(tool_call_id)
            result_text = _content_as_text(msg).strip()
            if _looks_like_good_result(query_entry["sparql_query"], result_text):
                current_group["queries"].append(
                    {
                        "sparql_query": query_entry["sparql_query"],
                        "result": result_text,
                    }
                )

    useful_grouped_results = [group for group in grouped_results if group["queries"]]
    with open("useful_queries_by_question.json", "w", encoding="utf-8") as f:
        json.dump(useful_grouped_results, f, ensure_ascii=False, indent=2)
    return useful_grouped_results


def _tokenize_query_for_jaccard(query: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[A-Za-z_][A-Za-z0-9_:/#.-]*|\?[A-Za-z_][A-Za-z0-9_]*", query.lower()
        )
        if token
    }


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    intersection = a & b
    return 1.0 - (len(intersection) / len(union))


def _select_most_diverse_queries(
    queries: list[dict[str, Any]],
    max_queries: int = 5,
) -> list[dict[str, Any]]:
    if len(queries) <= max_queries:
        return queries

    token_sets = [
        _tokenize_query_for_jaccard(str(query_entry.get("sparql_query", "")))
        for query_entry in queries
    ]

    selected_indexes = [0]
    remaining_indexes = set(range(1, len(queries)))

    while remaining_indexes and len(selected_indexes) < max_queries:
        best_idx = None
        best_score = -1.0
        for idx in remaining_indexes:
            distances = [
                _jaccard_distance(token_sets[idx], token_sets[selected_idx])
                for selected_idx in selected_indexes
            ]
            score = min(distances) if distances else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected_indexes.append(best_idx)
        remaining_indexes.remove(best_idx)

    return [queries[idx] for idx in selected_indexes]


def format_useful_queries_as_text_blob(
    grouped_queries: list[dict[str, Any]],
    max_queries_total: int = 10,
) -> str:
    all_queries: list[dict[str, Any]] = []
    for group in grouped_queries:
        user_question = str(group.get("user_question", "")).strip()
        if not user_question:
            continue
        for query_entry in group.get("queries", []):
            all_queries.append(
                {
                    "user_question": user_question,
                    "sparql_query": query_entry.get("sparql_query", ""),
                    "result": query_entry.get("result", ""),
                }
            )

    diverse_queries = _select_most_diverse_queries(
        all_queries,
        max_queries=max_queries_total,
    )
    print(f"Total useful queries across all questions: {len(all_queries)}")

    lines: list[str] = []
    for query_entry in diverse_queries:
        sparql_query = str(query_entry.get("sparql_query", "")).strip()
        lines.append("\n<query>")
        if sparql_query:
            lines.append(sparql_query)
        lines.append("</query>")
    return "\n".join(lines).strip()


def save_useful_queries_text_blob(
    grouped_queries: list[dict[str, Any]],
    output_path: str = "useful_queries_by_question.txt",
    max_queries_total: int = 10,
) -> str:
    text_blob = format_useful_queries_as_text_blob(
        grouped_queries,
        max_queries_total=max_queries_total,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_blob)
    return text_blob


def _build_postgres_conn_string() -> str:
    postgres_db = "sparql_traces"
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    if not all((postgres_user, postgres_host, postgres_password)):
        raise ValueError(
            "Missing one or more Postgres env vars: POSTGRES_USER, POSTGRES_HOST, POSTGRES_PASSWORD"
        )
    return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}/{postgres_db}"


def get_checkpoint_messages(thread_id: str) -> list[dict]:
    with PostgresSaver.from_conn_string(_build_postgres_conn_string()) as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        latest = checkpointer.get(config)
        if latest:
            messages = latest["channel_values"].get("messages", [])
            all_msgs = []
            for i, msg in enumerate(messages):
                all_msgs.append(
                    {
                        "index": i,
                        "role": msg.type,
                        "content": msg.content,
                        "metadata": getattr(msg, "additional_kwargs", {}),
                    }
                )
            return all_msgs
        return []


if __name__ == "__main__":
    thread_id = "OpenCitations Meta_session"
    successful_messages = get_successful_query_messages(thread_id)
    useful_queries = get_useful_queries_by_question(thread_id)
    text_blob = format_useful_queries_as_text_blob(
        useful_queries,
        max_queries_total=15,
    )
    print(text_blob)
