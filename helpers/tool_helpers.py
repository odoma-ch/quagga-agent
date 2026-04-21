from dotenv import load_dotenv
load_dotenv()

import json
import html
import requests

from models import State
from langgraph.graph import END

_TOOL_DISPLAY_NAMES = {
    "run_sparql_subagent_tool": "Querying knowledge graph",
    "validate_sparql_tool": "Validating and executing SPARQL query",
    "describe_knowledge_graphs_tool": "Checking available knowledge graphs",
    "scope_checker_tool": "Selecting knowledge graph(s)",
}
_TERMINAL_SPARQL_MARKER = "The SPARQL has been executed:"
_NO_KG_SELECTED_MESSAGE = "No knowledge graph selected. Call scope_checker_tool first."


def _format_tool_call(message) -> str:
    """Render an LLM tool-call message as a readable markdown block."""
    parts = []
    for tc in message.tool_calls:
        name = tc.get("name", "unknown_tool")
        args = tc.get("args", {})
        display = _TOOL_DISPLAY_NAMES.get(name, name)
        header = f"**{display}**"

        # TODO: check if we need to show args
        # if args:
        #     # Show the arguments compactly
        #     args_str = ", ".join(f"`{k}`: {v}" for k, v in args.items())
        #     parts.append(f"{header}\n> {args_str}\n")
        # else:
        parts.append(f"{header}\n")
    return "\n".join(parts)


def _format_tool_result(message) -> str:
    """Render a tool result message as a collapsible markdown details block."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    tool_name = getattr(message, "name", "tool")
    display = _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)

    # Sometimes the model returns a JSON blob that encodes the *next* tool call
    # (instead of emitting a native tool_calls message)
    try:
        parsed = json.loads(content)
    except Exception:
        try:
            parsed = json.loads(html.unescape(content))
        except Exception:
            parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        rendered_calls = []
        for tc in parsed["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            if not name and isinstance(tc.get("name="), str):
                name = tc.get("name=")
            if not name:
                continue
            rendered_calls.append(f"**{_TOOL_DISPLAY_NAMES.get(name, name)}**")
        if rendered_calls:
            return "\n".join(rendered_calls) + "\n"

    # If the tool returns JSON as a raw string, render it explicitly as JSON.
    if isinstance(parsed, (dict, list)):
        content = f"```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"

    has_details = "<details" in content.lower()

    if tool_name == "validate_sparql_tool":
        return (
            f"<details>\n<summary>Result from {display}</summary>\n\n"
            f"{content}\n"
            f"</details>\n\n"
        )
    elif has_details:
        return f"**Result from {display}**\n\n{content}\n\n"
    else:
        return (
            f"<details>\n<summary>Result from {display}</summary>\n\n"
            f"{content}\n"
            f"</details>\n\n"
        )


def _extract_terminal_tool_answer(message) -> str | None:
    """Return answer text when tool output should be surfaced as final assistant text."""
    if (
        getattr(message, "type", None) != "tool"
        or getattr(message, "name", "") != "validate_sparql_tool"
    ):
        return None
    content = message.content if isinstance(message.content, str) else str(message.content)
    return content


def _after_tool(state: State):
    """Route after tool execution with enforced SPARQL validation flow."""
    last_message = state["messages"][-1]
    if (
        getattr(last_message, "type", None) == "tool"
        and getattr(last_message, "name", "") == "validate_sparql_tool"
    ):
        return END
    if (
        getattr(last_message, "type", None) == "tool"
        and getattr(last_message, "name", "") == "run_sparql_subagent_tool"
    ):
        content = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
        # run_sparql_subagent_tool can return a precondition error instead of subagent output;
        # only force validation when there is an actual subagent result to validate.
        if content.strip() == _NO_KG_SELECTED_MESSAGE:
            return "sparql_chat_agent"
        return "force_validate"
    return "sparql_chat_agent"


def _is_sparql_endpoint_up(endpoint_url: str, timeout: int = 15) -> bool:
    """Check if a SPARQL endpoint is reachable by sending a lightweight test query."""
    try:
        response = requests.get(
            endpoint_url,
            params={"query": "ASK { ?s ?p ?o }"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout
        )
        return 200 <= response.status_code < 300
    except requests.exceptions.RequestException:
        return False