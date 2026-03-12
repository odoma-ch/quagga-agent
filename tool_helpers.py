from dotenv import load_dotenv
load_dotenv()

import requests

from models import State
from langgraph.graph import END

_TOOL_DISPLAY_NAMES = {
    "shex_retriever": "Searching ShEx schemas",
    "validate_sparql_tool": "Generating and validating SPARQL query",
}
_TERMINAL_SPARQL_MARKER = "The SPARQL has been executed:"


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
    preview = content[:500] + ("…" if len(content) > 500 else "")

    if tool_name == "validate_sparql_tool":
        return (
            f"<details>\n<summary>Result from {display}</summary>\n\n"
            f"{content}\n"
            f"</details>\n\n"
        )
    else:   
        return (
            f"<details>\n<summary>Result from {display}</summary>\n\n"
            f"```text\n{preview}\n```\n"
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
    if _TERMINAL_SPARQL_MARKER in content:
        return content
    return None


def _after_tool(state: State):
    """Route to END once query validation has succeeded."""
    last_message = state["messages"][-1]
    if _extract_terminal_tool_answer(last_message):
        return END
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