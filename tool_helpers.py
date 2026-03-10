from dotenv import load_dotenv
load_dotenv()

import requests


_TOOL_DISPLAY_NAMES = {
    "shex_retriever": "Searching ShEx schemas",
    "validate_sparql_tool": "Generating and validating SPARQL query",
}


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
            f"<text>\n{preview}\n</text>\n</details>\n\n"
        )
    else:   
        return (
            f"<details>\n<summary>Result from {display}</summary>\n\n"
            f"<code>\n{preview}\n</code>\n</details>\n\n"
        )


def _is_sparql_endpoint_up(endpoint_url: str, timeout: int = 15) -> bool:
    """Check if a SPARQL endpoint is reachable by sending a lightweight test query."""
    try:
        # ask is a super simple light weight query [looks for a triple in the graph]
        response = requests.get(
            endpoint_url,
            params={"query": "ASK { ?s ?p ?o }", "format": "json"},
            timeout=timeout
        )
        return 200 <= response.status_code < 300
    except requests.exceptions.RequestException:
        return False
