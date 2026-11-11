import os
import logging
import json
import asyncio
import contextvars
from typing import Any, Type
from pydantic import BaseModel, Field, create_model
import httpx

from langchain_core.tools import BaseTool

logger = logging.getLogger("quagga-mcp-client")

_current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_thread_id", default=""
)

_mcp_server_url: str = ""
_mcp_tools: list[BaseTool] = []


def _parse_sse_response(text: str) -> str:
    for line in text.split("\n"):
        if line.startswith("data:"):
            return line[5:].strip()
    return text


def set_current_thread_id(thread_id: str):
    from agent_utils.agent_runtime_state import runtime_state

    _current_thread_id.set(thread_id)
    runtime_state.current_thread_id = thread_id


def get_current_thread_id() -> str:
    return _current_thread_id.get()


def _get_session_state():
    from agent_utils.agent_runtime_state import runtime_state

    thread_id = _current_thread_id.get() or runtime_state.current_thread_id
    return runtime_state.get_or_create_session(thread_id)


def _json_schema_to_pydantic(schema: dict, model_name: str) -> Type[BaseModel]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    field_definitions = {}
    for field_name, field_schema in properties.items():
        field_type = Any
        is_required = field_name in required

        json_type = field_schema.get("type", "string")
        if json_type == "string":
            field_type = str
        elif json_type == "integer":
            field_type = int
        elif json_type == "number":
            field_type = float
        elif json_type == "boolean":
            field_type = bool
        elif json_type == "array":
            field_type = list
        elif json_type == "object":
            field_type = dict

        if is_required:
            field_definitions[field_name] = (field_type, ...)
        else:
            default = field_schema.get("default", None)
            field_definitions[field_name] = (field_type, default)

    return create_model(model_name, **field_definitions)


_MCP_MAX_PAGE_SIZE = 25
_MCP_TIMEOUT_SECONDS = 30.0
_MCP_MAX_RESPONSE_CHARS = 15000


def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    session_state = _get_session_state()

    if not session_state.mcp_session_id:
        return f"MCP session not initialized for this thread. Please restart the conversation."

    if (
        isinstance(arguments.get("size"), int)
        and arguments["size"] > _MCP_MAX_PAGE_SIZE
    ):
        arguments["size"] = _MCP_MAX_PAGE_SIZE

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "mcp-session-id": session_state.mcp_session_id,
    }

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    with httpx.Client(timeout=_MCP_TIMEOUT_SECONDS) as client:
        response = client.post(_mcp_server_url, json=request, headers=headers)
        response.raise_for_status()

        content = response.text
        message_data = _parse_sse_response(content)

        if not message_data:
            return "No response from MCP server"

        try:
            result = json.loads(message_data)
            if "error" in result:
                return f"MCP error: {result['error']}"
            if "result" in result and isinstance(result["result"], dict):
                content_result = result["result"].get("content", [])
                if isinstance(content_result, list):
                    text = "\n".join(str(item) for item in content_result)
                else:
                    text = str(content_result)
                if len(text) > _MCP_MAX_RESPONSE_CHARS:
                    text = (
                        text[:_MCP_MAX_RESPONSE_CHARS]
                        + "\n\n[Response truncated to prevent overload. Use smaller page sizes or more specific queries.]"
                    )
                return text
            return str(result)
        except json.JSONDecodeError:
            return message_data


class MCPToolWrapper(BaseTool):
    mcp_tool_name: str = ""
    mcp_tool_description: str = ""

    def _run(self, **kwargs) -> str:
        global _mcp_server_url
        if not _mcp_server_url:
            return "MCP server not configured. Please set MCP_SERVER_URL."

        session_state = _get_session_state()
        if not session_state.mcp_initialized:
            return "MCP session not initialized for this thread. Please restart the conversation."

        try:
            result = _call_mcp_tool(self.mcp_tool_name, kwargs)
            return result
        except Exception as e:
            logger.exception(f"Error calling MCP tool {self.mcp_tool_name}")
            return f"Error calling MCP tool: {str(e)}"


async def _discover_mcp_tools(server_url: str) -> list[BaseTool]:
    """Discover available tools from MCP server (called once at startup)"""
    global _mcp_tools

    if _mcp_tools:
        logger.info("MCP tools already discovered")
        return _mcp_tools

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "quagga-agent", "version": "1.0.0"},
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        prefilt_response = await client.post(server_url, headers=headers)
        temp_session_id = prefilt_response.headers.get("mcp-session-id", "")

        if not temp_session_id:
            raise ValueError("Failed to obtain MCP session ID during discovery")

        headers["mcp-session-id"] = temp_session_id

        response = await client.post(server_url, json=init_request, headers=headers)
        response.raise_for_status()

        list_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = await client.post(server_url, json=list_request, headers=headers)
        response.raise_for_status()

        content = response.text
        message_data = _parse_sse_response(content)

        try:
            result = json.loads(message_data)
            if "result" in result and "tools" in result["result"]:
                for tool_def in result["result"]["tools"]:
                    tool_name = tool_def.get("name", "unknown")
                    input_schema = tool_def.get("inputSchema", {})
                    schema_title = input_schema.get("title", f"{tool_name}Input")

                    args_schema = _json_schema_to_pydantic(input_schema, schema_title)

                    tool = MCPToolWrapper(
                        name=tool_name,
                        description=tool_def.get(
                            "description", f"MCP tool: {tool_name}"
                        ),
                        args_schema=args_schema,
                        mcp_tool_name=tool_name,
                        mcp_tool_description=tool_def.get("description", ""),
                    )
                    _mcp_tools.append(tool)
                    logger.info(f"Discovered MCP tool: {tool_name}")
        except Exception as e:
            logger.exception(f"Error parsing tools list: {e}")

    return _mcp_tools


async def _initialize_mcp_session_for_thread(server_url: str) -> bool:
    """Initialize MCP session for the current thread (called per-thread)"""
    from agent_utils.agent_runtime_state import runtime_state

    thread_id = runtime_state.current_thread_id
    session_state = runtime_state.get_or_create_session(thread_id)

    if session_state.mcp_initialized:
        return True

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "quagga-agent", "version": "1.0.0"},
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        prefilt_response = await client.post(server_url, headers=headers)
        session_state.mcp_session_id = prefilt_response.headers.get(
            "mcp-session-id", ""
        )

        if not session_state.mcp_session_id:
            logger.error(f"Failed to obtain MCP session ID for thread {thread_id}")
            return False

        logger.info(
            f"MCP session ID obtained for thread {thread_id}: {session_state.mcp_session_id[:20]}..."
        )
        headers["mcp-session-id"] = session_state.mcp_session_id

        response = await client.post(server_url, json=init_request, headers=headers)
        response.raise_for_status()

        session_state.mcp_initialized = True
        logger.info(f"MCP session initialized for thread {thread_id}")

    return True


async def initialize_mcp_tools(server_url: str) -> list[BaseTool]:
    """Discover MCP tools at startup (one-time)"""
    global _mcp_server_url

    if not server_url:
        logger.warning("MCP_SERVER_URL not set, skipping MCP tool initialization")
        return []

    _mcp_server_url = server_url

    try:
        logger.info(f"Discovering MCP tools from {server_url}")
        tools = await _discover_mcp_tools(server_url)
        logger.info(f"Discovered {len(tools)} MCP tools")
    except Exception as e:
        logger.exception(f"Failed to initialize MCP tools: {e}")

    return _mcp_tools


async def initialize_mcp_session() -> bool:
    """Initialize MCP session for current thread (per-thread)"""
    global _mcp_server_url

    if not _mcp_server_url:
        return False

    try:
        return await _initialize_mcp_session_for_thread(_mcp_server_url)
    except Exception as e:
        logger.exception(f"Failed to initialize MCP session: {e}")
        return False


def get_mcp_server_url() -> str:
    return os.getenv("MCP_SERVER_URL", "")


def get_mcp_tools() -> list[BaseTool]:
    """Return discovered MCP tools (for adding to agent tool list)"""
    return _mcp_tools


async def shutdown_mcp_client():
    global _mcp_server_url, _mcp_tools
    _mcp_server_url = ""
    _mcp_tools = []
    logger.info("MCP client shutdown")
