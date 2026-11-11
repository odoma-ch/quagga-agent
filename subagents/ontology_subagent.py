import os
import re
import sys
import httpx
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

try:
    from markdownify import markdownify as md

    _HAS_MARKDOWNIFY = True
except ImportError:
    _HAS_MARKDOWNIFY = False

load_dotenv()


MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5 MB
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 120.0

_ACCEPT_HEADERS = {
    "markdown": "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1",
    "text": "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
    "html": "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.1",
}

_BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _html_to_markdown(html: str) -> str:
    if _HAS_MARKDOWNIFY:
        return md(
            html,
            heading_style="ATX",
            bullets="-",
            code_language="",
            strip=["script", "style", "meta", "link"],
        )
    return re.sub(r"<[^>]+>", " ", html).strip()


def _html_to_text(html: str) -> str:
    text = re.sub(
        r"<(script|style|noscript|iframe|object|embed)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


async def webfetch(
    url: str,
    fmt: str = "markdown",
    timeout: Optional[float] = None,
) -> dict:
    """
    Fetch a URL and return a structured dict.

    Returns
    -------
    {
        "title":        str,
        "output":       str,   # content in requested format
        "content_type": str,
        "status_code":  int,
    }
    """
    print(f"Fetching URL: {url}")
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    effective_timeout = min(timeout or DEFAULT_TIMEOUT, MAX_TIMEOUT)
    accept = _ACCEPT_HEADERS.get(fmt, "*/*")
    headers = {**_BASE_HEADERS, "Accept": accept}

    async with httpx.AsyncClient(
        timeout=effective_timeout,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = await client.get(url)
        if (
            response.status_code == 403
            and response.headers.get("cf-mitigated") == "challenge"
        ):
            response = await client.get(
                url, headers={**headers, "User-Agent": "opencode"}
            )

    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}")

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_RESPONSE_SIZE:
        raise RuntimeError("Response too large (exceeds 5 MB)")

    raw = response.content
    if len(raw) > MAX_RESPONSE_SIZE:
        raise RuntimeError("Response too large (exceeds 5 MB)")

    content_type = response.headers.get("content-type", "")
    mime = content_type.split(";")[0].strip().lower()
    title = f"{url} ({content_type})"

    if mime.startswith("image/") and mime not in (
        "image/svg+xml",
        "image/vnd.fastbidsheet",
    ):
        import base64

        b64 = base64.b64encode(raw).decode()
        return {
            "title": title,
            "output": "Image fetched successfully",
            "content_type": mime,
            "status_code": response.status_code,
            "attachment": f"data:{mime};base64,{b64}",
        }

    text = raw.decode(response.encoding or "utf-8", errors="replace")
    is_html = "text/html" in content_type

    if fmt == "markdown" and is_html:
        output = _html_to_markdown(text)
    elif fmt == "text" and is_html:
        output = _html_to_text(text)
    else:
        output = text

    return {
        "title": title,
        "output": output,
        "content_type": mime,
        "status_code": response.status_code,
    }


@tool
async def webfetch_tool(
    url: str, format: str = "markdown", timeout: float = DEFAULT_TIMEOUT
) -> str:
    """
    Fetch the content of a URL and return a Markdown report of what is there.

    Args:
        url:     The URL to fetch (must start with http:// or https://).
        format:  "markdown" (default), "text", or "html".
        timeout: Request timeout in seconds (max 120).
    """
    result = await webfetch(url, fmt=format, timeout=timeout)
    return (
        f"## Web Fetch Report\n\n"
        f"**URL:** {url}\n"
        f"**Content-Type:** {result['content_type']}\n"
        f"**Status:** {result['status_code']}\n\n"
        f"---\n\n"
        f"{result['output']}"
    )


base_url = os.getenv("PSNC_OKD_BASE_URL")
model = os.getenv("PSNC_OKD_LARGE_MODEL")
api_key = os.getenv("PSNC_OKD_API_KEY")

llm = ChatOpenAI(
    base_url=base_url,
    model=model,
    api_key=api_key,
    temperature=0,
)


class WebFetchAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    input_text: str  # full user input (text + URLs + question)
    urls: List[str]  # all URLs extracted from input_text
    fetched: dict  # url -> {output, content_type, status_code} | {error}
    final_report: str


def extract_urls(state: WebFetchAgentState) -> WebFetchAgentState:
    """Extract every http(s):// URL from the input text."""
    urls = re.findall(r'https?://[^\s<>"\')\]]+', state["input_text"])
    # deduplicate while preserving order
    seen: set = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return {**state, "urls": unique}


async def fetch_all_urls(state: WebFetchAgentState) -> WebFetchAgentState:
    """Fetch all extracted URLs concurrently."""

    async def _fetch_one(url: str) -> tuple[str, dict]:
        try:
            result = await webfetch(url, fmt="markdown")
            # truncate large pages to keep the LLM context manageable (20000 characters)
            result["output"] = result["output"][:20000]
            return url, result
        except Exception as exc:
            return url, {
                "error": str(exc),
                "output": "",
                "content_type": "",
                "status_code": 0,
            }

    results = await asyncio.gather(*[_fetch_one(u) for u in state["urls"]])
    return {**state, "fetched": dict(results)}


def answer_question(state: WebFetchAgentState) -> WebFetchAgentState:
    """Build a context block from fetched pages and ask the LLM to answer."""
    fetched = state["fetched"]

    if not fetched:
        report = "No URLs were found in the input text."
        return {
            **state,
            "final_report": report,
            "messages": state["messages"] + [AIMessage(content=report)],
        }

    context_parts = []
    for url, data in fetched.items():
        if data.get("error"):
            context_parts.append(f"### {url}\n\n**Error:** {data['error']}\n")
        else:
            header = (
                f"### {url}\n"
                f"**Content-Type:** {data.get('content_type', '')}\n"
                f"**Status:** {data.get('status_code', '')}\n\n"
            )
            context_parts.append(header + data["output"])

    context = "\n\n---\n\n".join(context_parts)

    system = SystemMessage(content=("""
You are a helpful assistant. 
The user has provided some text containing URLs. 
The content of those URLs has been fetched and is provided below as context. 
Answer the user's question based on that content. 
Be thorough but concise. Use Markdown formatting.
If the user's question is not related to the content of the URLs, say so clearly and explain what was tried and why it couldn't be resolved.
        """))
    user = HumanMessage(
        content=(
            f"## Fetched URL Content\n\n{context}\n\n"
            f"---\n\n"
            f"## User Input\n\n{state['input_text']}"
        )
    )

    response = llm.invoke([system, user])
    report = response.content

    return {
        **state,
        "final_report": report,
        "messages": state["messages"] + [AIMessage(content=report)],
    }


graph = StateGraph(WebFetchAgentState)
graph.add_node("extract_urls", extract_urls)
graph.add_node("fetch_all_urls", fetch_all_urls)
graph.add_node("answer_question", answer_question)

graph.add_edge(START, "extract_urls")
graph.add_edge("extract_urls", "fetch_all_urls")
graph.add_edge("fetch_all_urls", "answer_question")
graph.add_edge("answer_question", END)

subagent = graph.compile()


def run_pipeline(shex_prefixes: str, user_question: str) -> str:
    """Run the ontology subagent with shex prefixes and a user question, returning the report."""
    prompt = f"""{shex_prefixes}

Scan the prefix URLs and give me a report of what might be potentially ontologies/classes which can be used to answer the user question: "{user_question}"

**Consider the following aspects in your analysis:**

1. **Ontology Relevance**: Identify ontologies that are semantically related to the question's domain
2. **Class Hierarchy Depth**: Consider both direct classes and their parent/child relationships
3. **Property Relationships**: Look for classes that have relevant properties or relationships
4. **Subclass Relationships**: Include parent classes that might contain instances through inheritance
5. **Cross-ontology Mappings**: Check if there are equivalent classes across different ontologies
6. **Instance Patterns**: Consider which classes would likely have actual instances to answer the question
7. **Domain-Specific Ontologies**: Prioritize domain-specific ontologies over general ones when relevant
8. **Temporal/Geographic Context**: Consider if location or time-related ontologies might be relevant

**Report Structure:**
- List of potentially relevant ontologies with brief rationale
- Key classes and their relationships 
- Suggested class hierarchies to explore
- Any cross-referenced ontologies that might provide complementary information
- Do not mention about other ontologies that should be used to answer the question, just the ones that are potentially relevant.
"""

    async def _run():
        return await subagent.ainvoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "input_text": prompt,
                "urls": [],
                "fetched": {},
                "final_report": "",
            }
        )

    result = asyncio.run(_run())
    return result["final_report"]


if __name__ == "__main__":
    #     example_prefixes = """
    # PREFIX eo: <http://lod.ehri-project-test.eu/ontology#>
    # PREFIX io: <https://www.ica.org/standards/RiC/ontology#>
    # PREFIX ww: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    # """

    #     example_question = "How many archival descriptions does the EHRI KG contain?"

    example_prefixes = """
PREFIX pc: <http://purl.org/vocab/frbr/core#>
PREFIX purlSF: <http://purl.org/spar/fabio/>
PREFIX purlSP: <http://purl.org/spar/pro/>
PREFIX prismstandardNB2: <http://prismstandard.org/namespaces/basic/2.0/>
PREFIX purlSD: <http://purl.org/spar/datacite/>
PREFIX essepuntato20L: <http://www.essepuntato.it/2010/06/literalreification/>
"""

    example_question = "What are the latest 10 articles published in the journal with ISSN 2641-3337 (Quantitative Science Studies)?"

    print(f"\n{'='*60}")
    print(f"QUESTION: {example_question}")
    print(f"{'='*60}")
    report = run_pipeline(example_prefixes, example_question)
    print(report)
