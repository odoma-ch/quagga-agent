import os
import sys
import uuid
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from api import app


EVAL_QUESTIONS = [
    "What is Quagga?",
    "How many publications are in the OpenCitations dataset?",
    "List all authors who published more than 5 papers.",
    # "Are there other ways to categorize publications in GESIS, perhaps by field or research area?",
    # "What is the most cited article in OpenCitations?",
    # "Give me a list of organizations in GESIS together with the number of datasets they published.",
    # "I want to query OpenCitations. How many publications are indexed in there? break them down by publication typology",
    # "who is the most prolific Humanities author in terms of number of publications?"
]

# Questions that may be answered directly by the model
# without going through a tool call/result path.
TRACE_ASSERT_EXCEPTIONS = {
    "What is Quagga?",
}

# Interesting conversation extracts:
# https://quagga-agent-webui-graphia-app1-staging.apps.bst2.paas.psnc.pl/s/957b2ffc-6872-4008-a733-49953d116cb2
# https://quagga-agent-webui-graphia-app1-staging.apps.bst2.paas.psnc.pl/s/4df38006-07c6-4017-8b71-1338cd9edf8f

TEST_ANSWER_PHRASES = {
    "What is Quagga?": ["Quagga", "SPARQL", "query"],
    "How many publications are in the OpenCitations dataset?": [
        "Retrieving knowledge graph metadata", "Generating, validating and executing SPARQL query", "| count |"],
     "List all authors who published more than 5 papers.": [
        "Checking available knowledge graphs", "Selecting knowledge graph(s)", "Ask the user which knowledge graph would they want me to query"]
}

TEST_ANSWER_TOOLS = {
    "How many publications are in the OpenCitations dataset": [
        "**Selecting knowledge graph(s)**\n\n\n", 
        "**Retrieving knowledge graph metadata**\n\n\n", 
        "**Generating, validating and executing SPARQL query**\n\n\n"
    ],
    "List all authors who published more than 5 papers.":["**Checking available knowledge graphs**\n\n\n"]
}


@pytest.fixture(scope="module")
def agent_client():
    """
    Starts the FastAPI app with the real lifespan, hitting actual external services.
    Requires all environment variables (PSNC_OKD_API_KEY, PSNC_OKD_BASE_URL, etc.) to be set.
    """
    with TestClient(app) as client:
        yield client


def _chat(client, question: str, chat_id: str | None = None, stream: bool = False) -> dict:
    """Helper: POST a question to /v1/chat/completions, return parsed response."""
    body = {
        "messages": [{"role": "user", "content": question}],
        "chat_id": chat_id or str(uuid.uuid4()),
        "stream": stream,
    }
    resp = client.post("/v1/chat/completions", json=body)
    resp.raise_for_status()
    return resp.json()


@pytest.mark.parametrize("question", EVAL_QUESTIONS)
def test_full_flow(agent_client: TestClient, question: str):
    """
    Send each evaluation question through the full agent pipeline and
    assert that a non-empty answer comes back.

    Add assertions below based on what you expect for each question.
    """
    response = _chat(agent_client, question)
    # check if the response is a valid chat completion [data model check]
    assert response["object"] == "chat.completion"
    answer = response["choices"][0]["message"]["content"]
    # check type and if there is some kind of answer
    assert isinstance(answer, str)
    assert len(answer) > 0
    # check if the answer contains the expected phrases
    for phrase in TEST_ANSWER_PHRASES[question]:
        assert phrase in answer


@pytest.mark.parametrize("question", EVAL_QUESTIONS)
def test_execution_trace(agent_client: TestClient, question: str):
    """
    Stream the agent response and inspect the execution trace:
      - which graph nodes were visited
      - what tool calls were made
      - spot-checks on tool results

    Relies on the x_chunk_type field in each SSE chunk.
    """
    chat_id = str(uuid.uuid4())
    body = {
        "messages": [{"role": "user", "content": question}],
        "chat_id": chat_id,
        "stream": True,
    }

    resp = agent_client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse all SSE chunks
    chunks = []
    for raw_line in resp.text.splitlines():
        if raw_line.startswith("data:") and raw_line != "data: [DONE]":
            chunks.append(json.loads(raw_line[5:].strip()))

    assert chunks, "No SSE chunks received"

    # Categorise chunks by their x_chunk_type
    # e.g. "tool_call", "tool_result", "llm_text"
    nodes_visited: list[str] = []
    tool_calls: list[str] = []
    tool_results: list[str] = []
    llm_texts: list[str] = []

    for chunk in chunks:
        chunk_type = chunk.get("x_chunk_type", "unknown")
        content = chunk["choices"][0]["delta"].get("content", "")

        nodes_visited.append(chunk_type)

        if chunk_type == "tool_call":
            tool_calls.append(content)
        elif chunk_type == "tool_result":
            tool_results.append(content)
        elif chunk_type == "llm_text":
            llm_texts.append(content)

    # node/flow assertions of the result chunks
    if question not in TRACE_ASSERT_EXCEPTIONS:
        assert "tool_call" in nodes_visited, "Expected at least one tool call"
        assert "tool_result" in nodes_visited, "Expected at least one tool result"

    assert "llm_text" in nodes_visited, "Expected a final LLM response"
    assert "final" in nodes_visited, "Expected a final chunk"

    # check if tool calls of the expected list are present in the generated ones
    for tool_calls in TEST_ANSWER_TOOLS[question]:
        for call_string in tool_calls:
            assert call_string in tool_calls, "Expected tool call to be present"
