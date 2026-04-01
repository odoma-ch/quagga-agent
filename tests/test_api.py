import os
import sys
import uuid
import json
import pytest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from api import app
from tests.answers_const import (
    EVAL_QUESTIONS, TRACE_ASSERT_EXCEPTIONS, TEST_ANSWER_TOOLS
)
from dotenv import load_dotenv
load_dotenv()


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
    start_time = datetime.now()
    response = _chat(agent_client, question)
    end_time = datetime.now()
    # check if the response is a valid chat completion [data model check]
    assert response["object"] == "chat.completion"
    answer = response["choices"][0]["message"]["content"]
    # check type and if there is some kind of answer
    assert isinstance(answer, str)
    assert len(answer) > 0
    logger.info(f"Time taken for {question}: {end_time - start_time}")

    # semantic quality check (instead of just checking for phrases)
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ModuleNotFoundError:
        pytest.skip("deepeval is not installed. Install with: pip install deepeval")

    deepeval_api_key = os.getenv("DEEPEVAL_API_KEY")
    deepeval_base_url = os.getenv("DEEPEVAL_BASE_URL")
    deepeval_model = os.getenv("DEEPEVAL_MODEL")

    if not deepeval_api_key:
        pytest.skip("DEEPEVAL_API_KEY is required for DeepEval GEval")

    os.environ["OPENAI_API_KEY"] = deepeval_api_key
    if deepeval_base_url:
        os.environ["OPENAI_BASE_URL"] = deepeval_base_url

    metric = GEval(
        name="answer_relevance_and_correctness",
        criteria=(
"""
Evaluate whether the final answer is logical and relevant given the user's input and the
full API response context. The answer should directly address the user's question, and any
final query or reported result must include all columns and fields the user requested. The
fields present in the final results should match what was asked for and align with the query
logic — there should be no obvious contradictions or omitted fields. If the user did not
explicitly specify a knowledge graph, it is acceptable for the assistant to prompt the user
to select one rather than generating a fabricated query or result.
"""
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.7,
        model=deepeval_model,
    )
    response_context = json.dumps(response, ensure_ascii=True, default=str)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=[response_context],
    )
    metric.measure(test_case)
    assert metric.score is not None, f"DeepEval returned no score. reason={metric.reason}"
    assert metric.score >= 0.7, (
        f"DeepEval score too low: {metric.score:.2f}. reason={metric.reason}"
    )


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

    assert tool_calls, "Expected at least one tool call"
    assert "Retrieving knowledge graph metadata" not in str(tool_calls[-1]), "Last tool call should not be retrieving metadata"
    # check if tool calls of the expected list are present in the generated ones
    for expected_call in TEST_ANSWER_TOOLS[question]:
        assert any(expected_call in actual for actual in tool_calls), (
            f"Expected tool call not found: {expected_call}"
        )
