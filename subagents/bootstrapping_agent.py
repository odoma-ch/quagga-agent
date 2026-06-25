import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent_utils.agent_constants import GRAPHS_INVOLVED

load_dotenv()

logger = logging.getLogger(__name__)

GRAPH_DESCRIPTIONS_DIR = Path(__file__).parent / "graph_descriptions"

GRAPH_ENDPOINTS = {
    "Culture Knowledge Graph": "https://nfdi4culture.de/sparql",
    "GESIS KG": "https://data.gesis.org/gesiskg/sparql",
    "EHRI - KG": "https://lod.ehri-project-test.eu/sparql",
    "PHAROS": "https://artresearch.net/sparql",
    "OpenCitations Meta": "https://sparql.opencitations.net/meta",
    "LinkedMusic": "https://virtuoso.simssa.ca/sparql",
}

BOOTSTRAP_SYSTEM_PROMPT = """You are a domain expert in $expertise_domain and an advanced SPARQL query designer.

Your task is to generate diverse, high-quality natural-language federated research questions that a real $expertise_domain would ask when working with the **$kg_name** knowledge graph.
These questions should be answerable via SPARQL queries against the endpoint: $endpoint

## Ontology Description (cached from prior exploration)
$ontology_description

## Mandatory Generation Process (follow in order)

1. **Expert-First Brainstorming**: Start from genuine disciplinary intent, not from schema terms. Think about the kinds of investigations, comparisons, and evidence-seeking tasks a real $expertise_domain performs.

2. **Ontology as Feasibility Filter**: Use the ontology only to validate and refine candidate federated questions. Keep the research intent, but rewrite wording so each final question maps to available classes/properties/relations. Do not invent schema elements.

3. **Natural User Phrasing**: Final questions must read like what a researcher would ask in plain language, not like SPARQL.

## Domain Realism Guidance

- Favor questions that reflect authentic professional workflows (e.g., attribution, chronology, influence, provenance, collaboration networks, publication/performance history, reception, institutional context).
- Avoid ontology-only artifacts such as class-name echoing, predicate listing, or schema-centric wording.
- Have focus on making questions whihc are federated.

## Generation Guidelines

1. **Technical Complexity Mix**: Generate questions across different SPARQL construction difficulty levels:
   - **Simple**: Direct fact lookups, basic counts, existence checks
   - **Medium**: Multi-hop joins, filtered aggregations, temporal/geo constraints
   - **Hard**: Nested/complex aggregations, ranking/top-k, cross-entity reasoning, subqueries
   - Difficulty reflects query construction complexity, not scholarly importance.

2. **Answerability**: Every question must be practically answerable from the data. Avoid questions that require full-table scans or unbounded keyword searches.

3. **Output Format**: Return a JSON array of objects, each with:
   - "question": the natural-language question (string)
   - "difficulty": "simple" | "medium" | "hard"
   - "rationale": one-sentence explanation of why this question is interesting for a $expertise_domain and answerable from this graph

Generate exactly $num_questions questions.
"""

USER_PROMPT = """Generate $num_questions natural-language questions about the $kg_name knowledge graph, from the perspective of a practicing $expertise_domain.

The graph endpoint is: $endpoint

Use this workflow:
1) Draft questions that a real $expertise_domain would ask in their day-to-day research practice.
2) Keep only questions that can be grounded in the ontology provided in the system message.
3) If needed, minimally rephrase questions for schema compatibility without losing their real research intent.

Return ONLY the JSON array — no markdown fences, no preamble, no explanation outside the JSON.
"""


def load_cached_ontology(kg_name: str) -> str:
    desc_path = GRAPH_DESCRIPTIONS_DIR / kg_name
    if not desc_path.exists():
        raise FileNotFoundError(
            f"No cached ontology description found for '{kg_name}' at {desc_path}. "
            f"Run the SPARQL subagent exploration first to generate it."
        )
    return desc_path.read_text(encoding="utf-8")


def build_llm() -> ChatOpenAI:
    base_url = "https://openrouter.ai/api/v1"
    model = "qwen/qwen3.7-max"
    api_key = "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"

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=1.2,
        top_p=0.95,
    )


def generate_questions(
    kg_name: str,
    expertise_domain: str,
    num_questions: int = 20,
    endpoint: str | None = None,
) -> list[dict]:
    if kg_name not in GRAPHS_INVOLVED:
        logger.warning(
            f"'{kg_name}' not in GRAPHS_INVOLVED {GRAPHS_INVOLVED}. Proceeding anyway."
        )

    endpoint = endpoint or GRAPH_ENDPOINTS.get(kg_name)
    if not endpoint:
        raise ValueError(
            f"No endpoint known for '{kg_name}'. Provide --endpoint explicitly."
        )

    ontology_description = load_cached_ontology(kg_name)
    logger.info(f"Loaded ontology description for '{kg_name}' ({len(ontology_description)} chars)")

    llm = build_llm()

    system_prompt = (
        BOOTSTRAP_SYSTEM_PROMPT
        .replace("$kg_name", kg_name)
        .replace("$endpoint", endpoint)
        .replace("$expertise_domain", expertise_domain)
        .replace("$ontology_description", ontology_description)
        .replace("$num_questions", str(num_questions))
    )

    user_prompt = (
        USER_PROMPT
        .replace("$kg_name", kg_name)
        .replace("$endpoint", endpoint)
        .replace("$expertise_domain", expertise_domain)
        .replace("$num_questions", str(num_questions))
    )

    logger.info(f"Generating {num_questions} questions for '{kg_name}' as {expertise_domain} expert...")

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM output as JSON. Raw output:\n%s", raw)
        raise

    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON array, got {type(questions).__name__}")

    logger.info(f"Generated {len(questions)} questions")
    return questions


def main():
    parser = argparse.ArgumentParser(description="Bootstrap question generation for a knowledge graph")
    parser.add_argument(
        "--kg",
        type=str,
        required=True,
        choices=GRAPHS_INVOLVED,
        help="Knowledge graph name (must match a cached ontology description)",
    )
    parser.add_argument(
        "--expertise",
        type=str,
        required=True,
        help='Domain expertise persona, e.g. "Digital Humanities Researcher", "Bibliometrician", "Art Historian"',
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions to generate (default: 20)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Override the SPARQL endpoint URL (inferred from kg name if omitted)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON output file (prints to stdout if omitted)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    questions = generate_questions(
        kg_name=args.kg,
        expertise_domain=args.expertise,
        num_questions=args.num_questions,
        endpoint=args.endpoint,
    )

    output_json = json.dumps(questions, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Wrote {len(questions)} questions to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
