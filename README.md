# Quagga Agent

An LLM agent for conversational question answering over knowledge graphs.

Given a natural-language question, Quagga Agent decides which knowledge graph (KG) is relevant, retrieves metadata about it (ShEx schemas, graph samples), and generates and validates SPARQL queries to answer the question. The metadata is generated with the [EXYGEN API](https://github.com/odoma-ch/exygen). The agent is exposed as an OpenAI-compatible API, so it works out of the box with [OpenWebUI](https://github.com/open-webui/open-webui) or any other OpenAI-compatible client.

## Demo

Watch a short video showing how to access and use GRAPHIA's Quagga agent instance:

[![Quagga agent demo](https://img.youtube.com/vi/lcOEKtBUK5Q/maxresdefault.jpg)](https://www.youtube.com/watch?v=lcOEKtBUK5Q)

Quagga Agent received the Award for Maturity at the third [HERMES Data Challenge "From Notes to Nodes"](https://www.hermes-hub.de/forschen/datachallenges/challenges/challenge-2026.html) (2026), where Team MusiGraphia took part with it. Read the [report](https://odoma-ch.github.io/hermes-musigraphia-report/) submitted to the challenge.

## Features

- **Schema-guided SPARQL generation**: ShEx schema chunks are retrieved from a vector store (ChromaDB) to ground query generation.
- **Query validation**: generated SPARQL is executed against the endpoint and checked before the answer is returned.
- **Multi-KG scope checking**: the agent determines which KG a question belongs to. Descriptions for the supported graphs live in [`subagents/graph_descriptions/`](./subagents/graph_descriptions): Wikidata, GoTriple, EHRI, GESIS, DNB, OpenCitations Meta, PHAROS, LinkedMusic and the Culture Knowledge Graph.
- **Subagents** for KG exploration, ontology lookup and bootstrapping.
- **OpenAI-compatible API** (FastAPI), with conversation history checkpointed in Postgres.
- **Pluggable LLM backends** via LiteLLM, with fallback routing (see [`agent_utils/provider_router.py`](./agent_utils/provider_router.py)).

For the design in detail, see [docs/AGENT_ARCHITECTURE.md](./docs/AGENT_ARCHITECTURE.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `api.py` | FastAPI server exposing the OpenAI-compatible API |
| `run_agent.py`, `tools.py`, `models.py` | Agent graph (LangGraph), tools and data models |
| `agent_utils/` | Configuration, runtime state, provider routing, MCP client |
| `subagents/` | SPARQL exploration, ontology and bootstrapping subagents, KG descriptions |
| `prompts/` | System prompts and rule files |
| `filters/` | OpenWebUI filter that passes `chat_id` to the agent |
| `deployment/` | Docker Compose and Kubernetes/OKD manifests, deployment guides |
| `docs/` | Architecture notes, API documentation, evaluation dataset |
| `tests/` | API tests and Locust load test |

## Getting started

Just want to try it? The [demo video](https://www.youtube.com/watch?v=lcOEKtBUK5Q) shows how to use GRAPHIA's hosted Quagga agent instance without installing anything. To run your own instance, expand the section below.

<details>
<summary>Run your own instance (setup instructions)</summary>

### Prerequisites

- Python 3.12 (or Docker)
- Access to LLM endpoints for the main agent, sub-agent and tool models
- A Postgres instance for conversation checkpointing (optional for local development)

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and set the values. For local development set `LOCAL_MODE="True"`. The main variables are:

| Variable | Purpose |
| --- | --- |
| `PSNC_OKD_BASE_URL`, `PSNC_OKD_API_KEY`, `PSNC_OKD_LARGE_MODEL` | Main agent model |
| `PSNC_HPC_BASE_URL`, `PSNC_HPC_API_KEY`, `PSNC_OKD_MEDIUM_MODEL` | Sub-agent model |
| `PSNC_SMALL_TOOL_BASE_URL`, `PSNC_SMALL_TOOL_API_KEY`, `PSNC_SMALL_TOOL_MODEL` | Small model for tool calls |
| `QUAGGA_OPENROUTER_*` | Fallback model routing |
| `POSTGRES_*` | Postgres checkpointing |
| `MCP_SERVER_URL` | MCP server used by the agent |
| `OPENWEBUI_URL`, `OPENWEBUI_TOKEN` | OpenWebUI integration |
| `SPARQL_QUERY_TIMEOUT` | Timeout for SPARQL execution |

### Run with Docker Compose

Starts the API and OpenWebUI together. See [deployment/RUNNING_LOCALLY.md](./deployment/RUNNING_LOCALLY.md) for the full walkthrough.

```bash
docker-compose up
```

OpenWebUI is then available at `http://localhost:3000` and the API at `http://localhost:8004`.

### Run from source

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8004
```

Then start OpenWebUI, either with `open-webui serve --port 8000` or with Docker:

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

In OpenWebUI, go to **Admin → Connections → OpenAI API** and set:

- URL: `http://<host>:8004/v1`
- Key: any non-empty string (e.g. `sk-dummy`)

The agent appears in the model selector as **Graphia SPARQL agent**.

### Enable the `chat_id` filter

The agent needs the `chat_id` to keep track of conversation history.

1. Open `<openwebui-url>/admin/functions` and add a new function.
2. Paste the code from [`filters/__init__.py`](./filters/__init__.py).
3. Save, enable the function, and turn on the **Global** toggle in its three-dot menu.

### Troubleshooting

- If OpenWebUI hangs on startup, find the process with `lsof -i:8000` and kill it.
- If the agent hangs locally on startup, delete the `chroma_db` folder and let the vector store rebuild.

</details>

## Using the agent as an API

The API is OpenAI-compatible (`/v1/chat/completions`). See [agent_as_api.md](./agent_as_api.md) and [docs/API_DOCUMENTATION.md](./docs/API_DOCUMENTATION.md).

## Deployment

Kubernetes/OKD manifests for the API, Postgres and the web UI are in [`deployment/`](./deployment). See [deployment/RUNNING_OPENWEBUI.md](./deployment/RUNNING_OPENWEBUI.md).

## Testing

```bash
pip install -r tests/requirements-test.txt
```

The API tests are in [`tests/test_api.py`](./tests/test_api.py). For the Locust load test, see [tests/README.md](./tests/README.md).

## License

Released under the [Functional Source License, Version 1.1, MIT Future License](./LICENSE) (FSL-1.1-MIT).

In short: you may use, copy, modify and redistribute the code for any purpose other than a competing commercial use, including internal use, non-commercial education and non-commercial research. Each version converts to the MIT license two years after it is released. Read the [LICENSE](./LICENSE) for the binding terms.

## Credits

Developed by [Harshdeep Singh](https://github.com/Harshdeep1996) at [Odoma](https://github.com/odoma-ch), in the context of the EU-funded [GRAPHIA project](https://graphia-ssh.eu/) (grant ID: [101188018](https://cordis.europa.eu/project/id/101188018)), as part of Task 4.1 (LLM4SSH).
