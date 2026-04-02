## Description

This repository contains the code for `quagga agent`, an LLM agent for conversational question answering over knowledge graphs. The agent uses the [EXYGEN API](https://github.com/odoma-ch/exygen) to generate KG metadata (e.g. ShEX schemas, graph samples, etc.) to enable and guide the generation of SPARQL queries based on user questions. The agent integrates an OpenWebUI frontend and LiteLLM/OKD for LLM-powered query generation.

For some technical documentation about the agent's architecture, see [docs/AGENT_ARCHITECTURE.md](./docs/AGENT_ARCHITECTURE.md)

## How to run `quagga agent`

### Pre-requisite

```
# Remember you should be in the agent folder to run the SPARQL generating agent
pip install -r requirements.txt

# For local development, copy .env.example to .env and set LOCAL_MODE="True"
cp .env.example .env
# Note: Contact the administrators for the actual values to put in .env
```

### For running the GRAPHIA agent API:

- Start the OpenWebUI: `open-webui serve --port 8000`
- Or if the command line is not available, run via docker:
  ```
  docker run -d -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:main
  ```
- Start the API in another terminal tab: `uvicorn api:app --host 0.0.0.0 --port 8004`
- In OpenWebUI → Admin → Connections → OpenAI API:
    - URL: http://<host>:8004/v1
    - Key: any non-empty string (e.g. sk-dummy)
    - The agent will appear as a model named `Graphia SPARQL agent` in the OpenWebUI model selector.
    - Sometimes, OpenWebUI can cause issues, do the `lsof -i:8000` command and kill the process with the right PID.
- Might need to delete the `chroma_db` from the repo and rebuild the vector for some reason it causes hangup locally.

### For setting up the filter on OpenWebUI for `chat_id`:

- We need the `chat_id` to keep track of the conversations and its corresponding history.
- Go to: `localhost:8000/admin/functions` (or where the UI is hosted but with the corresponding path suffix) and add a function
- and then copy/paste the code in: `filters/__init__.py`
- After saving it, click on the toggle to `Enable` it and then press on the three dot menu to enable the `Global` toggle.

## License

The code contained in this repository is proprietary and confidential.

See [LICENSE](./LICENSE) for details.

## Credits

The code contained in this repository is being developed by [Harshdeep Singh](https://github.com/Harshdeep1996) ([Odoma](https://github.com/odoma-ch)). This work is being carried out in the context of the EU-funded [GRAPHIA project](https://graphia-ssh.eu/) (grant ID: [101188018](https://cordis.europa.eu/project/id/101188018)), as part of Task 4.1 (LLM4SSH).
