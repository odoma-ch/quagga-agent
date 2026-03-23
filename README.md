## Description

This repository contains the SPARQL generating agent for the GRAPHIA project. It uses metadata such as ShEX schemas and samples to generate SPARQL queries based on user prompts, integrating with OpenWebUI and LiteLLM/OKD for LLM-powered query generation.

## Pre requisite:

```
# Remember you should be in the agent folder to run the SPARQL generating agent
pip install -r requirements.txt

# For local development, copy .env.example to .env and set LOCAL_MODE="True"
cp .env.example .env
# Note: Contact the administrators for the actual values to put in .env
```

## For running the GRAPHIA agent API:

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

## For setting up the filter on OpenWebUI for `chat_id`:

- We need the `chat_id` to keep track of the conversations and its corresponding history.
- Go to: `localhost:8000/admin/functions` (or where the UI is hosted but with the corresponding path suffix) and add a function
- and then copy/paste the code in: `filters/__init__.py`
- After saving it, click on the toggle to `Enable` it and then press on the three dot menu to enable the `Global` toggle.

---

## Archivable documentation

This can be useful for the developer if they want to do standalone LLM calls to LiteLLM or a deployed model on the OKD cluster.

### Running the agent without a sample file [crucial for demo]:

```
python run_agent.py --k 5 --endpoint-url https://sparql.opencitations.net/meta --path-to-shex ../test/artifacts/shex/shex_2025-08-11T17-01-11_oc_meta.ttl --collection-name opencitations_meta                            
```

### Running the agent with a sample file [work in progess currently]:

- The samples for the following command to work should exist in the `generate_sample/samples` folder.

```
python run_agent.py --k 5 --endpoint-url https://sparql.opencitations.net/meta --path-to-shex ../test/artifacts/shex/shex_2025-08-11T17-01-11_oc_meta.ttl --collection-name opencitations_meta --path-to-sample ../generate_sample/samples/opencitations_meta_2_20_TOPN=20_PP=0.5.ttl
```

### Triggering the LLM on LiteLLM

```
curl -X POST https://llm.graphia-ssh.eu/api/chat/completions \
-H "Authorization: Bearer sk-dummy" \
-H "Content-Type: application/json" \
-d '{"model": "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf","messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

```
curl -X POST https://llm.graphia-ssh.eu/api/chat/completions \
-H "Authorization: Bearer sk-dummy" \
-H "Content-Type: application/json" \
-d '{"model": "DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf","messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

### Triggering the LLM on OKD
```
curl https://quagga-llm-route-graphia-app1-staging.apps.bst2.paas.psnc.pl/v1/chat/completions -H "Content-Type: application/json"   -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
    "messages": [
      {
        "role": "system",
        "content": "You are a coding assistant, skilled in programming."
      },
      {
        "role": "user",
        "content": "Write a hello world program in C++."
      }
    ]
  }' 2>/dev/null | jq -C
```