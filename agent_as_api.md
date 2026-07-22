# Agent as API

The Quagga SPARQL Agent runs as an OpenAI-compatible API on an OKD pod. Follow these steps to call it locally via port-forwarding.

## Prerequisites

### 1. Connect to the VPN

Access to the OKD cluster is **only valid when the VPN is active**. Connect to the VPN before proceeding with any of the steps below.

### 2. Login to OKD

```bash
oc login --web
```

This opens a browser window for authentication. Complete the login there.

### 3. Port-forward the API pod

```bash
oc port-forward deployment/quagga-agent-api 3000:8004
```

This forwards the API running on port 3000 inside the OKD pod to `localhost:3000`. Keep this terminal open while using the API.

### 4. Call the API

The API exposes an OpenAI-compatible chat completions endpoint at:

```
http://localhost:3000/v1/chat/completions
```

No authentication token is required.

#### Python example

```python
import json
import uuid
import requests

BACKEND_URL = "http://localhost:3000/v1/chat/completions"


def trigger_agent(question: str, chat_id: str | None = None):
    payload = {
        "messages": [{"role": "user", "content": question}],
        "chat_id": chat_id or str(uuid.uuid4()),
        "model": "graphchat",
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(BACKEND_URL, json=payload, headers=headers, stream=True)
    resp.raise_for_status()
    full_response = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            chunk = line[6:]
            try:
                data = json.loads(chunk)
                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    full_response += content
            except json.JSONDecodeError:
                pass
    return full_response
```

#### Usage

```python
answer = trigger_agent("What is this knowledge graph about?")
print(answer)
```
