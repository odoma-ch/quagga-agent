## Runbook: Setup in Agent API in UI

- In OpenWebUI → Admin → Connections → OpenAI API:
    - URL: `http://<host>:8004/v1`
    - Key: any non-empty string (e.g. sk-dummy)
    - The agent will appear as a model named `Graphia SPARQL agent` in the OpenWebUI model selector.

- For setting up the filter on OpenWebUI for `chat_id`:
    - We need the `chat_id` to keep track of the conversations and its corresponding history.
    - Go to: `localhost:8000/admin/functions` (or where the UI is hosted but with the corresponding path suffix) and add a function
    - and then copy/paste the code in: `filters/__init__.py`
    - After saving it, click on the toggle to `Enable` it and then press on the three dot menu to enable the `Global` toggle.
