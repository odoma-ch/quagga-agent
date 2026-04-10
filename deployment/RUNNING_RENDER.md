## Runbook: OpenWebUI on Render

These steps are **only for deploying OpenWebUI** using the Blueprint file at `deployment/render-openwebui.yaml`.

1) Install and authenticate Render CLI

```bash
brew install render
render login
```

2) Select the correct workspace

```bash
render workspace set
```

Pick `dev-graphia` when prompted.

3) Validate the Blueprint file

Run from the repository root:

```bash
render blueprints validate deployment/render-openwebui.yaml
```

This checks that the YAML is valid for Render Blueprints (it does not deploy anything).

4) Instantiate (create resources) in Render

1. Commit and push your changes.
2. In Render Dashboard, go to **New -> Blueprint**.
3. Connect this repository and branch.
4. Set **Blueprint Path** to `deployment/render-openwebui.yaml`.
5. Fill values for any `sync: false` env vars (for example `WEBUI_URL`).
6. Click **Deploy Blueprint**.

5) After first deploy

- Confirm service health at `/health`.
- Open the Render public URL and verify OpenWebUI loads.


## Runbook: Graphchat Agent on Render

These steps are **only for deploying OpenWebUI** using the Blueprint file at `deployment/render-agentapi.yaml`.

1) Validate the Blueprint file

Run from the repository root:

```bash
render blueprints validate deployment/render-agentapi.yaml
```

This checks that the YAML is valid for Render Blueprints (it does not deploy anything).

2) Instantiate (create resources) in Render

1. Commit and push your changes.
2. In Render Dashboard, go to **New -> Blueprint**.
3. Connect this repository and branch [usually main].
4. Set **Blueprint Path** to `deployment/render-agentapi.yaml`.
5. Fill in the environment variables by specifying the key-value either from the `.env` or `.env.example` file.
6. Click **Deploy Blueprint**.


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
