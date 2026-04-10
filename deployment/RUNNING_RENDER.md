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
