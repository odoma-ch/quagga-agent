## Running locally UI and API Agent:

1. Start both services:

   ```bash
   PSNC_OKD_BASE_URL=<medium-model-base-url> \
   PSNC_OKD_API_KEY=<medium-model-api-key> \
   PSNC_HPC_BASE_URL=<large-model-base-url> \
   PSNC_HPC_API_KEY=<large-model-api-key> \
   PSNC_OKD_MEDIUM_MODEL=<medium-model-name> \
   PSNC_OKD_LARGE_MODEL=<large-model-name> \
   docker-compose up
   ```

2. Open the functions admin page:
   - `http://localhost:3000/admin/functions`

3. In another tab, open:
   - `https://raw.githubusercontent.com/odoma-ch/quagga-agent/refs/heads/main/filters/__init__.py`

4. Copy the script that injects the `chat_id`.

5. In the admin page, click **New Function** and create:
   - **Function name:** `ChatIDInjector`
   - **Description:** `Injects ChatID`
   - Save and confirm.

6. After saving, you will be redirected back to:
   - `http://localhost:3000/admin/functions`

7. For the `ChatIDInjector` filter:
   - Click the 3-dot menu and enable the **Global** toggle.
   - Enable the main toggle as well (it should turn green).
