# credit-card-optimizer

Minimal OpenEnv-compatible credit card optimization environment for Hugging Face deployment and OpenAI-client-based inference.

## Run locally
```powershell
uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
uv run uvicorn credit_card_env.server.app:app --host 0.0.0.0 --port 7860
```

In a second terminal:
```powershell
$env:HF_TOKEN="your_hf_token"
uv run python inference.py
```
