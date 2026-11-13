# ProspectIQ Backend

FastAPI service powering the autonomous AI Sales Intelligence & Outreach Agent.

---

## Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **LLM Engine:** [Groq](https://groq.com/) (using `openai/gpt-oss-120b` for deep sales analysis)
- **Validation:** [Pydantic v2](https://docs.pydantic.dev/) & [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- **Logging:** [Loguru](https://github.com/Delgan/loguru) structured logging with request correlation IDs and latency tracking
- **Package Manager:** [`uv`](https://docs.astral.sh/uv/)

---

## Getting Started

### 1. Virtual Environment & Dependencies

```bash
# Inside the backend/ directory
uv venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e ".[dev]"
```

### 2. Environment Configuration

Copy the template and add your free Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
DEFAULT_MODEL=openai/gpt-oss-120b
FAST_MODEL=openai/gpt-oss-20b
```

### 3. Start the Server

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`

Alternatively, press `F5` in VS Code to start via the pre-configured **FastAPI: ProspectIQ** launch configuration.

### 4. Run Tests

```bash
uv run pytest -v
```

---

## API Endpoints

### `GET /health`

Returns service status.

**Response:**

```json
{
  "status": "ok",
  "app": "ProspectIQ"
}
```

---


### `POST /api/chat`

Sends a prompt to the Groq LLM with sales intelligence system prompt and returns structured completion metrics.

**Request Body (`application/json`):**

```json
{
  "prompt": "Analyze Stripe's core business model and primary customer segments.",
  "temperature": 0.7,
  "max_tokens": 1024
}
```

_All fields except `prompt` are optional (`temperature`, `max_tokens`, `top_p`, `model`, `system_prompt`, `request_id`, `timeout_seconds`, `max_retries`, `stream`). Defaults to `stream: false` returning JSON; when `stream: true`, returns Server-Sent Events (`text/event-stream`)._

**Response (`200 OK`):**

```json
{
  "request_id": "req_608cc418",
  "content": "**Stripe — Global Payments Infrastructure**\n\n...",
  "finish_reason": "stop",
  "is_truncated": false,
  "prompt_tokens": 138,
  "completion_tokens": 245,
  "total_tokens": 383,
  "latency_ms": 712.45
}
```
