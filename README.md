# ProspectIQ

**Autonomous AI Sales Intelligence & Outreach Platform**

ProspectIQ takes target companies or prospective leads, autonomously researches their offerings and market position, qualifies lead fit using concrete supporting evidence, drafts hyper-personalized outreach, pauses for human review and approval, and dispatches messages across selected channels.

Built systematically across 8 production-grade milestones:

- **Level 1: Direct LLM Engine** (Parameters, streaming, timeouts, retries, truncation detection, structured outputs, structured logs)
- **Level 2: Research & Retrieval** (Search, extraction, ranking, verifiable citations, evidence insufficiency handling)
- **Level 3: Tool-Using Agent** (Deterministic schemas, permissions, safety sandboxes, automated tests)
- **Level 4: Stateful Memory** (PostgreSQL task state persistence, restart resilience, multi-turn memory)
- **Level 5: Durable Workflows** (Human-in-the-loop approval pause/resume, failure recovery, idempotent sends)
- **Level 6: Tracing & Evals** (Langfuse tracing, golden evaluation dataset, regression benchmarks)
- **Level 7: Production Hardening** (FastAPI service, AuthN/Z, Docker, CI/CD, operational alerts)
- **Level 8: Capstone ProspectIQ** (End-to-end autonomous research and outreach engine)

---

## Quickstart

### 1. Environment Setup
```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
uv pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and paste your free Groq API key: GROQ_API_KEY=gsk_...
```

### 2. Run the Development Server
```bash
uv run uvicorn src.prospectiq.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 3. Run Automated Tests
```bash
uv run pytest -v
```

---

For agent development rules, coding standards, and architectural constraints, see [CLAUDE.md](file:///Users/nishanthan/Projects/adk/prospectiq/CLAUDE.md).
