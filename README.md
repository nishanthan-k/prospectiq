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

## Monorepo Packages

* [**backend/**](backend/README.md) – FastAPI service, Groq LLM integration, sales intelligence engine, and database persistence.
* **frontend/** – Web dashboard for lead submission, evidence inspection, and outreach review *(planned for later levels)*.

For backend setup, environment configuration, and API reference, see [backend/README.md](backend/README.md).

---

For agent development rules, coding standards, and architectural constraints, see [CLAUDE.md](CLAUDE.md).
