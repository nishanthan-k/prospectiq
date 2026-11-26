from collections.abc import Sequence
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.agent.models import QualificationVerdict, ToolCallRecord
from app.db.models import SessionMemory


class ExtractedMemory(BaseModel):
    """Structured working memory candidate extracted from an agent run."""

    memory_type: str = Field(..., description="'fact' | 'qualification' | 'citation' | 'profile'")
    key: str = Field(..., description="Normalized memory key (e.g. 'webpage.stripe.com')")
    value: Any = Field(..., description="Structured memory payload")
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_verified: bool = False


def _domain_matches(url: str, target_domain: Optional[str]) -> bool:
    if not url or not target_domain:
        return False
    host = (urlparse(url).hostname or "").lower()
    target = target_domain.lower().strip()
    return host == target or host.endswith(f".{target}")


def extract_memories_from_run(
    tool_calls: Sequence[ToolCallRecord],
    target_domain: Optional[str] = None,
    qualification: Optional[QualificationVerdict] = None,
) -> list[ExtractedMemory]:
    """
    Extract grounded working memory candidates with provenance from executed tool calls
    and optional qualification verdicts.
    """
    memories: list[ExtractedMemory] = []

    for tc in tool_calls:
        if tc.is_error or not isinstance(tc.output, dict):
            continue

        if tc.tool_name == "fetch_webpage" and tc.output.get("status") == "success":
            url = str(tc.output.get("url") or "").strip()
            title = str(tc.output.get("title") or "").strip()
            content = str(tc.output.get("content") or "").strip()
            if not url or not content:
                continue

            host = urlparse(url).hostname or "page"
            path = (urlparse(url).path or "/").strip("/") or "home"
            key = f"webpage.{host}.{path}".replace("/", ".")[:240]
            is_primary = _domain_matches(url, target_domain)
            quote = content[:240].strip()

            memories.append(
                ExtractedMemory(
                    memory_type="fact",
                    key=key,
                    value={"title": title, "summary": content[:400], "url": url},
                    source_url=url,
                    source_quote=quote,
                    confidence=0.90 if is_primary else 0.75,
                    is_verified=is_primary,
                )
            )

        elif tc.tool_name == "evaluate_lead_fit":
            memories.append(
                ExtractedMemory(
                    memory_type="qualification",
                    key="qualification.latest",
                    value={
                        "company_name": tc.output.get("company_name"),
                        "fit_score": tc.output.get("fit_score"),
                        "is_qualified": tc.output.get("is_qualified"),
                        "status": tc.output.get("status"),
                        "matched_criteria": tc.output.get("matched_criteria", []),
                        "unmatched_criteria": tc.output.get("unmatched_criteria", []),
                    },
                    source_url=None,
                    source_quote=tc.output.get("evaluation_note"),
                    confidence=0.85,
                    is_verified=True,
                )
            )

    if qualification is not None and not any(m.memory_type == "qualification" for m in memories):
        memories.append(
            ExtractedMemory(
                memory_type="qualification",
                key="qualification.latest",
                value=qualification.model_dump(),
                source_url=None,
                source_quote=None,
                confidence=0.80,
                is_verified=False,
            )
        )

    return memories


def format_session_memories_for_prompt(memories: Sequence[SessionMemory]) -> str:
    """
    Format grounded SessionMemory records with provenance, timestamps, confidence,
    and an explicit skepticism directive for multi-turn agent prompts.
    """
    if not memories:
        return ""

    lines: list[str] = ["[PRIOR SESSION CONTEXT — HISTORICAL OBSERVATIONS]"]
    for mem in memories:
        obs_date = mem.extracted_at.strftime("%Y-%m-%d") if mem.extracted_at else "unknown"
        src = mem.source_url or "internal evaluation"
        header = (
            f"- [{mem.memory_type}] {mem.key} "
            f"(Observed {obs_date} from {src}, confidence: {mem.confidence:.2f}"
        )
        if mem.is_verified:
            header += ", verified"
        header += "):"
        lines.append(header)

        if isinstance(mem.value, dict):
            summary = mem.value.get("summary") or mem.value.get("status") or str(mem.value)
            lines.append(f"  Summary: {summary}")
        else:
            lines.append(f"  Value: {mem.value}")

        if mem.source_quote:
            lines.append(f'  Quote: "{mem.source_quote}"')

    lines.append(
        "NOTE: Treat prior context as historical observations. "
        "If newer search results or fetched pages contradict this context, prioritize the fresher evidence."
    )
    return "\n".join(lines)
