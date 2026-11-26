import json
from collections.abc import Sequence
from typing import Any

from app.core.json_utils import _safe_parse_json

TOTAL_CONTEXT_BUDGET = 8000
GENERATION_RESERVE = 1500
SYSTEM_AND_SCHEMA_RESERVE = 1500
DEFAULT_HISTORY_BUDGET = 5000


def estimate_tokens(text: str) -> int:
    """Estimate token count using a deterministic ~4 chars/token heuristic."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate token count for a single OpenAI/Gemini chat message dict."""
    serialized = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    return estimate_tokens(serialized) + 4


def estimate_history_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Estimate total token count across a conversation message list."""
    return sum(estimate_message_tokens(msg) for msg in messages)


def compact_tool_payload(raw_content: str, max_summary_chars: int = 200) -> str:
    """
    Compact a bulky tool response payload into a concise summary JSON string
    preserving essential metadata (url, status) while bounding token usage.
    """
    if isinstance(raw_content, str):
        try:
            parsed: Any = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed = _safe_parse_json(raw_content)
    else:
        parsed = raw_content

    if isinstance(parsed, dict):
        if parsed.get("compacted") is True:
            return raw_content
        compacted: dict[str, Any] = {"compacted": True}
        if "url" in parsed:
            compacted["url"] = parsed["url"]
        if "status" in parsed:
            compacted["status"] = parsed["status"]
        if "error" in parsed:
            compacted["error"] = str(parsed["error"])[:max_summary_chars]
        else:
            raw_text = str(
                parsed.get("content")
                or parsed.get("summary")
                or parsed.get("evaluation_note")
                or json.dumps(parsed, ensure_ascii=False)
            )
            compacted["summary"] = (
                raw_text[:max_summary_chars] + "..."
                if len(raw_text) > max_summary_chars
                else raw_text
            )
        return json.dumps(compacted, ensure_ascii=False)

    if isinstance(parsed, list):
        preview = json.dumps(parsed[:1], ensure_ascii=False)[:max_summary_chars]
        return json.dumps(
            {
                "compacted": True,
                "results_count": len(parsed),
                "summary": preview + ("..." if len(preview) >= max_summary_chars else ""),
            },
            ensure_ascii=False,
        )

    text = str(raw_content or "")
    if len(text) <= max_summary_chars:
        return text
    return json.dumps(
        {"compacted": True, "summary": text[:max_summary_chars] + "..."},
        ensure_ascii=False,
    )


def compact_messages(
    messages: Sequence[dict[str, Any]],
    max_history_tokens: int = DEFAULT_HISTORY_BUDGET,
    preserve_recent_turns: int = 2,
) -> list[dict[str, Any]]:
    """
    Apply token-budgeted compaction to conversation history.

    - If history fits within max_history_tokens, returns messages unchanged.
    - Otherwise preserves the initial system & user goal messages and the latest
      `preserve_recent_turns` assistant+tool turns verbatim, while compacting older
      tool output payloads into <= 200-character summaries.
    """
    result = [dict(m) for m in messages]
    if estimate_history_tokens(result) <= max_history_tokens:
        return result

    # Identify assistant turn start indices to preserve the latest N turns verbatim
    assistant_indices = [i for i, m in enumerate(result) if m.get("role") == "assistant"]
    if preserve_recent_turns > 0 and len(assistant_indices) > preserve_recent_turns:
        protected_start = assistant_indices[-preserve_recent_turns]
    elif preserve_recent_turns > 0 and assistant_indices:
        protected_start = assistant_indices[0]
    else:
        protected_start = len(result)

    # Preserve initial system (0) and user goal (1) messages; compact older tool messages
    for idx in range(2, protected_start):
        msg = result[idx]
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            compacted_msg = dict(msg)
            compacted_msg["content"] = compact_tool_payload(msg["content"], max_summary_chars=200)
            result[idx] = compacted_msg

    return result
