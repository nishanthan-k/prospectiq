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


def _find_protected_start(messages: Sequence[dict[str, Any]], preserve_recent_turns: int) -> int:
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if preserve_recent_turns > 0 and len(assistant_indices) > preserve_recent_turns:
        return assistant_indices[-preserve_recent_turns]
    if preserve_recent_turns > 0 and assistant_indices:
        return assistant_indices[0]
    return len(messages)


def compact_messages(
    messages: Sequence[dict[str, Any]],
    max_history_tokens: int = DEFAULT_HISTORY_BUDGET,
    preserve_recent_turns: int = 2,
) -> list[dict[str, Any]]:
    """
    Apply token-budgeted compaction to conversation history, enforcing
    `max_history_tokens` as a hard ceiling:

    1. Within budget -> return unchanged.
    2. Compact older intermediate tool outputs (<= 200 chars).
    3. Compact older intermediate assistant text content if still over budget.
    4. Drop oldest complete middle (assistant + matching tool) turns if still over budget.
    5. Compact remaining recent tool outputs and truncate oversized user context if needed.
    """
    result = [dict(m) for m in messages]
    if estimate_history_tokens(result) <= max_history_tokens:
        return result

    protected_start = _find_protected_start(result, preserve_recent_turns)

    # Pass 1: Compact older intermediate tool messages
    for idx in range(2, protected_start):
        msg = result[idx]
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            compacted_msg = dict(msg)
            compacted_msg["content"] = compact_tool_payload(msg["content"], max_summary_chars=200)
            result[idx] = compacted_msg

    if estimate_history_tokens(result) <= max_history_tokens:
        return result

    # Pass 2: Compact older assistant text content in range(2, protected_start)
    for idx in range(2, protected_start):
        msg = result[idx]
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            content = msg["content"]
            if len(content) > 200:
                compacted_msg = dict(msg)
                compacted_msg["content"] = content[:200] + "..."
                result[idx] = compacted_msg

    if estimate_history_tokens(result) <= max_history_tokens:
        return result

    # Pass 3: Drop oldest complete middle turns (assistant + subsequent tool/user messages)
    while estimate_history_tokens(result) > max_history_tokens:
        protected_start = _find_protected_start(result, preserve_recent_turns=1)
        if protected_start <= 2:
            break
        # Find the end of the oldest middle turn starting at index 2
        turn_end = 3
        while turn_end < protected_start and result[turn_end].get("role") != "assistant":
            turn_end += 1
        del result[2:turn_end]

    if estimate_history_tokens(result) <= max_history_tokens:
        return result

    # Pass 4: Compact any remaining bulky tool outputs in recent turns
    for idx in range(2, len(result)):
        msg = result[idx]
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            compacted_msg = dict(msg)
            compacted_msg["content"] = compact_tool_payload(msg["content"], max_summary_chars=120)
            result[idx] = compacted_msg
            if estimate_history_tokens(result) <= max_history_tokens:
                return result

    # Pass 5: If initial user message (e.g. huge prior_memory_block) still exceeds budget, truncate it
    if len(result) > 1 and isinstance(result[1].get("content"), str):
        overhead = estimate_history_tokens(result) - estimate_message_tokens(result[1])
        allowed_user_chars = max(100, (max_history_tokens - overhead - 16) * 4)
        user_text = result[1]["content"]
        if len(user_text) > allowed_user_chars:
            truncated_user = dict(result[1])
            truncated_user["content"] = (
                user_text[:allowed_user_chars] + "... [truncated to fit budget]"
            )
            result[1] = truncated_user

    return result
