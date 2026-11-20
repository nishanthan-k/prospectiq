import ast
import json
import re
from typing import Optional


def _clean_json_string(text: str) -> str:
    """Strip reasoning/thought tags and markdown fences from JSON string."""
    # 1. Remove thinking / reasoning tags (e.g. from OSS models)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 2. Extract from markdown code fence if present
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        # Strip code fences from start/end
        cleaned = re.sub(r"^\s*```(?:json)?", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)

    return cleaned.strip()


def safe_parse_json(text: str) -> Optional[dict]:
    """
    Resiliently parse JSON output from LLM:
    - Handles raw/unescaped control characters in strings via strict=False.
    - Strips markdown fences and reasoning/thought tags.
    - Cleans trailing commas.
    - Extracts JSON substring if surrounded by prose.
    - Recovers from truncated JSON by closing brackets.
    """
    if not text or not text.strip():
        return None

    cleaned = _clean_json_string(text)

    # Strategy 1: Direct parse with strict=False (allows literal newlines inside strings)
    try:
        parsed = json.loads(cleaned, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Strategy 2: Remove trailing commas (e.g. {"a": 1, })
    no_trailing = re.sub(r",\s*([\}\]])", r"\1", cleaned)
    try:
        parsed = json.loads(no_trailing, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Strategy 3: Extract outermost JSON object { ... }
    match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try trailing comma fix on candidate
        no_trailing_candidate = re.sub(r",\s*([\}\]])", r"\1", candidate)
        try:
            parsed = json.loads(no_trailing_candidate, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Strategy 4: Fallback to ast.literal_eval for single-quoted Python dict representations
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Strategy 5: Handle truncated JSON by appending closing tokens
    base_candidate = match.group(1) if match else cleaned
    repairs = [
        base_candidate + '"}]}',
        base_candidate + '"]}',
        base_candidate + '"]}}',
        base_candidate + "}",
        base_candidate + '"}',
        base_candidate + '"]',
    ]
    for attempt in repairs:
        try:
            parsed = json.loads(attempt, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return None


# Backward-compatible alias
_safe_parse_json = safe_parse_json
