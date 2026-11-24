import asyncio
import json
import time
from typing import Any, Optional

from app.agent.models import AgentRunResult, ToolCallRecord
from app.agent.tools.base import ToolOutput
from app.agent.tools.registry import ToolRegistry, get_default_registry
from app.core.config import get_settings
from app.core.logger import log
from app.llm import LLMRequest, ToolCallPayload, call_llm

settings = get_settings()

AGENT_SYSTEM_PROMPT = """You are an elite Autonomous Sales Intelligence Agent.
Your objective is to thoroughly investigate, qualify, and synthesize commercial intelligence on target companies based EXCLUSIVELY on verifiable factual evidence.

OPERATING PRINCIPLES:
1. AUTONOMOUS INVESTIGATION: Use the available tools to search for company websites, read their primary product and solutions pages, and evaluate qualification fit.
2. PARALLEL DISCOVERY: When you need multiple facts or wish to inspect multiple URLs or search queries, call the relevant tools in parallel within a single turn.
3. GROUNDED FINDINGS: Never hallucinate features, customer segments, or business models. Rely strictly on facts returned by tools.
4. SYNTHESIS COMPLETION: When you have gathered sufficient concrete evidence, respond with a comprehensive final answer detailing:
   - Core company offerings and primary products/services
   - Target market and business model
   - Qualification assessment against target criteria
   - Specific supporting evidence and quotes from sources found."""


class AgentRunner:
    """
    Autonomous multi-turn agent execution loop with deterministic tool calling,
    parallel execution, loop guards, cycle detection, and forced synthesis.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        model: Optional[str] = None,
        per_tool_timeout: float = 10.0,
        max_steps: int = 6,
        allowed_tools: Optional[list[str]] = None,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.model = model or settings.default_model
        self.per_tool_timeout = per_tool_timeout
        self.max_steps = max_steps
        self.allowed_tools = allowed_tools

    async def run(
        self,
        goal: str,
        company_name: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
        max_steps: Optional[int] = None,
        allowed_tools: Optional[list[str]] = None,
        request_id: Optional[str] = None,
    ) -> AgentRunResult:
        """
        Execute the autonomous agent loop until the objective is satisfied
        or the max_steps ceiling is reached.
        """
        effective_max_steps = max_steps if max_steps is not None else self.max_steps
        start_time = time.perf_counter()
        req_id = request_id or f"agt_{int(start_time * 1000)}"

        log.info(
            f"[{req_id}] Starting agent run: goal='{goal[:80]}...', "
            f"company='{company_name}', domain='{domain}', max_steps={effective_max_steps}"
        )

        # 1. Build initial conversation message history
        initial_user_prompt = f"Goal: {goal}"
        if company_name:
            initial_user_prompt += f"\nTarget Company: {company_name}"
        if domain:
            initial_user_prompt += f"\nTarget Domain: {domain}"
        if context:
            initial_user_prompt += f"\nExisting Notes / Context:\n{context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_prompt},
        ]

        tool_call_trace: list[ToolCallRecord] = []
        call_signatures: list[str] = []
        total_tokens = 0
        effective_allowed = allowed_tools if allowed_tools is not None else self.allowed_tools
        openai_tools = self.registry.get_openai_tools(allowed_names=effective_allowed)

        # 2. Multi-turn execution loop
        for step in range(effective_max_steps):
            log.info(f"[{req_id}] Agent step {step + 1}/{effective_max_steps} started")

            # Call LLM with current history and available tool definitions
            req = LLMRequest(
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                model=self.model,
                temperature=0.2,
                max_tokens=512,
                request_id=f"{req_id}_s{step}",
            )

            resp = await call_llm(req)
            total_tokens += resp.total_tokens

            # Case A: Model emitted final answer (no tool calls)
            if not resp.tool_calls:
                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                log.info(
                    f"[{req_id}] Agent satisfied goal at step {step + 1} "
                    f"(latency={latency_ms:.0f}ms, total_tokens={total_tokens})"
                )
                return AgentRunResult(
                    final_answer=resp.content,
                    steps_taken=step + 1,
                    is_completed=True,
                    tool_calls=tool_call_trace,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )

            # Case B: Model requested tool invocations
            # 1. Append assistant message with tool_calls (strict OpenAI/Gemini protocol)
            if resp.raw_message:
                messages.append(resp.raw_message)
            else:
                assistant_tool_calls: list[dict[str, Any]] = []
                for tc in resp.tool_calls:
                    tc_dict: dict[str, Any] = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    if tc.extra_content:
                        tc_dict["extra_content"] = tc.extra_content
                    assistant_tool_calls.append(tc_dict)

                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or None,
                        "tool_calls": assistant_tool_calls,
                    }
                )

            # 2. Cycle detection: check if identical tool signature was called in recent turns
            step_sig = "|".join(
                f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}" for tc in resp.tool_calls
            )
            cycle_detected = step_sig in call_signatures[-2:] if call_signatures else False
            call_signatures.append(step_sig)

            if cycle_detected:
                log.warning(
                    f"[{req_id}] Cycle detected at step {step + 1}: "
                    f"Identical tool call repeated. Injecting breaker directive."
                )

            # 3. Parallel execute all tool calls via asyncio.gather
            async def _exec_single(
                tc: ToolCallPayload,
            ) -> tuple[ToolCallPayload, ToolOutput, float]:
                t0 = time.perf_counter()
                out = await self.registry.execute_tool(
                    name=tc.name,
                    args=tc.arguments,
                    timeout_seconds=self.per_tool_timeout,
                )
                dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return tc, out, dur_ms

            tool_results = await asyncio.gather(*[_exec_single(tc) for tc in resp.tool_calls])

            # 4. Append ALL tool responses in 1:1 pairing matching their tool_call_id
            for tc, out, dur_ms in tool_results:
                is_err = not out.success
                output_payload = out.data if out.success else {"error": out.error}

                # If cycle was detected, inject guidance into tool result
                if cycle_detected:
                    warning_msg = (
                        "Warning: This exact tool call was already executed previously. "
                        "Do not repeat this query. Proceed to synthesize your final findings."
                    )
                    if isinstance(output_payload, dict):
                        output_payload["_system_note"] = warning_msg
                    elif isinstance(output_payload, list):
                        output_payload = {
                            "results": output_payload,
                            "_system_note": warning_msg,
                        }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(output_payload),
                    }
                )

                tool_call_trace.append(
                    ToolCallRecord(
                        step=step,
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        arguments=tc.arguments,
                        output=output_payload,
                        is_error=is_err,
                        latency_ms=dur_ms,
                    )
                )

        # 3. Forced Final Synthesis Turn if max_steps reached without final text
        log.warning(
            f"[{req_id}] Reached max_steps ({effective_max_steps}) without final answer. "
            f"Triggering forced final synthesis turn."
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum step limit. "
                    "Synthesize your final evaluation, company offerings, and key findings "
                    "based strictly on the evidence and tool results gathered so far."
                ),
            }
        )

        synth_req = LLMRequest(
            messages=messages,
            tools=None,
            tool_choice=None,
            model=self.model,
            temperature=0.2,
            max_tokens=1500,
            request_id=f"{req_id}_forced_synth",
        )

        synth_resp = await call_llm(synth_req)
        total_tokens += synth_resp.total_tokens
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return AgentRunResult(
            final_answer=synth_resp.content,
            steps_taken=effective_max_steps + 1,
            is_completed=False,  # Max steps ceiling was reached
            tool_calls=tool_call_trace,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )
