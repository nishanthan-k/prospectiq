import asyncio
import json
import time
from typing import Any, Optional

from app.agent.models import AgentRunResult, ToolCallRecord
from app.agent.tools.base import ToolOutput, ToolPermissionLevel
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
    bounded parallel execution, concurrency limits, loop guards, cycle detection,
    and forced synthesis.

    Note (Level 3 Approval & Permission Limitation):
        `ToolRegistry.execute_tool()` enforces both permission levels and
        `requires_approval` checks. Currently, `AgentRunner` passes
        `authorized_permission_level` (defaulting to `READ_ONLY`) but does not
        grant `is_approved=True` or filter `get_openai_tools()` schemas by
        permission level. Approval-required / write tools are safely rejected at
        execution time; full Human-in-the-Loop (HITL) approval pause/resume is
        deferred to Level 5 (Durable Workflows).
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        model: Optional[str] = None,
        per_tool_timeout: float = 10.0,
        max_steps: int = 6,
        allowed_tools: Optional[list[str]] = None,
        max_tool_calls_per_turn: int = 5,
        max_concurrency: int = 3,
        authorized_permission_level: ToolPermissionLevel = ToolPermissionLevel.READ_ONLY,
    ) -> None:
        self.registry = registry or get_default_registry()
        self.model = model or settings.default_model

        if (
            isinstance(per_tool_timeout, bool)
            or not isinstance(per_tool_timeout, (int, float))
            or per_tool_timeout <= 0
        ):
            raise ValueError(f"per_tool_timeout must be > 0 (got {per_tool_timeout})")
        self.per_tool_timeout = float(per_tool_timeout)

        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or not (1 <= max_steps <= 20)
        ):
            raise ValueError(f"max_steps must be an integer between 1 and 20 (got {max_steps})")
        self.max_steps = max_steps

        if allowed_tools is not None:
            for tool_name in allowed_tools:
                if not self.registry.has(tool_name):
                    raise ValueError(
                        f"Unknown tool in allowed_tools: '{tool_name}' is not registered."
                    )
        self.allowed_tools = allowed_tools

        if (
            isinstance(max_tool_calls_per_turn, bool)
            or not isinstance(max_tool_calls_per_turn, int)
            or max_tool_calls_per_turn < 1
        ):
            raise ValueError(
                f"max_tool_calls_per_turn must be an integer >= 1 (got {max_tool_calls_per_turn})"
            )
        self.max_tool_calls_per_turn = max_tool_calls_per_turn

        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError(f"max_concurrency must be an integer >= 1 (got {max_concurrency})")
        self.max_concurrency = max_concurrency
        self.authorized_permission_level = authorized_permission_level

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
        # 1. Internal validation of inputs
        if not goal or not goal.strip():
            raise ValueError("goal cannot be empty or only whitespace")

        effective_max_steps = max_steps if max_steps is not None else self.max_steps
        if not isinstance(effective_max_steps, int) or not (1 <= effective_max_steps <= 20):
            raise ValueError(
                f"max_steps must be an integer between 1 and 20 (got {effective_max_steps})"
            )

        effective_allowed = allowed_tools if allowed_tools is not None else self.allowed_tools
        if effective_allowed is not None:
            for tool_name in effective_allowed:
                if not self.registry.has(tool_name):
                    raise ValueError(
                        f"Unknown tool in allowed_tools: '{tool_name}' is not registered."
                    )

        start_time = time.perf_counter()
        req_id = request_id or f"agt_{int(start_time * 1000)}"

        log.info(
            f"[{req_id}] Starting agent run: goal='{goal[:80]}...', "
            f"company='{company_name}', domain='{domain}', max_steps={effective_max_steps}"
        )

        # 2. Build initial conversation message history
        initial_user_prompt = f"Goal: {goal.strip()}"
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
        openai_tools = self.registry.get_openai_tools(allowed_names=effective_allowed)

        # Semaphore to bound parallel tool concurrency
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # 3. Multi-turn execution loop
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
                clean_content = (resp.content or "").strip()
                if not clean_content:
                    log.warning(
                        f"[{req_id}] Model returned empty content without tool calls at step {step + 1}."
                    )
                    # Controlled retry if steps remain
                    if step < effective_max_steps - 1:
                        messages.append(
                            {
                                "role": "user",
                                "content": "Your previous response was empty. Please provide your findings or invoke an appropriate tool.",
                            }
                        )
                        continue
                    else:
                        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                        return AgentRunResult(
                            final_answer="Agent failed to produce a final answer (received empty response).",
                            steps_taken=step + 1,
                            is_completed=False,
                            tool_calls=tool_call_trace,
                            total_tokens=total_tokens,
                            latency_ms=latency_ms,
                        )

                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                log.info(
                    f"[{req_id}] Agent satisfied goal at step {step + 1} "
                    f"(latency={latency_ms:.0f}ms, total_tokens={total_tokens})"
                )
                return AgentRunResult(
                    final_answer=clean_content,
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

            # 3. Parallel execute tool calls with concurrency and bounds guards
            async def _exec_single(
                idx: int,
                tc: ToolCallPayload,
            ) -> tuple[ToolCallPayload, ToolOutput, float]:
                # Enforce maximum tool calls per turn
                if idx >= self.max_tool_calls_per_turn:
                    log.warning(
                        f"[{req_id}] Rejecting tool call {idx + 1} ('{tc.name}'): "
                        f"exceeded max_tool_calls_per_turn ({self.max_tool_calls_per_turn})"
                    )
                    return (
                        tc,
                        ToolOutput(
                            success=False,
                            data=None,
                            error=(
                                f"Tool execution rejected: exceeded maximum allowable tool calls "
                                f"per turn ({self.max_tool_calls_per_turn})."
                            ),
                        ),
                        0.0,
                    )

                # Reject malformed JSON arguments explicitly before execution
                if tc.parse_error:
                    log.warning(
                        f"[{req_id}] Tool '{tc.name}' received malformed arguments: {tc.parse_error}"
                    )
                    return (
                        tc,
                        ToolOutput(
                            success=False,
                            data=None,
                            error=f"Malformed tool call arguments: {tc.parse_error}",
                        ),
                        0.0,
                    )

                t0 = time.perf_counter()
                async with semaphore:
                    out = await self.registry.execute_tool(
                        name=tc.name,
                        args=tc.arguments,
                        timeout_seconds=self.per_tool_timeout,
                        authorized_permission_level=self.authorized_permission_level,
                    )
                dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return tc, out, dur_ms

            tool_results = await asyncio.gather(
                *[_exec_single(idx, tc) for idx, tc in enumerate(resp.tool_calls)]
            )

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

        # 4. Forced Final Synthesis Turn if max_steps reached without final text
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

        clean_synth = (synth_resp.content or "").strip()
        if not clean_synth:
            clean_synth = "Agent forced synthesis produced an empty response."

        return AgentRunResult(
            final_answer=clean_synth,
            steps_taken=effective_max_steps + 1,
            is_completed=False,  # Max steps ceiling was reached
            tool_calls=tool_call_trace,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )
