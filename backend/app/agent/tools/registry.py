from typing import Any, Optional

from app.agent.tools.base import BaseTool, ToolOutput, ToolPermissionLevel
from app.agent.tools.definitions import (
    EvaluateLeadFitTool,
    FetchWebpageTool,
    SearchCompanyTool,
)
from app.core.logger import log


class ToolRegistry:
    """Manages tool registration, OpenAI schema generation, and safe execution dispatching."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance in the registry."""
        if tool.name in self._tools:
            log.warning(f"Overwriting existing tool registration: '{tool.name}'")
        self._tools[tool.name] = tool
        log.debug(f"Registered tool '{tool.name}' ({tool.permission_level.value})")

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by its snake_case name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    def list_tools(
        self,
        permission_filter: Optional[ToolPermissionLevel] = None,
        allowed_names: Optional[list[str]] = None,
    ) -> list[BaseTool]:
        """List registered tools matching optional permission and allowlist filters."""
        tools = list(self._tools.values())
        if permission_filter is not None:
            tools = [t for t in tools if t.permission_level == permission_filter]
        if allowed_names is not None:
            allowed_set = set(allowed_names)
            tools = [t for t in tools if t.name in allowed_set]
        return tools

    def get_openai_tools(
        self,
        allowed_names: Optional[list[str]] = None,
        permission_filter: Optional[ToolPermissionLevel] = None,
    ) -> list[dict[str, Any]]:
        """Generate OpenAI/Gemini function-calling tool schema definitions."""
        tools = self.list_tools(
            permission_filter=permission_filter,
            allowed_names=allowed_names,
        )
        return [tool.to_openai_schema() for tool in tools]

    async def execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        timeout_seconds: float = 10.0,
        authorized_permission_level: ToolPermissionLevel = ToolPermissionLevel.READ_ONLY,
        is_approved: bool = False,
    ) -> ToolOutput:
        """
        Safely execute a registered tool by name with permissions, approvals,
        timeout, and exception containment.
        If the tool does not exist, returns a structured error ToolOutput.
        """
        tool = self.get(name)
        if not tool:
            log.warning(f"Tool execution requested for unknown tool: '{name}'")
            return ToolOutput(
                success=False,
                data=None,
                error=f"Tool '{name}' is not registered or not recognized.",
            )

        # 1. Enforce permission level: write_action tools require WRITE_ACTION authorization
        if (
            tool.permission_level == ToolPermissionLevel.WRITE_ACTION
            and authorized_permission_level != ToolPermissionLevel.WRITE_ACTION
        ):
            log.warning(
                f"Permission denied for tool '{name}': requires {tool.permission_level.value}, "
                f"authorized: {authorized_permission_level.value}"
            )
            return ToolOutput(
                success=False,
                data=None,
                error=(
                    f"Permission denied: Tool '{name}' requires '{tool.permission_level.value}' "
                    f"permission level (authorized: '{authorized_permission_level.value}')."
                ),
            )

        # 2. Enforce approval requirement
        if tool.requires_approval and not is_approved:
            log.warning(f"Approval required for tool '{name}': execution rejected without approval")
            return ToolOutput(
                success=False,
                data=None,
                error=f"Approval required: Tool '{name}' requires explicit authorization before execution.",
            )

        log.info(f"Dispatching tool '{name}' with timeout={timeout_seconds}s")
        return await tool.run_with_guard(args, timeout_seconds=timeout_seconds)


def get_default_registry() -> ToolRegistry:
    """Instantiate and return a ToolRegistry populated with the standard core sales tools."""
    registry = ToolRegistry()
    registry.register(SearchCompanyTool())
    registry.register(FetchWebpageTool())
    registry.register(EvaluateLeadFitTool())
    return registry
