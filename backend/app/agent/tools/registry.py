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
        """Generate OpenAI/Groq function-calling tool schema definitions."""
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
    ) -> ToolOutput:
        """
        Safely execute a registered tool by name with timeout and exception containment.
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

        log.info(f"Dispatching tool '{name}' with timeout={timeout_seconds}s")
        return await tool.run_with_guard(args, timeout_seconds=timeout_seconds)


def get_default_registry() -> ToolRegistry:
    """Instantiate and return a ToolRegistry populated with the standard core sales tools."""
    registry = ToolRegistry()
    registry.register(SearchCompanyTool())
    registry.register(FetchWebpageTool())
    registry.register(EvaluateLeadFitTool())
    return registry
