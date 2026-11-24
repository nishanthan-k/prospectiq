import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from app.core.logger import log


class ToolPermissionLevel(str, Enum):
    """Permission tier for agent tools."""

    READ_ONLY = "read_only"
    WRITE_ACTION = "write_action"


class ToolOutput(BaseModel):
    """Unified result container returned by all tool executions."""

    success: bool
    data: Any
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for all deterministic agent tools."""

    name: str
    description: str
    args_schema: type[BaseModel]
    permission_level: ToolPermissionLevel = ToolPermissionLevel.READ_ONLY
    requires_approval: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert Pydantic argument schema to OpenAI/Gemini function-calling specification."""
        json_schema = self.args_schema.model_json_schema()
        # Clean top-level title for cleaner prompt presentation
        json_schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": json_schema,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Concrete tool implementation. Must be implemented by subclasses."""
        pass

    async def run_with_guard(
        self,
        raw_args: dict[str, Any],
        timeout_seconds: float = 10.0,
    ) -> ToolOutput:
        """
        Safely validate arguments against Pydantic schema and execute with timeout
        and robust exception containment.
        """
        try:
            validated_args = self.args_schema.model_validate(raw_args)
        except ValidationError as err:
            log.warning(f"Tool [{self.name}] argument validation failed: {err}")
            return ToolOutput(
                success=False,
                data=None,
                error=f"Invalid arguments for tool '{self.name}': {err}",
            )

        try:
            data = await asyncio.wait_for(
                self.execute(**validated_args.model_dump()),
                timeout=timeout_seconds,
            )
            return ToolOutput(success=True, data=data, error=None)
        except (TimeoutError, asyncio.TimeoutError):
            log.warning(f"Tool [{self.name}] execution timed out after {timeout_seconds}s")
            return ToolOutput(
                success=False,
                data=None,
                error=f"Tool '{self.name}' timed out after {timeout_seconds}s",
            )
        except Exception as exc:
            log.warning(f"Tool [{self.name}] execution failed with error: {exc}")
            return ToolOutput(
                success=False,
                data=None,
                error=f"Tool '{self.name}' execution failed: {exc}",
            )
