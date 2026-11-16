"""Abstract base class for all baseline defenses."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDefense(ABC):
    """Every baseline defense must implement these two methods."""

    @abstractmethod
    def process_output(self, tool_output: str, tool_name: str) -> str:
        """Process raw tool output, returning a defended version.

        Args:
            tool_output: The raw string returned by the tool.
            tool_name: The name of the tool that produced the output.

        Returns:
            The post-processed output string.
        """
        ...

    @abstractmethod
    def get_system_addendum(self) -> str:
        """Return text to append to the system prompt for this defense.

        Returns:
            A system-prompt fragment instructing the model how to treat
            defended content.
        """
        ...
