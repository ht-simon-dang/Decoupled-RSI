"""Reasoning backends, and the roles they are asked to fill.

Roles, not vendors, are what the loop depends on. ``deep_researcher`` means "the
strongest reasoning available"; which process serves it is a deployment
question, and the registry answers it.
"""

from .base import BackendError, BackendResult, BackendUnavailable, ReasoningBackend
from .claude_cli_backend import ClaudeCLIBackend
from .ollama_backend import OllamaBackend
from .registry import ROLES, BackendRegistry

__all__ = [
    "ROLES",
    "BackendError",
    "BackendRegistry",
    "BackendResult",
    "BackendUnavailable",
    "ClaudeCLIBackend",
    "OllamaBackend",
    "ReasoningBackend",
]
