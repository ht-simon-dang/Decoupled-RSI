"""Secret redaction for logs, manifests and reports."""

from __future__ import annotations

import re
from typing import Any

# Ordered (name, pattern) list of secret-like strings.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{16,}")),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{16,}")),
    ("generic_secret_kv", re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*\S{6,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{6,}")),
]

_REDACTION = "«REDACTED»"

# Environment variable names whose *values* must never be surfaced.
_SENSITIVE_ENV_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")


def redact(text: str) -> str:
    """Replace secret-like substrings with a placeholder."""
    if not text:
        return text
    out = text
    for _name, pattern in SECRET_PATTERNS:
        out = pattern.sub(_REDACTION, out)
    return out


def redact_obj(obj: Any) -> Any:
    """Recursively redact strings within dict/list structures."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v) for v in obj)
    return obj


def redact_env_names(env: dict[str, str]) -> list[str]:
    """Return only env var *names* (values dropped), for manifest recording."""
    return sorted(env.keys())


def is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(h in upper for h in _SENSITIVE_ENV_HINTS)
