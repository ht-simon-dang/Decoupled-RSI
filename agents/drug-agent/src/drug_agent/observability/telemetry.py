"""Structured logging + span/metric collection.

Local mode writes newline-delimited JSON events and keeps an in-memory metric
registry. All log records pass through secret redaction. An OTLP exporter is a
documented seam (``telemetry='otlp'``) but never required for the default build.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TextIO, cast

import structlog

from ..security.redaction import redact_obj
from ..util import new_id, utcnow_iso

_CONFIGURED = False


class _LazyStderr(io.TextIOBase):
    """Write to whatever ``sys.stderr`` is *now*, not whatever it was at configure time.

    ``configure_logging`` latches on ``_CONFIGURED`` and runs once per process, but
    ``sys.stderr`` can be replaced afterwards — test runners and CLI harnesses swap in a
    capture buffer and close it when the command returns. Binding the stream object here
    would pin that buffer, and every later log write would raise "I/O operation on closed
    file", failing the *task* rather than the logging. Resolving it per write costs one
    attribute lookup and keeps logging incapable of breaking the run.
    """

    def write(self, data: str) -> int:
        return sys.stderr.write(data)

    def flush(self) -> None:
        sys.stderr.flush()


def _redact_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    return redact_obj(event_dict)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        format="%(message)s", stream=_LazyStderr(), level=getattr(logging, level, logging.INFO)
    )
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_processor,
    ]
    processors.append(structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, logging.INFO)),
        # cast: PrintLogger only ever calls .write()/.flush(), which _LazyStderr
        # provides; TextIOBase is not TextIO as far as the type checker is concerned.
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, _LazyStderr())),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "drug_agent", **context: Any) -> Any:
    configure_logging()
    return structlog.get_logger(name).bind(**context)


@dataclass
class Span:
    name: str
    span_id: str
    trace_id: str
    parent_id: str | None
    start: float
    attributes: dict[str, Any] = field(default_factory=dict)
    end: float | None = None
    status: str = "ok"

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": redact_obj(self.attributes),
            "ts": utcnow_iso(),
        }


class Telemetry:
    """Collects spans + metrics for a run. Local mode buffers in memory/JSONL."""

    def __init__(self, run_id: str | None = None, mode: str = "local") -> None:
        self.run_id = run_id
        self.mode = mode
        self.spans: list[Span] = []
        self.metrics: dict[str, float] = {}
        self.log = get_logger("telemetry", run_id=run_id or "-")

    def incr(self, name: str, value: float = 1.0) -> None:
        self.metrics[name] = self.metrics.get(name, 0.0) + value

    def gauge(self, name: str, value: float) -> None:
        self.metrics[name] = value

    @contextmanager
    def span(self, name: str, parent: Span | None = None, **attributes: Any):
        sp = Span(
            name=name,
            span_id=new_id("span"),
            trace_id=(parent.trace_id if parent else (self.run_id or new_id("trace"))),
            parent_id=parent.span_id if parent else None,
            start=time.perf_counter(),
            attributes=dict(attributes),
        )
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.attributes["error"] = type(exc).__name__
            raise
        finally:
            sp.end = time.perf_counter()
            self.spans.append(sp)
            self.incr(f"span.{name}.count")
            self.gauge(f"span.{name}.last_ms", sp.duration_ms)
            self.log.info("span", **sp.to_dict())

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metrics": dict(self.metrics),
            "spans": [s.to_dict() for s in self.spans],
        }
