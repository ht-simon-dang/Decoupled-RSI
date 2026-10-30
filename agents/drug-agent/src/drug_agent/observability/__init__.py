"""Observability: structured logs, spans, metrics, tool metering, run journals.

Three layers, in increasing durability:

  * :mod:`telemetry` — structured logs and spans for the process that is running.
  * :mod:`toolmeter` — in-memory metering of tool calls for one unit of work.
  * :mod:`journal`   — the per-request record that outlives the process: what was
    asked, every model exchange, which tools ran and how they performed, and
    what came out. It lives in ``running-history/``, outside the disposable run
    state, with :mod:`history` maintaining the folder's indexes and
    :mod:`training` exporting its dialogs for fine-tuning.
"""

from __future__ import annotations

from .history import ensure_readme, migrate, rebuild_index, write_human_index
from .journal import (
    ChatExchange,
    RunJournal,
    default_journal_root,
    find_request_dir,
    iter_request_dirs,
    list_requests,
    load_request,
    tool_statistics,
)
from .telemetry import Span, Telemetry, configure_logging, get_logger
from .toolmeter import ToolLedger, ToolUsage, carry_metering, instrument, measure, metering
from .training import export_dialogs, iter_dialogs

__all__ = [
    "ChatExchange",
    "RunJournal",
    "Span",
    "Telemetry",
    "ToolLedger",
    "ToolUsage",
    "carry_metering",
    "configure_logging",
    "default_journal_root",
    "ensure_readme",
    "export_dialogs",
    "find_request_dir",
    "get_logger",
    "instrument",
    "iter_dialogs",
    "iter_request_dirs",
    "list_requests",
    "load_request",
    "measure",
    "metering",
    "migrate",
    "rebuild_index",
    "tool_statistics",
    "write_human_index",
]
