"""The gate a candidate must clear before it reaches the user-serving system."""

from .gate import GATE_CHECKS, compare, decide, render_report

__all__ = ["GATE_CHECKS", "compare", "decide", "render_report"]
