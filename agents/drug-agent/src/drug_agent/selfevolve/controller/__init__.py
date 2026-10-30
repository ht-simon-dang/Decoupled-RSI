"""The loop that drives analysis → hypothesis → candidate → evaluation → gate."""

from .loop import EvolutionController

__all__ = ["EvolutionController"]
