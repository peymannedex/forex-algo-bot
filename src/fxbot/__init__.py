"""Modular event-driven algorithmic foreign-exchange trading platform."""

from fxbot.domain.models import Bar, OHLC, SymbolSpec, Tick

__all__ = ["Bar", "OHLC", "SymbolSpec", "Tick"]
__version__ = "0.1.0"
