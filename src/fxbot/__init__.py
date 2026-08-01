"""Modular event-driven algorithmic foreign-exchange trading platform."""

from fxbot.domain.models import OHLC, Bar, SymbolSpec, Tick

__all__ = ["OHLC", "Bar", "SymbolSpec", "Tick"]
__version__ = "0.1.0"
