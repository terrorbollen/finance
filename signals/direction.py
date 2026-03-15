"""Canonical trading direction enum shared across all modules."""

from enum import Enum


class Direction(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
