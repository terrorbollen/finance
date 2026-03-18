"""Canonical trading direction enum shared across all modules."""

from enum import Enum


class Direction(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


# Integer class indices used by the softmax output layer.
# BUY=0, HOLD=1, SELL=2 — this mapping is fixed at training time and must
# not change without retraining the model.  Reference these constants instead
# of hardcoding 0/1/2 so that any future encoding change is a one-line edit.
BUY_IDX: int = 0
HOLD_IDX: int = 1
SELL_IDX: int = 2

# Ordered list matching the softmax output dimension — index i → Direction
IDX_TO_DIRECTION: list[Direction] = [Direction.BUY, Direction.HOLD, Direction.SELL]
