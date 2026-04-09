"""Re-export shim — canonical definition lives in models/direction.py.

Kept for backwards compatibility with any code that imports from signals.direction.
"""

from models.direction import (  # noqa: F401
    BUY_IDX,
    HOLD_IDX,
    IDX_TO_DIRECTION,
    SELL_IDX,
    Direction,
)
