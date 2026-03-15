"""Pydantic model for the training configuration saved to checkpoints/.

ModelConfig is the contract between training and inference. It is written
by ModelTrainer and read by SignalGenerator and Backtester. Using Pydantic
ensures that a missing or malformed field raises an immediate, clear error
at load time rather than a cryptic crash inside the pipeline.

See ARCHITECTURE.md §3 for the full field reference.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

#: Historical data fetch period for training and backtesting.
FETCH_PERIOD = "10y"


class ModelConfig(BaseModel):
    """Validated representation of signal_model_config.json."""

    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    sequence_length: int
    input_dim: int
    interval: str = "1d"
    training_fetch_date: date
    holdout_start_date: date
    buy_threshold: float = 0.015
    sell_threshold: float = -0.015
    prediction_horizons: list[int] = [5, 10, 20]

    @field_validator("feature_std")
    @classmethod
    def std_must_be_positive(cls, v: list[float]) -> list[float]:
        if any(s <= 0 for s in v):
            raise ValueError("All feature_std values must be positive (was prepare_data() called?)")
        return v

    @model_validator(mode="after")
    def dimensions_must_be_consistent(self) -> ModelConfig:
        n = self.input_dim
        if len(self.feature_columns) != n:
            raise ValueError(
                f"input_dim={n} but len(feature_columns)={len(self.feature_columns)}"
            )
        if len(self.feature_mean) != n:
            raise ValueError(
                f"input_dim={n} but len(feature_mean)={len(self.feature_mean)}"
            )
        if len(self.feature_std) != n:
            raise ValueError(
                f"input_dim={n} but len(feature_std)={len(self.feature_std)}"
            )
        return self

    # ------------------------------------------------------------------
    # Convenience accessors returning numpy arrays for pipeline use
    # ------------------------------------------------------------------

    @property
    def feature_mean_array(self) -> np.ndarray:
        return np.array(self.feature_mean)

    @property
    def feature_std_array(self) -> np.ndarray:
        return np.array(self.feature_std)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> ModelConfig:
        """Load and validate config from a JSON file."""
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str) -> None:
        """Write config to a JSON file, creating parent directories as needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @staticmethod
    def checkpoint_paths(name: str | None = None) -> dict[str, str]:
        """Return the canonical checkpoint file paths for a named model.

        Args:
            name: Model name (e.g. 'financials'). Uses 'checkpoints/<name>/'.
                  If None, uses the default 'checkpoints/' directory.
        """
        base = f"checkpoints/{name}" if name else "checkpoints"
        return {
            "weights": f"{base}/signal_model.weights.h5",
            "config": f"{base}/signal_model_config.json",
            "calibration": f"{base}/calibration.json",
            "calibration_directional": f"{base}/calibration_directional.json",
        }
