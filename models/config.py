"""Pydantic model for the training configuration saved to checkpoints/.

ModelConfig is the contract between training and inference. It is written
by ModelTrainer and read by SignalGenerator and Backtester. Using Pydantic
ensures that a missing or malformed field raises an immediate, clear error
at load time rather than a cryptic crash inside the pipeline.

See ARCHITECTURE.md §3 for the full field reference.
"""

from __future__ import annotations

import json
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
    buy_threshold: float
    sell_threshold: float
    prediction_horizons: list[int]
    tickers: list[str] = []

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
            raise ValueError(f"input_dim={n} but len(feature_columns)={len(self.feature_columns)}")
        if len(self.feature_mean) != n:
            raise ValueError(f"input_dim={n} but len(feature_mean)={len(self.feature_mean)}")
        if len(self.feature_std) != n:
            raise ValueError(f"input_dim={n} but len(feature_std)={len(self.feature_std)}")
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

    # ------------------------------------------------------------------
    # Model registry — maps short alias → versioned folder name
    # ------------------------------------------------------------------

    _REGISTRY = Path("checkpoints/REGISTRY.json")

    @staticmethod
    def resolve_name(name: str) -> str:
        """Resolve a short alias (e.g. 'indexes') to its versioned folder name.

        If the alias has no registry entry the name is returned unchanged,
        which preserves backward compatibility with pre-versioned directories.
        """
        if ModelConfig._REGISTRY.exists():
            registry = json.loads(ModelConfig._REGISTRY.read_text())
            if name in registry:
                return registry[name]
        return name

    @staticmethod
    def update_registry(alias: str, versioned_name: str) -> None:
        """Point alias → versioned_name in the checkpoint registry."""
        registry: dict[str, str] = {}
        if ModelConfig._REGISTRY.exists():
            registry = json.loads(ModelConfig._REGISTRY.read_text())
        registry[alias] = versioned_name
        ModelConfig._REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        ModelConfig._REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")

    @staticmethod
    def checkpoint_paths(name: str | None = None) -> dict[str, str]:
        """Return the canonical checkpoint file paths for a named model.

        If *name* is a registered alias (e.g. 'indexes') it is resolved to the
        latest versioned folder (e.g. 'indexes-20260421-143022') via the
        registry.  An explicit versioned name or an unregistered name is used
        as-is, preserving backward compatibility with pre-versioned directories.

        Args:
            name: Model alias or versioned name (e.g. 'indexes' or
                  'indexes-20260421-143022'). None uses 'checkpoints/'.
        """
        resolved = ModelConfig.resolve_name(name) if name else None
        base = f"checkpoints/{resolved}" if resolved else "checkpoints"
        return {
            "weights": f"{base}/signal_model.weights.h5",
            "config": f"{base}/signal_model_config.json",
            "calibration": f"{base}/calibration.json",
            "calibration_directional": f"{base}/calibration_directional.json",
        }
