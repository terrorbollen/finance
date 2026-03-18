"""Confidence calibration for trading signals.

Uses isotonic regression to map raw model confidence scores to
calibrated probabilities that reflect actual historical accuracy.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass
class CalibrationBucket:
    """A single calibration bucket with raw confidence range and calibrated value."""

    raw_min: float
    raw_max: float
    calibrated: float
    sample_count: int


class ConfidenceCalibrator:
    """
    Calibrates model confidence scores based on historical backtest data.

    Uses isotonic regression principles: maps raw confidence buckets to
    their actual observed accuracy, ensuring monotonicity.
    """

    def __init__(self, num_buckets: int = 10):
        """
        Initialize the calibrator.

        Args:
            num_buckets: Number of confidence buckets for calibration
        """
        self.num_buckets = num_buckets
        self.buckets: list[CalibrationBucket] = []
        self.is_fitted = False
        self.fitted_at: datetime | None = None

        # Minimum samples required per bucket for reliable calibration
        self.min_samples_per_bucket = 10

    def fit(
        self,
        raw_confidences: np.ndarray,
        correct: np.ndarray,
        enforce_monotonic: bool = True,
    ) -> "ConfidenceCalibrator":
        """
        Fit the calibrator on historical predictions.

        Args:
            raw_confidences: Array of raw model confidence scores (0-1)
            correct: Boolean array indicating if prediction was correct
            enforce_monotonic: If True, ensure calibrated values are monotonic

        Returns:
            Self for method chaining
        """
        if len(raw_confidences) != len(correct):
            raise ValueError("Confidence and correct arrays must have same length")

        if len(raw_confidences) == 0:
            raise ValueError("Cannot fit calibrator with empty data")

        # Create bucket boundaries
        bucket_edges = np.linspace(0, 1, self.num_buckets + 1)

        raw_buckets = []
        for i in range(self.num_buckets):
            low, high = bucket_edges[i], bucket_edges[i + 1]

            # Find predictions in this bucket
            if i == self.num_buckets - 1:
                # Last bucket includes upper bound
                mask = (raw_confidences >= low) & (raw_confidences <= high)
            else:
                mask = (raw_confidences >= low) & (raw_confidences < high)

            bucket_correct = correct[mask]
            sample_count = len(bucket_correct)

            if sample_count >= self.min_samples_per_bucket:
                calibrated = float(np.mean(bucket_correct))
            else:
                # Not enough samples - use raw midpoint as fallback
                calibrated = (low + high) / 2

            raw_buckets.append(
                CalibrationBucket(
                    raw_min=low,
                    raw_max=high,
                    calibrated=calibrated,
                    sample_count=sample_count,
                )
            )
        # Enforce monotonicity using isotonic regression
        if enforce_monotonic:
            calibrated_values = [b.calibrated for b in raw_buckets]
            calibrated_values = self._isotonic_regression(calibrated_values)
            for i, bucket in enumerate(raw_buckets):
                bucket.calibrated = calibrated_values[i]

        self.buckets = raw_buckets
        self.is_fitted = True
        self.fitted_at = datetime.now(UTC)

        return self

    def _isotonic_regression(self, values: list[float]) -> list[float]:
        """
        Apply isotonic regression to ensure monotonically increasing output.

        Uses sklearn's IsotonicRegression which implements the correct O(n)
        pool adjacent violators algorithm.
        """
        if len(values) == 0:
            return values

        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        x = list(range(len(values)))
        return list(ir.fit_transform(x, values))

    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a single raw confidence score.

        Args:
            raw_confidence: Raw model confidence (0-1)

        Returns:
            Calibrated confidence (0-1)
        """
        if not self.is_fitted:
            # Return raw confidence if not fitted
            return raw_confidence

        # Clamp to valid range
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # Find the appropriate bucket
        for bucket in self.buckets:
            if bucket.raw_min <= raw_confidence <= bucket.raw_max:
                return bucket.calibrated

        # Fallback to nearest bucket
        if raw_confidence < self.buckets[0].raw_min:
            return self.buckets[0].calibrated
        return self.buckets[-1].calibrated

    def calibrate_batch(self, raw_confidences: np.ndarray) -> np.ndarray:
        """
        Calibrate an array of raw confidence scores.

        Args:
            raw_confidences: Array of raw model confidences (0-1)

        Returns:
            Array of calibrated confidences (0-1)
        """
        return np.array([self.calibrate(c) for c in raw_confidences])

    def get_calibration_table(self) -> str:
        """Get a formatted table showing the calibration mapping."""
        if not self.is_fitted:
            return "Calibrator not fitted"

        lines = [
            "Confidence Calibration Table",
            "-" * 50,
            f"{'Raw Range':<20} {'Calibrated':<15} {'Samples':<10}",
            "-" * 50,
        ]

        for bucket in self.buckets:
            raw_range = f"{bucket.raw_min * 100:.0f}%-{bucket.raw_max * 100:.0f}%"
            lines.append(
                f"{raw_range:<20} {bucket.calibrated * 100:.1f}%{'':<9} {bucket.sample_count:<10}"
            )

        return "\n".join(lines)

    def staleness_warning(self, max_days: int = 30) -> str | None:
        """Return a warning string if the calibrator is older than max_days, else None."""
        if self.fitted_at is None:
            return "Calibration age unknown (fitted_at missing) — rerun `calibrate` to refresh."
        age = datetime.now(UTC) - self.fitted_at
        if age.days > max_days:
            return (
                f"Calibration is {age.days} days old (limit: {max_days}d) — "
                "rerun `calibrate` to refresh."
            )
        return None

    def save(self, path: str):
        """Save calibration parameters to JSON file."""
        data = {
            "num_buckets": self.num_buckets,
            "min_samples_per_bucket": self.min_samples_per_bucket,
            "is_fitted": self.is_fitted,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "buckets": [
                {
                    "raw_min": b.raw_min,
                    "raw_max": b.raw_max,
                    "calibrated": b.calibrated,
                    "sample_count": b.sample_count,
                }
                for b in self.buckets
            ],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConfidenceCalibrator":
        """Load calibration parameters from JSON file."""
        with open(path) as f:
            data = json.load(f)

        calibrator = cls(num_buckets=data["num_buckets"])
        calibrator.min_samples_per_bucket = data["min_samples_per_bucket"]
        calibrator.is_fitted = data["is_fitted"]
        raw_ts = data.get("fitted_at")
        calibrator.fitted_at = datetime.fromisoformat(raw_ts) if raw_ts else None
        calibrator.buckets = [
            CalibrationBucket(
                raw_min=b["raw_min"],
                raw_max=b["raw_max"],
                calibrated=b["calibrated"],
                sample_count=b["sample_count"],
            )
            for b in data["buckets"]
        ]

        return calibrator

    @classmethod
    def from_backtest_results(
        cls,
        backtest_results: list[dict],
        horizon: int = 5,
        num_buckets: int = 10,
    ) -> "ConfidenceCalibrator":
        """
        Create and fit a calibrator from backtest results.

        Args:
            backtest_results: List of backtest result dictionaries
            horizon: Which prediction horizon to use for calibration
            num_buckets: Number of calibration buckets

        Returns:
            Fitted ConfidenceCalibrator
        """
        confidences = []
        correct = []

        for result in backtest_results:
            for daily in result.get("daily_predictions", []):
                predictions = daily.get("predictions", {})
                # Handle both string and int keys
                pred = predictions.get(str(horizon)) or predictions.get(horizon)
                if pred and pred.get("is_correct") is not None:
                    confidences.append(pred["confidence"])
                    correct.append(pred["is_correct"])

        if not confidences:
            raise ValueError(f"No valid predictions found for horizon {horizon}")

        calibrator = cls(num_buckets=num_buckets)
        calibrator.fit(np.array(confidences), np.array(correct))

        return calibrator


class DirectionalCalibrator:
    """
    Holds separate ConfidenceCalibrators for BUY, SELL, and HOLD.

    A single global calibrator ignores the fact that the model may be
    systematically over- or under-confident in different directions.
    This class fits one calibrator per direction so each is corrected
    independently.
    """

    DIRECTIONS = ("BUY", "SELL", "HOLD")

    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets
        self.calibrators: dict[str, ConfidenceCalibrator] = {}

    @property
    def is_fitted(self) -> bool:
        return any(c.is_fitted for c in self.calibrators.values())

    def is_fitted_for(self, direction: str) -> bool:
        return direction in self.calibrators and self.calibrators[direction].is_fitted

    def fit(
        self,
        direction: str,
        raw_confidences: np.ndarray,
        correct: np.ndarray,
    ) -> "DirectionalCalibrator":
        """Fit the calibrator for a single direction."""
        cal = ConfidenceCalibrator(num_buckets=self.num_buckets)
        cal.fit(raw_confidences, correct)
        self.calibrators[direction] = cal
        return self

    def calibrate(self, direction: str, raw_confidence: float) -> float:
        """
        Return calibrated confidence for the given direction.
        Falls back to the raw value if that direction has no calibrator.
        """
        if self.is_fitted_for(direction):
            return self.calibrators[direction].calibrate(raw_confidence)
        return raw_confidence

    def get_calibration_table(self) -> str:
        """Get a formatted table showing the calibration mapping for each direction."""
        if not self.is_fitted:
            return "Directional calibrator not fitted"

        sections = []
        for direction in self.DIRECTIONS:
            if self.is_fitted_for(direction):
                cal = self.calibrators[direction]
                header = f"  {direction}"
                lines = [header, "  " + "-" * 50]
                lines.append(f"  {'Raw Range':<20} {'Calibrated':<15} {'Samples':<10}")
                lines.append("  " + "-" * 50)
                for bucket in cal.buckets:
                    raw_range = f"{bucket.raw_min * 100:.0f}%-{bucket.raw_max * 100:.0f}%"
                    lines.append(
                        f"  {raw_range:<20} {bucket.calibrated * 100:.1f}%{'':<9} {bucket.sample_count:<10}"
                    )
                sections.append("\n".join(lines))
            else:
                sections.append(f"  {direction}: not fitted (insufficient samples)")

        return "\n\n".join(sections)

    def staleness_warning(self, max_days: int = 30) -> str | None:
        """Return a warning if any fitted direction calibrator is stale, else None."""
        for cal in self.calibrators.values():
            warning = cal.staleness_warning(max_days)
            if warning:
                return warning
        return None

    def save(self, path: str):
        """Save all per-direction calibrators to a single JSON file."""
        data = {
            "num_buckets": self.num_buckets,
            "calibrators": {
                direction: {
                    "num_buckets": cal.num_buckets,
                    "min_samples_per_bucket": cal.min_samples_per_bucket,
                    "is_fitted": cal.is_fitted,
                    "fitted_at": cal.fitted_at.isoformat() if cal.fitted_at else None,
                    "buckets": [
                        {
                            "raw_min": b.raw_min,
                            "raw_max": b.raw_max,
                            "calibrated": b.calibrated,
                            "sample_count": b.sample_count,
                        }
                        for b in cal.buckets
                    ],
                }
                for direction, cal in self.calibrators.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DirectionalCalibrator":
        """Load per-direction calibrators from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        dc = cls(num_buckets=data["num_buckets"])
        for direction, cal_data in data["calibrators"].items():
            cal = ConfidenceCalibrator(num_buckets=cal_data["num_buckets"])
            cal.min_samples_per_bucket = cal_data["min_samples_per_bucket"]
            cal.is_fitted = cal_data["is_fitted"]
            raw_ts = cal_data.get("fitted_at")
            cal.fitted_at = datetime.fromisoformat(raw_ts) if raw_ts else None
            cal.buckets = [
                CalibrationBucket(
                    raw_min=b["raw_min"],
                    raw_max=b["raw_max"],
                    calibrated=b["calibrated"],
                    sample_count=b["sample_count"],
                )
                for b in cal_data["buckets"]
            ]
            dc.calibrators[direction] = cal

        return dc

    @classmethod
    def from_backtest_results(
        cls,
        backtest_results: list[dict],
        horizon: int = 5,
        num_buckets: int = 10,
    ) -> "DirectionalCalibrator":
        """
        Create and fit a DirectionalCalibrator from backtest results.

        Splits predictions by direction (BUY/SELL/HOLD) and fits a separate
        calibrator for each direction that has enough samples.

        Args:
            backtest_results: List of backtest result dictionaries
            horizon: Which prediction horizon to use for calibration
            num_buckets: Number of calibration buckets per direction

        Returns:
            Fitted DirectionalCalibrator (fitted for whichever directions had data)
        """
        direction_data: dict[str, tuple[list[float], list[bool]]] = {
            d: ([], []) for d in cls.DIRECTIONS
        }

        for result in backtest_results:
            for daily in result.get("daily_predictions", []):
                predictions = daily.get("predictions", {})
                pred = predictions.get(str(horizon)) or predictions.get(horizon)
                if pred and pred.get("is_correct") is not None:
                    direction = pred.get("predicted_signal", "HOLD")
                    if direction in direction_data:
                        direction_data[direction][0].append(pred["confidence"])
                        direction_data[direction][1].append(bool(pred["is_correct"]))

        dc = cls(num_buckets=num_buckets)
        for direction, (confidences, correct) in direction_data.items():
            if len(confidences) >= num_buckets:
                dc.fit(direction, np.array(confidences), np.array(correct))

        if not dc.is_fitted:
            raise ValueError(f"No direction had enough data for horizon {horizon}")

        return dc
