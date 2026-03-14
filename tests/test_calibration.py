"""Tests for the ConfidenceCalibrator class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from signals.calibration import CalibrationBucket, ConfidenceCalibrator


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator functionality."""

    def test_calibrator_initialization(self):
        """Test that calibrator initializes correctly."""
        calibrator = ConfidenceCalibrator(num_buckets=10)

        assert calibrator.num_buckets == 10
        assert calibrator.is_fitted is False
        assert len(calibrator.buckets) == 0

    def test_fit_basic(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test basic fitting of calibrator."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=10)

        result = calibrator.fit(confidences, correct)

        assert result is calibrator  # Returns self
        assert calibrator.is_fitted is True
        assert len(calibrator.buckets) == 10

    def test_fit_creates_valid_buckets(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test that fitting creates valid bucket boundaries."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=5)
        calibrator.fit(confidences, correct)

        for bucket in calibrator.buckets:
            assert 0 <= bucket.raw_min <= 1
            assert 0 <= bucket.raw_max <= 1
            assert bucket.raw_min <= bucket.raw_max
            assert 0 <= bucket.calibrated <= 1

    def test_calibrate_unfitted_returns_raw(self):
        """Test that unfitted calibrator returns raw confidence."""
        calibrator = ConfidenceCalibrator()

        assert calibrator.calibrate(0.5) == 0.5
        assert calibrator.calibrate(0.8) == 0.8

    def test_calibrate_fitted(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test calibration of fitted model."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=10)
        calibrator.fit(confidences, correct)

        # Test calibration returns valid values
        result = calibrator.calibrate(0.5)
        assert 0 <= result <= 1

        # Test edge cases
        assert 0 <= calibrator.calibrate(0.0) <= 1
        assert 0 <= calibrator.calibrate(1.0) <= 1

    def test_calibrate_clamps_input(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test that calibrator clamps out-of-range inputs."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=10)
        calibrator.fit(confidences, correct)

        # Values outside [0, 1] should be clamped
        result_low = calibrator.calibrate(-0.5)
        result_high = calibrator.calibrate(1.5)

        assert 0 <= result_low <= 1
        assert 0 <= result_high <= 1

    def test_calibrate_batch(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test batch calibration."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=10)
        calibrator.fit(confidences, correct)

        test_confidences = np.array([0.2, 0.5, 0.8])
        results = calibrator.calibrate_batch(test_confidences)

        assert len(results) == 3
        assert all(0 <= r <= 1 for r in results)

    def test_isotonic_regression_monotonic(self):
        """Test that isotonic regression produces monotonic output."""
        calibrator = ConfidenceCalibrator()

        values = [0.3, 0.5, 0.2, 0.7, 0.6]  # Non-monotonic
        result = calibrator._isotonic_regression(values)

        # Check monotonicity
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], "Output should be monotonically increasing"

    def test_save_and_load(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test saving and loading calibrator."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=10)
        calibrator.fit(confidences, correct)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            calibrator.save(str(path))

            # Load and verify
            loaded = ConfidenceCalibrator.load(str(path))

            assert loaded.num_buckets == calibrator.num_buckets
            assert loaded.is_fitted == calibrator.is_fitted
            assert len(loaded.buckets) == len(calibrator.buckets)

            # Check calibration produces same results
            for conf in [0.2, 0.5, 0.8]:
                assert calibrator.calibrate(conf) == loaded.calibrate(conf)

    def test_save_creates_directory(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test that save creates parent directory if needed."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator()
        calibrator.fit(confidences, correct)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "calibration.json"
            calibrator.save(str(path))

            assert path.exists()

    def test_get_calibration_table_unfitted(self):
        """Test calibration table for unfitted calibrator."""
        calibrator = ConfidenceCalibrator()
        table = calibrator.get_calibration_table()

        assert "not fitted" in table.lower()

    def test_get_calibration_table_fitted(
        self, sample_confidences: tuple[np.ndarray, np.ndarray]
    ):
        """Test calibration table for fitted calibrator."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=5)
        calibrator.fit(confidences, correct)

        table = calibrator.get_calibration_table()

        assert "Calibration" in table
        assert "Raw Range" in table
        assert "Calibrated" in table
        assert "Samples" in table

    def test_fit_empty_data_raises(self):
        """Test that fitting with empty data raises error."""
        calibrator = ConfidenceCalibrator()

        with pytest.raises(ValueError, match="empty"):
            calibrator.fit(np.array([]), np.array([]))

    def test_fit_mismatched_length_raises(self):
        """Test that mismatched array lengths raise error."""
        calibrator = ConfidenceCalibrator()

        with pytest.raises(ValueError, match="same length"):
            calibrator.fit(np.array([0.5, 0.6]), np.array([True]))

    def test_enforce_monotonic_false(self, sample_confidences: tuple[np.ndarray, np.ndarray]):
        """Test fitting without monotonicity enforcement."""
        confidences, correct = sample_confidences
        calibrator = ConfidenceCalibrator(num_buckets=10)
        calibrator.fit(confidences, correct, enforce_monotonic=False)

        assert calibrator.is_fitted
        # Buckets may not be monotonic without enforcement


class TestCalibrationBucket:
    """Tests for the CalibrationBucket dataclass."""

    def test_bucket_creation(self):
        """Test that bucket can be created."""
        bucket = CalibrationBucket(
            raw_min=0.0,
            raw_max=0.1,
            calibrated=0.05,
            sample_count=50,
        )

        assert bucket.raw_min == 0.0
        assert bucket.raw_max == 0.1
        assert bucket.calibrated == 0.05
        assert bucket.sample_count == 50
