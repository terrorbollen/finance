"""Tests for WalkForwardTrainer purge/embargo gap logic."""

from models.walk_forward import WalkForwardTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trainer(
    initial_train_days: int = 100,
    validation_days: int = 40,
    purge_gap: int = 0,
    embargo_gap: int = 0,
) -> WalkForwardTrainer:
    """Return a WalkForwardTrainer configured purely for window-generation tests."""
    return WalkForwardTrainer(
        initial_train_days=initial_train_days,
        validation_days=validation_days,
        sequence_length=20,
        prediction_horizons=[5],
        purge_gap=purge_gap,
        embargo_gap=embargo_gap,
    )


# ---------------------------------------------------------------------------
# Zero-gap behaviour matches the original (pre-purge) behaviour
# ---------------------------------------------------------------------------


class TestZeroGapBackwardCompatibility:
    """With purge_gap=0 and embargo_gap=0 the windows should be identical
    to what the original code produced (val_start == train_end)."""

    def test_val_start_equals_train_end_when_no_gap(self) -> None:
        t = _trainer(purge_gap=0, embargo_gap=0)
        windows = t.generate_windows(n_samples=300)
        assert windows, "Expected at least one window"
        for _train_start, train_end, val_start, _val_end in windows:
            assert val_start == train_end, (
                f"With zero purge_gap, val_start ({val_start}) should equal train_end ({train_end})"
            )

    def test_windows_are_contiguous_when_no_gap(self) -> None:
        t = _trainer(purge_gap=0, embargo_gap=0)
        windows = t.generate_windows(n_samples=300)
        for i in range(1, len(windows)):
            prev_val_end = windows[i - 1][3]
            cur_train_end = windows[i][1]
            assert cur_train_end == prev_val_end, (
                f"Window {i} train_end ({cur_train_end}) should equal "
                f"previous val_end ({prev_val_end}) with zero embargo"
            )

    def test_val_size_equals_validation_days_when_no_gap(self) -> None:
        validation_days = 40
        t = _trainer(validation_days=validation_days, purge_gap=0, embargo_gap=0)
        windows = t.generate_windows(n_samples=300)
        for _train_start, _train_end, val_start, val_end in windows:
            assert val_end - val_start == validation_days


# ---------------------------------------------------------------------------
# Purge gap: val_start is shifted forward
# ---------------------------------------------------------------------------


class TestPurgeGap:
    def test_purge_gap_shifts_val_start(self) -> None:
        purge = 7
        t_no_purge = _trainer(purge_gap=0, embargo_gap=0)
        t_purge = _trainer(purge_gap=purge, embargo_gap=0)

        n = 400
        windows_no_purge = t_no_purge.generate_windows(n)
        windows_purge = t_purge.generate_windows(n)

        # First window: train_end is the same; val_start should differ by purge
        assert windows_no_purge, "Need at least one window"
        assert windows_purge, "Need at least one window"

        _, train_end_np, val_start_np, _ = windows_no_purge[0]
        _, train_end_p, val_start_p, _ = windows_purge[0]

        assert train_end_np == train_end_p  # same training boundary
        assert val_start_p == val_start_np + purge, (
            f"val_start with purge ({val_start_p}) should be "
            f"val_start without purge ({val_start_np}) + purge ({purge})"
        )

    def test_purge_gap_reduces_val_size_by_correct_amount(self) -> None:
        """With purge_gap=P, each validation window shifts P bars into the
        original window, so val_end - val_start still equals validation_days
        but the overall covered range is larger by P bars."""
        validation_days = 40
        purge = 5
        t = _trainer(validation_days=validation_days, purge_gap=purge, embargo_gap=0)
        windows = t.generate_windows(n_samples=400)
        assert windows
        for _train_start, train_end, val_start, val_end in windows:
            # The gap is between train_end and val_start
            assert val_start - train_end == purge, (
                f"Gap between train_end ({train_end}) and val_start ({val_start}) should be {purge}"
            )
            # The validation window itself still has validation_days rows
            assert val_end - val_start == validation_days

    def test_purge_gap_no_overlap(self) -> None:
        """No row should appear in both the training window and the
        validation window (after applying the purge gap)."""
        purge = 5
        t = _trainer(purge_gap=purge, embargo_gap=0)
        windows = t.generate_windows(n_samples=400)
        for train_start, train_end, val_start, val_end in windows:
            train_indices = set(range(train_start, train_end))
            val_indices = set(range(val_start, val_end))
            overlap = train_indices & val_indices
            assert not overlap, f"Unexpected overlap between train and val: {overlap}"

    def test_default_purge_gap_equals_max_prediction_horizon(self) -> None:
        # purge_gap defaults to max(prediction_horizons) to cover the longest label look-ahead
        t = WalkForwardTrainer(prediction_horizons=[5, 10, 20])
        assert t.purge_gap == 20

    def test_explicit_purge_gap_overrides_default(self) -> None:
        t = WalkForwardTrainer(prediction_horizons=[5, 10, 20], purge_gap=10)
        assert t.purge_gap == 10


# ---------------------------------------------------------------------------
# Embargo gap: train_start advances on subsequent windows
# ---------------------------------------------------------------------------


class TestEmbargoGap:
    def test_default_embargo_gap_equals_sequence_length(self) -> None:
        sequence_length = 20
        t = WalkForwardTrainer(sequence_length=sequence_length)
        assert t.embargo_gap == sequence_length

    def test_explicit_embargo_gap_overrides_default(self) -> None:
        t = WalkForwardTrainer(sequence_length=20, embargo_gap=5)
        assert t.embargo_gap == 5

    def test_embargo_gap_advances_train_start(self) -> None:
        """With embargo_gap > 0, the train_start of windows beyond the first
        should be strictly greater than 0 (i.e., the embargo is applied)."""
        embargo = 10
        t = _trainer(initial_train_days=100, validation_days=40, purge_gap=0, embargo_gap=embargo)
        windows = t.generate_windows(n_samples=500)
        assert len(windows) >= 2, "Need at least two windows to test embargo"
        # The second window's train_start should have advanced
        first_train_start = windows[0][0]
        second_train_start = windows[1][0]
        assert second_train_start >= first_train_start + embargo, (
            f"Second window train_start ({second_train_start}) should be at least "
            f"first ({first_train_start}) + embargo ({embargo})"
        )

    def test_zero_embargo_does_not_advance_train_start(self) -> None:
        t = _trainer(initial_train_days=100, validation_days=40, purge_gap=0, embargo_gap=0)
        windows = t.generate_windows(n_samples=400)
        assert len(windows) >= 2, "Need at least two windows"
        # With zero embargo, train_start should stay at 0 (expanding window)
        for win in windows:
            assert win[0] == 0, "train_start should remain 0 with zero embargo_gap"

    def test_embargo_gap_does_not_overlap_previous_validation(self) -> None:
        """Rows at the start of the new training window that fall within the
        embargo period should not overlap with the previous validation window."""
        embargo = 15
        t = _trainer(initial_train_days=100, validation_days=40, purge_gap=5, embargo_gap=embargo)
        windows = t.generate_windows(n_samples=600)
        assert len(windows) >= 2
        for i in range(1, len(windows)):
            prev_val_start = windows[i - 1][2]
            cur_train_start = windows[i][0]
            # The embargo means the new training window starts at least
            # embargo bars after the start of the previous val window.
            assert cur_train_start >= prev_val_start, (
                f"Window {i} train_start ({cur_train_start}) should be >= "
                f"previous val_start ({prev_val_start}) with embargo={embargo}"
            )


# ---------------------------------------------------------------------------
# Combined purge + embargo
# ---------------------------------------------------------------------------


class TestCombinedGaps:
    def test_combined_gaps_produce_valid_windows(self) -> None:
        t = _trainer(
            initial_train_days=150,
            validation_days=50,
            purge_gap=5,
            embargo_gap=20,
        )
        windows = t.generate_windows(n_samples=700)
        assert windows, "Should produce at least one window"
        for train_start, train_end, val_start, val_end in windows:
            assert train_start >= 0
            assert train_end > train_start
            assert val_start > train_end  # purge gap enforced
            assert val_end > val_start
            assert val_end <= 700

    def test_combined_gaps_no_train_val_overlap(self) -> None:
        t = _trainer(
            initial_train_days=150,
            validation_days=50,
            purge_gap=5,
            embargo_gap=20,
        )
        windows = t.generate_windows(n_samples=700)
        for train_start, train_end, val_start, val_end in windows:
            train_set = set(range(train_start, train_end))
            val_set = set(range(val_start, val_end))
            assert not (train_set & val_set)

    def test_windows_do_not_exceed_n_samples(self) -> None:
        n = 500
        t = _trainer(purge_gap=5, embargo_gap=20)
        windows = t.generate_windows(n_samples=n)
        for _, _, _, val_end in windows:
            assert val_end <= n
