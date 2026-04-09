"""Tests for focal loss implementations in models/losses.py."""

import numpy as np
import tensorflow as tf

from models.losses import balanced_focal_loss, sparse_focal_loss


class TestSparseFocalLoss:
    """Tests for sparse_focal_loss."""

    def test_returns_callable(self):
        loss_fn = sparse_focal_loss()
        assert callable(loss_fn)

    def test_output_is_scalar(self):
        loss_fn = sparse_focal_loss()
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        loss_fn = sparse_focal_loss()
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        loss = loss_fn(y_true, y_pred)
        assert float(loss) > 0

    def test_perfect_predictions_give_near_zero_loss(self):
        """With near-perfect confidence, focal loss approaches zero."""
        loss_fn = sparse_focal_loss(gamma=2.0, alpha=1.0)
        eps = 1e-6
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        # Nearly perfect predictions
        y_pred = tf.constant([
            [1 - 2 * eps, eps, eps],
            [eps, 1 - 2 * eps, eps],
            [eps, eps, 1 - 2 * eps],
        ])
        loss = float(loss_fn(y_true, y_pred))
        assert loss < 0.01, f"Perfect predictions should yield near-zero loss, got {loss:.4f}"

    def test_wrong_predictions_give_higher_loss_than_correct(self):
        loss_fn = sparse_focal_loss(gamma=2.0, alpha=1.0)
        y_true = tf.constant([0], dtype=tf.int32)
        correct = tf.constant([[0.9, 0.05, 0.05]])
        wrong = tf.constant([[0.05, 0.05, 0.9]])
        loss_correct = float(loss_fn(y_true, correct))
        loss_wrong = float(loss_fn(y_true, wrong))
        assert loss_wrong > loss_correct

    def test_gamma_zero_approximates_cross_entropy(self):
        """gamma=0 with alpha=1 should match standard cross-entropy."""
        focal_fn = sparse_focal_loss(gamma=0.0, alpha=1.0)
        y_true = tf.constant([0, 1], dtype=tf.int32)
        y_pred = tf.constant([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        focal_loss = float(focal_fn(y_true, y_pred))
        ce_loss = float(
            tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            )
        )
        assert abs(focal_loss - ce_loss) < 0.01, (
            f"gamma=0, alpha=1 focal loss ({focal_loss:.4f}) should match "
            f"cross-entropy ({ce_loss:.4f})"
        )

    def test_higher_gamma_reduces_loss_on_easy_examples(self):
        """Higher gamma should down-weight well-classified examples more aggressively."""
        y_true = tf.constant([0], dtype=tf.int32)
        y_pred = tf.constant([[0.95, 0.03, 0.02]])  # easy example: very confident
        loss_low_gamma = float(sparse_focal_loss(gamma=0.5, alpha=1.0)(y_true, y_pred))
        loss_high_gamma = float(sparse_focal_loss(gamma=4.0, alpha=1.0)(y_true, y_pred))
        assert loss_high_gamma < loss_low_gamma, (
            "Higher gamma should reduce loss on easy (high-confidence correct) examples"
        )

    def test_loss_fn_name_encodes_params(self):
        fn = sparse_focal_loss(gamma=1.5, alpha=0.3)
        assert "1.5" in fn.__name__
        assert "0.3" in fn.__name__

    def test_batch_size_one(self):
        loss_fn = sparse_focal_loss()
        y_true = tf.constant([1], dtype=tf.int32)
        y_pred = tf.constant([[0.2, 0.6, 0.2]])
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == ()
        assert float(loss) > 0

    def test_uniform_predictions_produce_finite_loss(self):
        """1/3 uniform predictions should not produce NaN or inf."""
        loss_fn = sparse_focal_loss()
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[1 / 3, 1 / 3, 1 / 3]] * 3)
        loss = float(loss_fn(y_true, y_pred))
        assert np.isfinite(loss), f"Uniform predictions produced non-finite loss: {loss}"


class TestBalancedFocalLoss:
    """Tests for balanced_focal_loss (per-class weights)."""

    def test_returns_callable(self):
        loss_fn = balanced_focal_loss()
        assert callable(loss_fn)

    def test_output_is_scalar(self):
        loss_fn = balanced_focal_loss()
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == ()

    def test_uniform_weights_match_unweighted(self):
        """Uniform class weights should give the same loss regardless of label."""
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        loss_uniform = float(balanced_focal_loss(class_weights=[1.0, 1.0, 1.0])(y_true, y_pred))
        loss_none = float(balanced_focal_loss(class_weights=None)(y_true, y_pred))
        assert abs(loss_uniform - loss_none) < 1e-5

    def test_higher_weight_increases_loss_for_that_class(self):
        """Assigning more weight to class 0 should raise its contribution to total loss."""
        y_true = tf.constant([0], dtype=tf.int32)
        y_pred = tf.constant([[0.5, 0.3, 0.2]])
        loss_low = float(balanced_focal_loss(class_weights=[1.0, 1.0, 1.0])(y_true, y_pred))
        loss_high = float(balanced_focal_loss(class_weights=[5.0, 1.0, 1.0])(y_true, y_pred))
        assert loss_high > loss_low, "Higher weight for predicted class should increase loss"

    def test_zero_weight_class_contributes_zero(self):
        """A class with weight 0 should not contribute to the loss."""
        y_true = tf.constant([1], dtype=tf.int32)  # HOLD
        y_pred = tf.constant([[0.1, 0.8, 0.1]])
        loss_zeroed = float(balanced_focal_loss(class_weights=[1.0, 0.0, 1.0])(y_true, y_pred))
        # HOLD sample with weight 0 → loss should be ~0
        assert loss_zeroed < 1e-5, f"Zero-weight class should give ~0 loss, got {loss_zeroed}"

    def test_class_weights_and_sparse_focal_are_mutually_exclusive_by_design(self):
        """
        Regression for INVARIANTS.md: focal loss and sample_weight must not both be active.
        This test just confirms balanced_focal_loss accepts weights without raising.
        The training pipeline is responsible for not passing sample_weight alongside it.
        """
        loss_fn = balanced_focal_loss(gamma=2.0, class_weights=[0.5, 1.5, 1.0])
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
        loss = float(loss_fn(y_true, y_pred))
        assert np.isfinite(loss)

    def test_fn_name_encodes_gamma(self):
        fn = balanced_focal_loss(gamma=3.0)
        assert "3.0" in fn.__name__

    def test_loss_is_positive(self):
        loss_fn = balanced_focal_loss(class_weights=[1.0, 2.0, 1.0])
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        assert float(loss_fn(y_true, y_pred)) > 0
