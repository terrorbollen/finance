"""Custom loss functions for trading signal model."""

import tensorflow as tf
from tensorflow import keras


def sparse_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss for multi-class classification with sparse labels.

    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing training on hard negatives.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values = more focus on hard examples.
               gamma=0 is equivalent to cross-entropy. Default: 2.0
        alpha: Class weight balancing factor. Default: 0.25

    Returns:
        Loss function compatible with Keras model.compile()

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def focal_loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())

        # Get the number of classes
        num_classes = tf.shape(y_pred)[-1]

        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)

        # Calculate focal weight: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - p_t, gamma)

        # Apply focal weight and alpha
        focal_loss = alpha * focal_weight * cross_entropy

        # Sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    # Set function name for Keras
    focal_loss_fn.__name__ = f'sparse_focal_loss_g{gamma}_a{alpha}'

    return focal_loss_fn


def balanced_focal_loss(gamma: float = 2.0, class_weights: list = None):
    """
    Focal Loss with per-class weights for severe imbalance.

    Args:
        gamma: Focusing parameter
        class_weights: List of weights for each class [BUY, HOLD, SELL]
                      If None, uses uniform weights

    Returns:
        Loss function compatible with Keras model.compile()
    """
    if class_weights is None:
        class_weights = [1.0, 1.0, 1.0]

    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def balanced_focal_loss_fn(y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())

        # Get number of classes
        num_classes = tf.shape(y_pred)[-1]

        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

        # Get class weight for each sample
        sample_weights = tf.reduce_sum(y_true_one_hot * class_weights, axis=-1)

        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)

        # Calculate focal weight
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - p_t, gamma)

        # Apply focal weight
        focal_loss = focal_weight * cross_entropy

        # Sum over classes
        focal_loss = tf.reduce_sum(focal_loss, axis=-1)

        # Apply class weights
        weighted_loss = focal_loss * sample_weights

        return tf.reduce_mean(weighted_loss)

    balanced_focal_loss_fn.__name__ = f'balanced_focal_loss_g{gamma}'

    return balanced_focal_loss_fn
