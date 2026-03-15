"""LSTM model definition, training pipelines, and MLflow tracking.

This module owns everything from a normalized feature matrix to trained model
weights and the config file that bridges training to inference.

Public API:
  SignalModel        — two-output LSTM: signal classification (BUY/HOLD/SELL)
                       and price change regression
  ModelTrainer       — standard train/val/test pipeline; writes weights and
                       signal_model_config.json to checkpoints/
  WalkForwardTrainer — sliding-window retraining with purge+embargo gaps;
                       each window is a nested MLflow child run
  sparse_focal_loss  — focal loss for sparse integer labels (default)
  balanced_focal_loss — focal loss for one-hot encoded labels

Key contracts:
  - Class encoding is fixed: BUY=0, HOLD=1, SELL=2. Never change this without
    updating SignalGenerator and Backtester.
  - Data is never shuffled — temporal order is always preserved.
  - Normalization statistics (mean, std) are computed on the training split
    only and saved to the config for use at inference time.
  - Focal loss and sample weights are mutually exclusive (see INVARIANTS.md).
  - Adding features requires retraining — input_dim is fixed at training time.

See ARCHITECTURE.md §3-5 for the config format and sequence creation details.
"""

from models.losses import balanced_focal_loss, sparse_focal_loss
from models.signal_model import SignalModel
from models.training import ModelTrainer
from models.walk_forward import WalkForwardTrainer

__all__ = [
    "SignalModel",
    "ModelTrainer",
    "WalkForwardTrainer",
    "sparse_focal_loss",
    "balanced_focal_loss",
]
