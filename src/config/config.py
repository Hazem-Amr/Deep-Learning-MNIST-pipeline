"""
Configuration module for MNIST experiment pipeline.

Holds all hyperparameters and experiment metadata as a dataclass.
Supports auto-generated experiment names and JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    """
    Experiment configuration containing all hyperparameters
    and metadata for a single training run.

    Attributes:
        EPOCHS:           Number of training epochs.
        BATCH_SIZE:       Mini-batch size for training.
        LEARNING_RATE:    Learning rate for the Adam optimizer.
        INPUT_SIZE:       Flattened input dimension (28x28 = 784).
        NUM_CLASSES:      Number of output classes (digits 0-9).
        HIDDEN_LAYERS:    List of hidden layer sizes.
        ACTIVATION:       Activation function for hidden layers.
        VALIDATION_SPLIT: Fraction of training data used for validation.
        EXPERIMENT_NAME:  Unique name for this experiment run.
    """

    # ── Training hyperparameters ──────────────────────────────────
    EPOCHS: int = 10
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001

    # ── Model architecture ────────────────────────────────────────
    INPUT_SIZE: int = 784
    NUM_CLASSES: int = 10
    HIDDEN_LAYERS: List[int] = field(default_factory=lambda: [512, 256])
    ACTIVATION: str = "relu"

    # ── Data split ────────────────────────────────────────────────
    VALIDATION_SPLIT: float = 0.2

    # ── Experiment metadata ───────────────────────────────────────
    EXPERIMENT_NAME: Optional[str] = None

    def __post_init__(self):
        """Auto-generate experiment name if not provided."""
        if self.EXPERIMENT_NAME is None:
            layers_str = "_".join(str(h) for h in self.HIDDEN_LAYERS)
            self.EXPERIMENT_NAME = (
                f"mnist_lr_{self.LEARNING_RATE}_bs_{self.BATCH_SIZE}"
                f"_layers_{layers_str}"
            )

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary for JSON export."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable summary of the config."""
        lines = [
            "=" * 50,
            "  Experiment Configuration",
            "=" * 50,
            f"  Experiment Name  : {self.EXPERIMENT_NAME}",
            f"  Epochs           : {self.EPOCHS}",
            f"  Batch Size       : {self.BATCH_SIZE}",
            f"  Learning Rate    : {self.LEARNING_RATE}",
            f"  Input Size       : {self.INPUT_SIZE}",
            f"  Num Classes      : {self.NUM_CLASSES}",
            f"  Hidden Layers    : {self.HIDDEN_LAYERS}",
            f"  Activation       : {self.ACTIVATION}",
            f"  Validation Split : {self.VALIDATION_SPLIT}",
            "=" * 50,
        ]
        return "\n".join(lines)
