"""
Model trainer with TensorBoard integration.

Handles the training loop, validation monitoring,
and TensorBoard logging for experiment tracking.
"""

import os

import tensorflow as tf

from src.config import Config


class Trainer:
    """
    Trains a compiled Keras model using the provided data and config.

    Features:
        - Configurable epochs, batch size, and validation split
        - TensorBoard callback for live training visualization
        - Returns the full training history for downstream analysis
    """

    def __init__(self, model: tf.keras.Model, data, config: Config):
        """
        Args:
            model: Compiled Keras model.
            data:  MNISTDataset instance with loaded data.
            config: Experiment configuration.
        """
        self.model = model
        self.data = data
        self.config = config

    def train(self) -> tf.keras.callbacks.History:
        """
        Run the training loop.

        Returns:
            Keras History object containing per-epoch metrics.
        """
        print(f"[Trainer] Starting training -- {self.config.EXPERIMENT_NAME}")

        # ── TensorBoard callback ──────────────────────────────────
        log_dir = os.path.join("logs", self.config.EXPERIMENT_NAME)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,       # log weight histograms every epoch
            write_graph=True,
            update_freq="epoch",
        )

        # ── Train ─────────────────────────────────────────────────
        history = self.model.fit(
            self.data.x_train,
            self.data.y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            callbacks=[tensorboard_cb],
            verbose=1,
        )

        print(f"[Trainer] Training complete -- {self.config.EXPERIMENT_NAME}")
        return history
