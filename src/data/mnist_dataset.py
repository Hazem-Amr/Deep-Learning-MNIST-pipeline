"""
MNIST dataset loader and preprocessor.

Handles loading the MNIST dataset from Keras, normalizing pixel values,
flattening images, and one-hot encoding labels.
"""

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from src.config import Config


class MNISTDataset:
    """
    Loads and preprocesses the MNIST dataset.

    Preprocessing steps:
        1. Load raw data via keras.datasets.mnist
        2. Normalize pixel values from [0, 255] to [0, 1]
        3. Flatten images from (28, 28) to (784,)
        4. One-hot encode labels to (N, 10)
    """

    def __init__(self, config: Config):
        self.config = config

        # Placeholders — populated by load()
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None

    def load(self):
        """
        Load MNIST data, preprocess, and store in instance attributes.

        Returns:
            self — for method chaining.
        """
        print("[MNISTDataset] Loading MNIST data ...")

        # ── 1. Load raw data ──────────────────────────────────────
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

        # ── 2. Normalize pixel values to [0, 1] ──────────────────
        x_train = x_train_raw.astype("float32") / 255.0
        x_test = x_test_raw.astype("float32") / 255.0

        # ── 3. Flatten images to (N, 784) ─────────────────────────
        x_train = x_train.reshape(-1, self.config.INPUT_SIZE)
        x_test = x_test.reshape(-1, self.config.INPUT_SIZE)

        # ── 4. One-hot encode labels ──────────────────────────────
        y_train = tf.keras.utils.to_categorical(
            y_train_raw, num_classes=self.config.NUM_CLASSES
        )
        y_test = tf.keras.utils.to_categorical(
            y_test_raw, num_classes=self.config.NUM_CLASSES
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        print(
            f"[MNISTDataset] Data loaded — "
            f"Train: {self.x_train.shape}, Test: {self.x_test.shape}"
        )

        return self
