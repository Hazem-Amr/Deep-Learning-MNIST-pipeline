"""
Neural network model builder for MNIST classification.

Constructs a configurable Sequential Dense network with ReLU hidden layers
and softmax output, compiled with the Adam optimizer.
"""

import tensorflow as tf

from src.config import Config


class NeuralNetwork:
    """
    Builds and compiles a Keras Sequential model based on the provided Config.

    Architecture:
        Input(INPUT_SIZE)
        → [Dense(h, activation=ACTIVATION) for h in HIDDEN_LAYERS]
        → Dense(NUM_CLASSES, activation='softmax')
    """

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> tf.keras.Model:
        """
        Build and compile the model.

        Returns:
            Compiled tf.keras.Model ready for training.
        """
        print("[NeuralNetwork] Building model ...")

        layers = []

        # ── Input layer ───────────────────────────────────────────
        layers.append(
            tf.keras.layers.Input(shape=(self.config.INPUT_SIZE,))
        )

        # ── Hidden layers ─────────────────────────────────────────
        for units in self.config.HIDDEN_LAYERS:
            layers.append(
                tf.keras.layers.Dense(units, activation=self.config.ACTIVATION)
            )

        # ── Output layer ──────────────────────────────────────────
        layers.append(
            tf.keras.layers.Dense(self.config.NUM_CLASSES, activation="softmax")
        )

        # ── Assemble model ────────────────────────────────────────
        model = tf.keras.Sequential(layers)

        # ── Compile ───────────────────────────────────────────────
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()
        return model
