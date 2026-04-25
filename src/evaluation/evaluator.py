"""
Model evaluator for MNIST classification.

Runs model.evaluate() on the test set and returns
structured metrics ready for JSON export.
"""

import tensorflow as tf

from src.config import Config


class Evaluator:
    """
    Evaluates a trained Keras model on the held-out test set.

    Returns a dictionary of final metrics:
        - test_loss
        - test_accuracy
    """

    def __init__(self, model: tf.keras.Model, data, config: Config):
        """
        Args:
            model:  Trained Keras model.
            data:   MNISTDataset instance with loaded test data.
            config: Experiment configuration.
        """
        self.model = model
        self.data = data
        self.config = config

    def evaluate(self) -> dict:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary with 'test_loss' and 'test_accuracy' keys.
        """
        print(f"[Evaluator] Evaluating -- {self.config.EXPERIMENT_NAME}")

        test_loss, test_accuracy = self.model.evaluate(
            self.data.x_test,
            self.data.y_test,
            verbose=1,
        )

        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        }

        print(
            f"[Evaluator] Results — "
            f"Loss: {metrics['test_loss']:.4f}, "
            f"Accuracy: {metrics['test_accuracy']:.4f}"
        )

        return metrics
