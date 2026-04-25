"""
Utility for saving experiment results to disk.

Persists config, metrics, training history, and the trained model
under the experiment's log directory.
"""

import json
import os
import shutil

import tensorflow as tf

from src.config import Config


class ResultsSaver:
    """
    Saves all experiment artifacts to logs/<experiment_name>/.

    Files created:
        - config.json          -- full experiment configuration
        - metrics.json         -- final test metrics + training history
        - model.keras          -- trained Keras model
    """

    @staticmethod
    def save(
        config: Config,
        metrics: dict,
        history,
        model: tf.keras.Model = None,
    ) -> None:
        """
        Save experiment results to disk.

        Args:
            config:  Experiment configuration.
            metrics: Final evaluation metrics dict.
            history: Keras History object from training.
            model:   Trained Keras model to save.
        """
        log_dir = os.path.join("logs", config.EXPERIMENT_NAME)
        os.makedirs(log_dir, exist_ok=True)

        # -- 1. Save config ------------------------------------------------
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=4)
        print(f"[ResultsSaver] Config saved -> {config_path}")

        # -- 2. Build full metrics payload ----------------------------------
        training_history = {}
        if history is not None:
            training_history = {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            }

        full_metrics = {
            "experiment_name": config.EXPERIMENT_NAME,
            "final_metrics": metrics,
            "training_history": training_history,
        }

        # -- 3. Save metrics ------------------------------------------------
        metrics_path = os.path.join(log_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=4)
        print(f"[ResultsSaver] Metrics saved -> {metrics_path}")

        # -- 4. Save model --------------------------------------------------
        if model is not None:
            model_path = os.path.join(log_dir, "model.keras")
            model.save(model_path)
            print(f"[ResultsSaver] Model saved -> {model_path}")

    @staticmethod
    def save_best_model(best_experiment_name: str) -> None:
        """
        Copy the best experiment's model to logs/best_model/ for easy loading.

        Args:
            best_experiment_name: Name of the experiment with the best accuracy.
        """
        src_dir = os.path.join("logs", best_experiment_name)
        dst_dir = os.path.join("logs", "best_model")
        os.makedirs(dst_dir, exist_ok=True)

        # Copy model
        src_model = os.path.join(src_dir, "model.keras")
        dst_model = os.path.join(dst_dir, "model.keras")
        if os.path.exists(src_model):
            shutil.copy2(src_model, dst_model)

        # Copy config
        src_config = os.path.join(src_dir, "config.json")
        dst_config = os.path.join(dst_dir, "config.json")
        if os.path.exists(src_config):
            shutil.copy2(src_config, dst_config)

        # Copy metrics
        src_metrics = os.path.join(src_dir, "metrics.json")
        dst_metrics = os.path.join(dst_dir, "metrics.json")
        if os.path.exists(src_metrics):
            shutil.copy2(src_metrics, dst_metrics)

        print(f"[ResultsSaver] Best model saved -> {dst_dir}")
        print(f"[ResultsSaver] Source: {best_experiment_name}")

