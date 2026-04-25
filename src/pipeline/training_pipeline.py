"""
Training pipeline orchestrator.

Coordinates the full experiment workflow:
    Data loading → Model building → Training → Evaluation → Results saving
"""

from src.config import Config
from src.data import MNISTDataset
from src.model import NeuralNetwork
from src.training import Trainer
from src.evaluation import Evaluator
from src.utils import ResultsSaver


class TrainingPipeline:
    """
    Orchestrates a single end-to-end experiment run.

    Usage:
        pipeline = TrainingPipeline()
        pipeline.run(config)
    """

    def run(self, config: Config) -> dict:
        """
        Execute the full pipeline for a given configuration.

        Steps:
            1. Load and preprocess data   (MNISTDataset)
            2. Build and compile model     (NeuralNetwork)
            3. Train the model             (Trainer)
            4. Evaluate on test set        (Evaluator)
            5. Save results to disk        (ResultsSaver)

        Args:
            config: Experiment configuration.

        Returns:
            Dictionary of final test metrics.
        """
        print("\n" + "=" * 60)
        print(f"  PIPELINE START -- {config.EXPERIMENT_NAME}")
        print("=" * 60)
        print(config)

        # ── Step 1: Load data ─────────────────────────────────────
        dataset = MNISTDataset(config)
        dataset.load()

        # ── Step 2: Build model ───────────────────────────────────
        network = NeuralNetwork(config)
        model = network.build()

        # ── Step 3: Train ─────────────────────────────────────────
        trainer = Trainer(model, dataset, config)
        history = trainer.train()

        # ── Step 4: Evaluate ──────────────────────────────────────
        evaluator = Evaluator(model, dataset, config)
        metrics = evaluator.evaluate()

        # ── Step 5: Save results ──────────────────────────────────
        ResultsSaver.save(config, metrics, history, model)

        print(f"\n{'=' * 60}")
        print(f"  PIPELINE COMPLETE -- {config.EXPERIMENT_NAME}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"{'=' * 60}\n")

        return metrics
