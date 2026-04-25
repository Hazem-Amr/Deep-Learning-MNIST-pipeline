"""
MNIST Experiment Pipeline — Entry Point

Defines multiple experiment configurations and runs them
sequentially through the TrainingPipeline.

Usage:
    python -m src.main

TensorBoard:
    tensorboard --logdir=logs
"""

from src.config import Config
from src.pipeline import TrainingPipeline
from src.utils import ResultsSaver


def start():
    """
    Define experiment configs and execute all runs.

    Each Config can vary:
        - learning rate
        - batch size
        - hidden layer architecture
        - activation function
        - number of epochs
    """

    # -- Define experiment configurations ----------------------------------
    configs = [
        # Experiment 1: Baseline -- ReLU, lr=0.001, batch=64
        Config(
            LEARNING_RATE=0.001,
            BATCH_SIZE=64,
            HIDDEN_LAYERS=[512, 256],
            EPOCHS=10,
            EXPERIMENT_NAME="mnist_lr_0.001_bs_64_run1",
        ),

        # Experiment 2: Lower LR, larger batch
        Config(
            LEARNING_RATE=0.0001,
            BATCH_SIZE=128,
            HIDDEN_LAYERS=[512, 256],
            EPOCHS=10,
            EXPERIMENT_NAME="mnist_lr_0.0001_bs_128_run1",
        ),

        # Experiment 3: Smaller architecture
        Config(
            LEARNING_RATE=0.001,
            BATCH_SIZE=64,
            HIDDEN_LAYERS=[256, 128],
            EPOCHS=10,
            EXPERIMENT_NAME="mnist_small_arch_run1",
        ),
    ]

    # -- Run all experiments ------------------------------------------------
    pipeline = TrainingPipeline()
    all_results = {}

    for i, config in enumerate(configs, start=1):
        print(f"\n{'#' * 60}")
        print(f"  EXPERIMENT {i} / {len(configs)}")
        print(f"{'#' * 60}")

        metrics = pipeline.run(config)
        all_results[config.EXPERIMENT_NAME] = metrics

    # -- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE -- SUMMARY")
    print("=" * 60)
    for name, metrics in all_results.items():
        print(
            f"  {name:40s} | "
            f"Accuracy: {metrics['test_accuracy']:.4f} | "
            f"Loss: {metrics['test_loss']:.4f}"
        )
    print("=" * 60)

    # -- Save best model ----------------------------------------------------
    best_name = max(all_results, key=lambda n: all_results[n]["test_accuracy"])
    best_acc = all_results[best_name]["test_accuracy"]
    print(f"\n  Best experiment: {best_name} (Accuracy: {best_acc:.4f})")
    ResultsSaver.save_best_model(best_name)

    print(
        "\n  To visualize results in TensorBoard, run:\n"
        "    tensorboard --logdir=logs\n"
        "\n  To launch the GUI digit recognizer, run:\n"
        "    python -m src.gui\n"
    )


if __name__ == "__main__":
    start()

