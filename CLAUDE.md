# CLAUDE.md

Project context for AI assistants working on this codebase.

## Project Overview

Experiment-based MNIST digit classification pipeline built with TensorFlow/Keras. Supports multiple experiment runs with different hyperparameters, TensorBoard integration, and a Tkinter GUI for digit recognition.

## Architecture

```
Main.start() --> TrainingPipeline.run(config)
                    ├── MNISTDataset.load()
                    ├── NeuralNetwork.build()
                    ├── Trainer.train()
                    ├── Evaluator.evaluate()
                    └── ResultsSaver.save()
```

## Project Structure

```
src/
├── main.py                  # Entry point -- defines experiment configs, runs pipeline
├── gui.py                   # Tkinter GUI -- draws digits, uploads images, predicts
├── config/config.py         # Config dataclass (hyperparameters + experiment metadata)
├── data/mnist_dataset.py    # MNIST loading, normalization, flattening, one-hot encoding
├── model/neural_network.py  # Configurable Dense Sequential model builder
├── training/trainer.py      # Training loop with TensorBoard callbacks
├── evaluation/evaluator.py  # Test set evaluation, returns metrics dict
├── pipeline/training_pipeline.py  # Orchestrates: data -> model -> train -> eval -> save
└── utils/
    ├── results_saver.py     # Saves config.json, metrics.json, model.keras per experiment
    └── preprocessor.py      # MNIST-aligned preprocessing (bbox, crop, center-of-mass)
```

## GUI Preprocessing Pipeline

The `preprocessor.py` module aligns drawn/uploaded images with MNIST distribution:

1. Convert to grayscale
2. Invert colors if background is light (corner sampling, not global mean)
3. Detect bounding box of digit pixels (threshold > 30)
4. Crop to bounding box
5. Fit into 20x20 box preserving aspect ratio (MNIST convention)
6. Center in 28x28 canvas by center-of-mass (matching MNIST centering)
7. Normalize to [0, 1], flatten to (1, 784)

## Commands

```bash
# Run all experiments (trains models, saves best to logs/best_model/)
python -m src.main

# Launch GUI digit recognizer (loads saved best model)
python -m src.gui

# View training curves in TensorBoard
tensorboard --logdir=logs
```

## Key Conventions

- **Module imports** use `from src.config import Config` style (run from project root)
- **Each package** has an `__init__.py` that re-exports its main class
- **Config is a dataclass** -- all hyperparameters flow through it
- **Experiment outputs** go to `logs/<experiment_name>/` (config.json, metrics.json, model.keras)
- **Best model** is auto-copied to `logs/best_model/` after all experiments complete
- **Hidden layers use ReLU**, output uses softmax, optimizer is Adam
- **Input shape is (784,)** -- flattened 28x28, Dense MLP (not CNN)
- **Print statements must use ASCII only** -- Windows cp1252 breaks on Unicode arrows/dashes

## Dependencies

- Python 3.12
- TensorFlow/Keras
- TensorBoard
- NumPy
- Pillow (for GUI)
- scikit-learn (used in original notebook, not in pipeline)

## Output Structure

```
logs/
├── best_model/              # Best experiment (auto-selected by accuracy)
│   ├── model.keras
│   ├── config.json
│   └── metrics.json
├── <experiment_name>/       # Per-experiment outputs
│   ├── model.keras
│   ├── config.json
│   ├── metrics.json
│   ├── train/               # TensorBoard events
│   └── validation/          # TensorBoard events
```

## Adding New Experiments

Edit `src/main.py` and add a `Config` to the `configs` list:

```python
Config(
    LEARNING_RATE=0.01,
    BATCH_SIZE=32,
    HIDDEN_LAYERS=[1024, 512, 256],
    EPOCHS=15,
    EXPERIMENT_NAME="my_deep_net_run1",
)
```

Configurable fields: `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `INPUT_SIZE`, `NUM_CLASSES`, `HIDDEN_LAYERS`, `ACTIVATION`, `VALIDATION_SPLIT`, `EXPERIMENT_NAME`.
