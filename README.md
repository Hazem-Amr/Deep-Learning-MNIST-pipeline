# Deep Learning MNIST Pipeline

A complete machine learning pipeline for training, evaluating, and deploying deep learning models on the MNIST handwritten digit dataset. 

This project features configurable experiment tracking, model persistence, TensorBoard integration, and a Tkinter-based graphical user interface (GUI) for real-time digit recognition.

## Features

- **Configurable Training Pipeline**: Easily define multiple experiments with varying hyperparameters (learning rate, batch size, hidden layers, epochs).
- **Experiment Tracking**: Logs training metrics and losses for visualization in TensorBoard.
- **Model Persistence**: Automatically saves the best-performing model after running all experiments.
- **Interactive GUI**: A Tkinter application that loads the best saved model and allows users to:
  - Draw a digit directly on a canvas.
  - Upload an image file for prediction.
  - View real-time predictions and top-3 confidence scores.

## Project Structure

```text
├── logs/               # TensorBoard logs and saved models
│   └── best_model/     # Directory where the best performing model is saved
├── src/                # Source code directory
│   ├── config/         # Configuration classes for experiments
│   ├── data/           # Data loading and preprocessing components
│   ├── evaluation/     # Model evaluation and metrics
│   ├── model/          # Neural network architecture definitions
│   ├── pipeline/       # Training pipeline coordination
│   ├── training/       # Training loop and logic
│   ├── utils/          # Utilities for saving models and results
│   ├── gui.py          # Tkinter GUI application entry point
│   └── main.py         # Main entry point to run experiments
└── README.md           # Project documentation
```

## Setup & Requirements

Ensure you have Python installed. Install the required dependencies using `pip`:

```bash
pip install tensorflow numpy pillow
```

## Usage

### 1. Run the Training Pipeline

To train the models and run all defined experiments, execute the main script from the root directory:

```bash
python -m src.main
```
This will train several configurations, log the results, and save the best model to `logs/best_model/model.keras`.

### 2. Visualize with TensorBoard

You can visualize the training curves (loss and accuracy) using TensorBoard:

```bash
tensorboard --logdir=logs
```
Open the provided local URL (typically `http://localhost:6006/`) in your browser.

### 3. Launch the GUI

Once the model has been trained and saved, you can launch the interactive digit recognizer:

```bash
python -m src.gui
```
- **Draw**: Use your mouse to draw a digit in the black canvas and click "Predict".
- **Upload**: Click "Upload Image" to select a local image file.
- **Clear**: Clears the canvas to draw another digit.
