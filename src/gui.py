"""
MNIST Digit Recognizer -- Tkinter GUI

Features:
    - Draw a digit on the canvas and get a prediction
    - Upload an image file for recognition
    - Loads the best saved model from logs/best_model/

Usage:
    1. First run the pipeline:  python -m src.main
    2. Then launch the GUI:     python -m src.gui
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
import tensorflow as tf


# -- Constants ---------------------------------------------------------
CANVAS_SIZE = 280          # Drawing canvas (pixels)
MNIST_SIZE = 28            # MNIST image dimension
MODEL_PATH = os.path.join("logs", "best_model", "model.keras")

BG_COLOR = "#1e1e2e"       # Dark background
CANVAS_BG = "#000000"      # Black canvas (like MNIST)
ACCENT = "#89b4fa"         # Blue accent
TEXT_COLOR = "#cdd6f4"      # Light text
RESULT_COLOR = "#a6e3a1"   # Green for results
BTN_BG = "#313244"
BTN_ACTIVE = "#45475a"
BRUSH_COLOR = "white"
BRUSH_SIZE = 14


class DigitRecognizerApp:
    """Tkinter application for handwritten digit recognition."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        # PIL image mirror of the canvas (for accurate resizing)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._build_ui()
        self._load_model()

    # -- Load saved model -----------------------------------------------
    def _load_model(self):
        """Load the best saved model from disk."""
        if not os.path.exists(MODEL_PATH):
            self.model = None
            self.status_label.config(
                text="ERROR: No model found. Run 'python -m src.main' first!"
            )
            messagebox.showerror(
                "Model Not Found",
                "No trained model found at:\n"
                f"  {MODEL_PATH}\n\n"
                "Please run the training pipeline first:\n"
                "  python -m src.main",
            )
            return

        self.status_label.config(text="Loading model ...")
        self.root.update()

        self.model = tf.keras.models.load_model(MODEL_PATH)

        self.status_label.config(
            text=f"Model loaded from {MODEL_PATH} -- ready!"
        )

    # -- UI Construction ------------------------------------------------
    def _build_ui(self):
        """Build all UI widgets."""

        # Title
        title = tk.Label(
            self.root,
            text="MNIST Digit Recognizer",
            font=("Segoe UI", 20, "bold"),
            fg=ACCENT,
            bg=BG_COLOR,
        )
        title.pack(pady=(18, 4))

        subtitle = tk.Label(
            self.root,
            text="Draw a digit or upload an image",
            font=("Segoe UI", 11),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
        )
        subtitle.pack(pady=(0, 12))

        # Canvas frame
        canvas_frame = tk.Frame(self.root, bg=ACCENT, padx=2, pady=2)
        canvas_frame.pack()

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack()

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._reset_last_pos)
        self.last_x = None
        self.last_y = None

        # Buttons row
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(pady=14)

        btn_style = dict(
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_COLOR,
            bg=BTN_BG,
            activeforeground=TEXT_COLOR,
            activebackground=BTN_ACTIVE,
            relief="flat",
            padx=18,
            pady=6,
            cursor="hand2",
        )

        self.predict_btn = tk.Button(
            btn_frame, text="Predict", command=self._predict, **btn_style
        )
        self.predict_btn.grid(row=0, column=0, padx=6)

        self.clear_btn = tk.Button(
            btn_frame, text="Clear", command=self._clear_canvas, **btn_style
        )
        self.clear_btn.grid(row=0, column=1, padx=6)

        self.upload_btn = tk.Button(
            btn_frame, text="Upload Image", command=self._upload_image, **btn_style
        )
        self.upload_btn.grid(row=0, column=2, padx=6)

        # Result display
        self.result_label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 28, "bold"),
            fg=RESULT_COLOR,
            bg=BG_COLOR,
        )
        self.result_label.pack(pady=(2, 4))

        self.confidence_label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 12),
            fg=TEXT_COLOR,
            bg=BG_COLOR,
        )
        self.confidence_label.pack(pady=(0, 4))

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Loading model ...",
            font=("Segoe UI", 10),
            fg="#6c7086",
            bg=BG_COLOR,
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=12, pady=(0, 10))

    # -- Drawing --------------------------------------------------------
    def _paint(self, event):
        """Draw on canvas and PIL mirror simultaneously."""
        x, y = event.x, event.y

        if self.last_x is not None and self.last_y is not None:
            # Draw on tkinter canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill=BRUSH_COLOR,
                width=BRUSH_SIZE,
                capstyle=tk.ROUND,
                smooth=True,
            )
            # Draw on PIL mirror
            self.pil_draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255,
                width=BRUSH_SIZE,
            )

        self.last_x = x
        self.last_y = y

    def _reset_last_pos(self, event):
        """Reset last drawing position on mouse release."""
        self.last_x = None
        self.last_y = None

    def _clear_canvas(self):
        """Clear the drawing canvas and result."""
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="")
        self.confidence_label.config(text="")

    # -- Prediction -----------------------------------------------------
    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for MNIST prediction.

        Steps:
            1. Convert to grayscale
            2. Invert if background is white
            3. Resize to 28x28 with anti-aliasing
            4. Normalize to [0, 1]
            5. Flatten to (1, 784)
        """
        img = img.convert("L")

        # Invert if background is white (MNIST has black background)
        pixel_mean = np.array(img).mean()
        if pixel_mean > 127:
            img = ImageOps.invert(img)

        img = img.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)

        img_array = np.array(img, dtype="float32") / 255.0
        img_array = img_array.reshape(1, MNIST_SIZE * MNIST_SIZE)

        return img_array

    def _predict(self):
        """Predict the digit drawn on the canvas."""
        if self.model is None:
            messagebox.showwarning(
                "No Model",
                "No model loaded. Run 'python -m src.main' first.",
            )
            return

        img_array = self._preprocess_image(self.pil_image)
        self._run_prediction(img_array)

    def _upload_image(self):
        """Open file dialog, load image, and predict."""
        if self.model is None:
            messagebox.showwarning(
                "No Model",
                "No model loaded. Run 'python -m src.main' first.",
            )
            return

        filepath = filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not filepath:
            return

        try:
            img = Image.open(filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return

        # Display the uploaded image on canvas
        self._clear_canvas()
        display_img = img.copy().convert("L")

        pixel_mean = np.array(display_img).mean()
        if pixel_mean > 127:
            display_img = ImageOps.invert(display_img)

        display_img = display_img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
        self.pil_image = display_img
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        # Show on canvas using PhotoImage
        self._photo = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # Predict
        img_array = self._preprocess_image(img)
        self._run_prediction(img_array)

    def _run_prediction(self, img_array: np.ndarray):
        """Run model prediction and display results."""
        prediction = self.model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_digit]) * 100

        self.result_label.config(text=f"Prediction: {predicted_digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        # Show top-3 predictions in status bar
        top3_idx = np.argsort(prediction[0])[::-1][:3]
        top3_str = "  |  ".join(
            f"{int(i)}: {prediction[0][i]*100:.1f}%"
            for i in top3_idx
        )
        self.status_label.config(text=f"Top 3:  {top3_str}")


def main():
    """Launch the digit recognizer GUI."""
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
