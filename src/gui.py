"""
MNIST Digit Recognizer -- Tkinter GUI

Features:
    - Draw a digit on the canvas and get a prediction
    - Upload an image file for recognition
    - Loads the best saved model from logs/best_model/
    - Advanced MNIST-aligned preprocessing (bbox, crop, center-of-mass)
    - Preview panel showing what the model actually sees

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

from src.utils.preprocessor import preprocess_drawn_image, get_debug_images


# -- Constants ---------------------------------------------------------
CANVAS_SIZE = 280          # Drawing canvas (pixels)
PREVIEW_SIZE = 112         # Preview display (28x28 upscaled 4x)
MNIST_SIZE = 28            # MNIST image dimension
MODEL_PATH = os.path.join("logs", "best_model", "model.keras")

BG_COLOR = "#1e1e2e"       # Dark background
CANVAS_BG = "#000000"      # Black canvas (like MNIST)
ACCENT = "#89b4fa"         # Blue accent
TEXT_COLOR = "#cdd6f4"      # Light text
RESULT_COLOR = "#a6e3a1"   # Green for results
DIM_TEXT = "#6c7086"        # Dimmed text
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

        # -- Main area: Drawing canvas + Preview panel ------------------
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(padx=16)

        # Drawing canvas (left side)
        canvas_frame = tk.Frame(main_frame, bg=ACCENT, padx=2, pady=2)
        canvas_frame.grid(row=0, column=0)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack()

        # Preview panel (right side) -- shows what the model sees
        preview_outer = tk.Frame(main_frame, bg=BG_COLOR)
        preview_outer.grid(row=0, column=1, padx=(16, 0), sticky="n")

        preview_title = tk.Label(
            preview_outer,
            text="Model Input",
            font=("Segoe UI", 10, "bold"),
            fg=DIM_TEXT,
            bg=BG_COLOR,
        )
        preview_title.pack(pady=(0, 4))

        preview_border = tk.Frame(preview_outer, bg=ACCENT, padx=2, pady=2)
        preview_border.pack()

        self.preview_canvas = tk.Canvas(
            preview_border,
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE,
            bg=CANVAS_BG,
            highlightthickness=0,
        )
        self.preview_canvas.pack()

        self.preview_info = tk.Label(
            preview_outer,
            text="28 x 28 px",
            font=("Segoe UI", 9),
            fg=DIM_TEXT,
            bg=BG_COLOR,
        )
        self.preview_info.pack(pady=(4, 0))

        # Preprocessing stages labels
        self.stage_label = tk.Label(
            preview_outer,
            text="",
            font=("Segoe UI", 9),
            fg=DIM_TEXT,
            bg=BG_COLOR,
            justify="left",
            wraplength=PREVIEW_SIZE + 20,
        )
        self.stage_label.pack(pady=(8, 0))

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
            fg=DIM_TEXT,
            bg=BG_COLOR,
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=12, pady=(0, 10))

    # -- Drawing --------------------------------------------------------
    def _paint(self, event):
        """Draw on canvas and PIL mirror simultaneously."""
        x, y = event.x, event.y

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                fill=BRUSH_COLOR,
                width=BRUSH_SIZE,
                capstyle=tk.ROUND,
                smooth=True,
            )
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
        """Clear the drawing canvas, preview, and result."""
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.stage_label.config(text="")

    # -- Preview --------------------------------------------------------
    def _update_preview(self, img_source):
        """
        Run preprocessing and show what the model sees in the preview panel.

        Args:
            img_source: PIL Image to preprocess and preview.
        """
        stages = get_debug_images(img_source)
        final_28x28 = stages["final"]

        # Upscale 28x28 to PREVIEW_SIZE for display (nearest neighbor to keep sharp)
        preview_pil = Image.fromarray(final_28x28)
        preview_pil = preview_pil.resize(
            (PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST
        )

        self._preview_photo = ImageTk.PhotoImage(preview_pil)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            0, 0, anchor="nw", image=self._preview_photo
        )

        # Show preprocessing info
        cropped_shape = stages["cropped"].shape
        self.stage_label.config(
            text=(
                f"Crop: {cropped_shape[1]}x{cropped_shape[0]}\n"
                f"Range: [{final_28x28.min()}-{final_28x28.max()}]"
            )
        )

    # -- Prediction -----------------------------------------------------
    def _predict(self):
        """Predict the digit drawn on the canvas."""
        if self.model is None:
            messagebox.showwarning(
                "No Model",
                "No model loaded. Run 'python -m src.main' first.",
            )
            return

        # Use advanced preprocessor
        img_array = preprocess_drawn_image(self.pil_image, debug=True)
        self._update_preview(self.pil_image)
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

        self._photo = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # Use advanced preprocessor on the ORIGINAL image (not display copy)
        img_array = preprocess_drawn_image(img, debug=True)
        self._update_preview(img)
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
