"""
MNIST-aligned image preprocessor for drawn/uploaded digits.

Transforms raw canvas drawings or uploaded images into the same
distribution as MNIST training data:
    - White digit on black background
    - Centered within a 20x20 box inside 28x28 (like MNIST)
    - Normalized to [0, 1]

This is the KEY to making real-world digit recognition work.
Without proper preprocessing, even a 99% accurate MNIST model
will fail on hand-drawn input.
"""

import numpy as np
from PIL import Image, ImageOps

MNIST_SIZE = 28
DIGIT_BOX = 20   # MNIST digits fit in a ~20x20 box, centered in 28x28


def preprocess_drawn_image(
    img,
    debug: bool = False,
) -> np.ndarray:
    """
    Preprocess a drawn or uploaded digit image to match MNIST distribution.

    Pipeline:
        1. Convert to grayscale
        2. Invert colors if needed (MNIST = white digit on black)
        3. Detect bounding box of the digit
        4. Crop to bounding box
        5. Fit into a 20x20 box (maintain aspect ratio)
        6. Center in a 28x28 canvas (matching MNIST layout)
        7. Normalize pixel values to [0, 1]
        8. Flatten to (1, 784)

    Args:
        img: PIL Image or numpy array of the drawn digit.
        debug: If True, print shape/value diagnostics.

    Returns:
        numpy array of shape (1, 784), dtype float32, values in [0, 1].
    """

    # -- 1. Accept numpy or PIL, convert to grayscale PIL -----------------
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("L")

    if debug:
        print(f"[Preprocessor] Input size: {img.size}")

    # -- 2. Invert if needed (white-on-black like MNIST) ------------------
    img_array = np.array(img)
    if _background_is_light(img_array):
        img = ImageOps.invert(img)
        img_array = np.array(img)
        if debug:
            print("[Preprocessor] Inverted colors (was light background)")

    # -- 3. Detect bounding box of the digit ------------------------------
    bbox = _find_digit_bbox(img_array, threshold=30)

    if bbox is None:
        # Canvas is blank -- return zeros
        if debug:
            print("[Preprocessor] No digit detected -- returning blank")
        return np.zeros((1, MNIST_SIZE * MNIST_SIZE), dtype="float32")

    top, bottom, left, right = bbox
    if debug:
        print(f"[Preprocessor] Bounding box: top={top}, bottom={bottom}, "
              f"left={left}, right={right}")

    # -- 4. Crop to bounding box ------------------------------------------
    cropped = img_array[top:bottom + 1, left:right + 1]

    if debug:
        print(f"[Preprocessor] Cropped size: {cropped.shape}")

    # -- 5. Fit into 20x20 box maintaining aspect ratio -------------------
    fitted = _fit_to_box(cropped, DIGIT_BOX)

    if debug:
        print(f"[Preprocessor] Fitted to {DIGIT_BOX}x{DIGIT_BOX} box: "
              f"{fitted.shape}")

    # -- 6. Center in 28x28 canvas ----------------------------------------
    canvas = _center_in_canvas(fitted, MNIST_SIZE)

    if debug:
        print(f"[Preprocessor] Final canvas: {canvas.shape}")
        print(f"[Preprocessor] Value range: [{canvas.min():.3f}, "
              f"{canvas.max():.3f}]")

    # -- 7. Normalize to [0, 1] -------------------------------------------
    normalized = canvas.astype("float32") / 255.0

    # -- 8. Flatten to (1, 784) -------------------------------------------
    result = normalized.reshape(1, MNIST_SIZE * MNIST_SIZE)

    if debug:
        print(f"[Preprocessor] Output shape: {result.shape}, "
              f"dtype: {result.dtype}")
        print(f"[Preprocessor] Output range: [{result.min():.3f}, "
              f"{result.max():.3f}]")

    return result


def get_debug_images(img) -> dict:
    """
    Return intermediate preprocessing stages for visualization.

    Args:
        img: PIL Image or numpy array.

    Returns:
        Dictionary with keys:
            'original'  -- grayscale input (after inversion if needed)
            'cropped'   -- bounding-box crop of the digit
            'fitted'    -- digit fitted to 20x20 box
            'final'     -- 28x28 centered canvas (what the model sees)
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("L")

    img_array = np.array(img)
    if _background_is_light(img_array):
        img = ImageOps.invert(img)
        img_array = np.array(img)

    stages = {"original": img_array.copy()}

    bbox = _find_digit_bbox(img_array, threshold=30)
    if bbox is None:
        blank = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype="uint8")
        return {
            "original": img_array,
            "cropped": blank,
            "fitted": blank,
            "final": blank,
        }

    top, bottom, left, right = bbox
    cropped = img_array[top:bottom + 1, left:right + 1]
    stages["cropped"] = cropped

    fitted = _fit_to_box(cropped, DIGIT_BOX)
    stages["fitted"] = fitted

    canvas = _center_in_canvas(fitted, MNIST_SIZE)
    stages["final"] = canvas

    return stages


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _background_is_light(img_array: np.ndarray) -> bool:
    """
    Determine if the background is light (white-ish).

    Uses corner sampling instead of global mean -- more robust when
    the digit covers a large portion of the image.
    """
    h, w = img_array.shape
    # Sample corners (4 regions of 5x5 pixels each)
    corners = [
        img_array[:5, :5],       # top-left
        img_array[:5, -5:],      # top-right
        img_array[-5:, :5],      # bottom-left
        img_array[-5:, -5:],     # bottom-right
    ]
    corner_mean = np.mean([c.mean() for c in corners])
    return corner_mean > 127


def _find_digit_bbox(img_array: np.ndarray, threshold: int = 30):
    """
    Find the tight bounding box around non-zero pixels.

    Args:
        img_array: 2D grayscale array (white digit on black background).
        threshold: Minimum pixel value to consider as part of the digit.

    Returns:
        Tuple (top, bottom, left, right) or None if no digit found.
    """
    mask = img_array > threshold

    # Find rows and columns that contain digit pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    top = np.argmax(rows)
    bottom = len(rows) - 1 - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - 1 - np.argmax(cols[::-1])

    return (top, bottom, left, right)


def _fit_to_box(cropped: np.ndarray, box_size: int) -> np.ndarray:
    """
    Resize the cropped digit to fit inside a box_size x box_size area,
    preserving aspect ratio.

    This mimics how MNIST digits are formatted: each digit fits
    inside a ~20x20 pixel region.

    Args:
        cropped: 2D array of the cropped digit.
        box_size: Target box size (default 20 for MNIST).

    Returns:
        2D array of shape (box_size, box_size) or smaller with
        the digit centered.
    """
    h, w = cropped.shape

    if h == 0 or w == 0:
        return np.zeros((box_size, box_size), dtype="uint8")

    # Calculate scale factor to fit the larger dimension into box_size
    scale = box_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Resize using PIL for high-quality anti-aliasing
    pil_cropped = Image.fromarray(cropped)
    pil_resized = pil_cropped.resize((new_w, new_h), Image.LANCZOS)

    return np.array(pil_resized)


def _center_in_canvas(digit: np.ndarray, canvas_size: int) -> np.ndarray:
    """
    Place the digit in the center of a canvas_size x canvas_size black canvas.

    MNIST digits are centered by center-of-mass, so we do the same:
    compute the center of mass of the digit and align it with the
    center of the 28x28 canvas.

    Args:
        digit: 2D array of the resized digit.
        canvas_size: Size of the output canvas (28 for MNIST).

    Returns:
        2D array of shape (canvas_size, canvas_size).
    """
    canvas = np.zeros((canvas_size, canvas_size), dtype="uint8")
    h, w = digit.shape

    if h == 0 or w == 0:
        return canvas

    # Compute center of mass of the digit
    digit_float = digit.astype("float32")
    total_mass = digit_float.sum()

    if total_mass == 0:
        return canvas

    # Center of mass (row, col)
    rows = np.arange(h).reshape(-1, 1)
    cols = np.arange(w).reshape(1, -1)
    com_row = (rows * digit_float).sum() / total_mass
    com_col = (cols * digit_float).sum() / total_mass

    # Target: center of mass should be at canvas center
    canvas_center = canvas_size / 2.0
    offset_row = int(round(canvas_center - com_row))
    offset_col = int(round(canvas_center - com_col))

    # Calculate paste region (clamp to canvas bounds)
    src_top = max(0, -offset_row)
    src_left = max(0, -offset_col)
    dst_top = max(0, offset_row)
    dst_left = max(0, offset_col)

    paste_h = min(h - src_top, canvas_size - dst_top)
    paste_w = min(w - src_left, canvas_size - dst_left)

    if paste_h > 0 and paste_w > 0:
        canvas[
            dst_top:dst_top + paste_h,
            dst_left:dst_left + paste_w,
        ] = digit[
            src_top:src_top + paste_h,
            src_left:src_left + paste_w,
        ]

    return canvas
