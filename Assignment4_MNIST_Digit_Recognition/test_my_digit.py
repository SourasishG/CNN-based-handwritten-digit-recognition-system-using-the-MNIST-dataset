"""
Assignment 4: MNIST Handwritten Digit Recognition
Filename: test_my_digit.py

Task Covered:
- Task 11: Predict your own handwritten digit using the trained CNN model.
           Loads an image (default: my_digit/digit.png), preprocesses it to match
           the MNIST input format (28x28 grayscale, centered, normalized),
           makes predictions, and visualizes the results.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure Matplotlib does not try to open GUI windows in non-interactive environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "digit_cnn.keras"
DEFAULT_IMAGE_PATH = BASE_DIR / "my_digit" / "digit.png"
OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTION_PLOT_PATH = OUTPUTS_DIR / "my_digit_prediction.png"


def preprocess_image(image_path: Path):
    """
    Preprocess custom digit image to match MNIST format:
    1. Grayscale conversion
    2. Automatic background inversion (MNIST uses white stroke on black background)
    3. Bounding box cropping & aspect-ratio-preserving centering onto a 28x28 canvas
    4. Normalization to [0.0, 1.0] and batch dimension expansion (1, 28, 28, 1)
    """
    # Open image
    img = Image.open(image_path).convert("L")
    original_img = img.copy()

    img_arr = np.array(img, dtype=np.float32)

    # Invert if image background is light (e.g. black ink on white paper)
    # Check average corner brightness to determine background color
    h, w = img_arr.shape
    corners = [
        img_arr[0:5, 0:5],
        img_arr[0:5, w-5:w],
        img_arr[h-5:h, 0:5],
        img_arr[h-5:h, w-5:w]
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])
    if corner_mean > 127:
        img = ImageOps.invert(img)
        img_arr = 255.0 - img_arr

    # Threshold small noise to absolute 0
    img_arr[img_arr < 30] = 0

    # Find bounding box of the digit stroke
    non_empty_rows = np.where(img_arr.max(axis=1) > 0)[0]
    non_empty_cols = np.where(img_arr.max(axis=0) > 0)[0]

    if len(non_empty_rows) > 0 and len(non_empty_cols) > 0:
        row_min, row_max = non_empty_rows[0], non_empty_rows[-1]
        col_min, col_max = non_empty_cols[0], non_empty_cols[-1]

        # Crop to bounding box
        digit_crop = img.crop((col_min, row_min, col_max + 1, row_max + 1))

        # Resize the bounding box to fit inside 20x20 while preserving aspect ratio
        w_box, h_box = digit_crop.size
        if w_box > h_box:
            new_w = 20
            new_h = max(1, int(round((h_box * 20.0) / w_box)))
        else:
            new_h = 20
            new_w = max(1, int(round((w_box * 20.0) / h_box)))

        digit_resized = digit_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Place onto a 28x28 blank (black) canvas, centered
        canvas = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        canvas.paste(digit_resized, (paste_x, paste_y))
        processed_img = canvas
    else:
        # Fallback if image is completely blank
        processed_img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to normalized numpy array
    norm_arr = np.array(processed_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(norm_arr, axis=(0, -1))  # Shape: (1, 28, 28, 1)

    return original_img, processed_img, input_tensor


def main():
    parser = argparse.ArgumentParser(description="Task 11: Predict handwritten digit using trained MNIST CNN.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE_PATH),
        help=f"Path to input digit image (default: {DEFAULT_IMAGE_PATH})"
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    print("=" * 65)
    print(" Task 11: Handwritten Digit Recognition on Custom Image")
    print("=" * 65)

    # Verify model exists
    if not MODEL_PATH.exists():
        print(f"\n[Error] Model file not found at: {MODEL_PATH}")
        print("Please train the model first by running:\n    python train.py\n")
        sys.exit(1)

    # Verify image exists
    if not image_path.exists():
        print(f"\n[Error] Image file not found at: {image_path}")
        print("Please place your handwritten digit image at:")
        print(f"    {DEFAULT_IMAGE_PATH}")
        print("or specify a custom path using --image <path_to_image>\n")
        sys.exit(1)

    print(f"\n[1/3] Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"[2/3] Preprocessing image: {image_path}")
    orig_img, proc_img, input_tensor = preprocess_image(image_path)

    print("[3/3] Running prediction...")
    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = probabilities[predicted_digit] * 100

    print("\n" + "=" * 45)
    print(f"  PREDICTED DIGIT: {predicted_digit}")
    print(f"  CONFIDENCE     : {confidence:.2f}%")
    print("=" * 45)

    print("\nClass Probabilities:")
    print("-" * 30)
    for digit in range(10):
        prob = probabilities[digit] * 100
        bar = "█" * int(prob // 4)
        indicator = " <--- (Prediction)" if digit == predicted_digit else ""
        print(f"  Digit {digit}: {prob:6.2f}% | {bar:<25}{indicator}")
    print("-" * 30)

    # Save visual verification plot
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img, cmap="gray")
    plt.title(f"Original Input Image\n({orig_img.size[0]}x{orig_img.size[1]})", fontsize=11)
    plt.axis("off")

    # Preprocessed Image
    plt.subplot(1, 3, 2)
    plt.imshow(proc_img, cmap="gray")
    plt.title("Preprocessed 28x28 Input\n(MNIST-aligned)", fontsize=11)
    plt.axis("off")

    # Probability Distribution Bar Chart
    plt.subplot(1, 3, 3)
    bars = plt.bar(range(10), probabilities * 100, color="cornflowerblue", edgecolor="black")
    bars[predicted_digit].set_color("forestgreen")
    plt.title(f"Prediction: {predicted_digit} ({confidence:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Digit (0-9)")
    plt.ylabel("Confidence (%)")
    plt.xticks(range(10))
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=300)
    plt.close()
    print(f"\n[Visualization Saved] Output plot saved to: {PREDICTION_PLOT_PATH}")


if __name__ == "__main__":
    main()

