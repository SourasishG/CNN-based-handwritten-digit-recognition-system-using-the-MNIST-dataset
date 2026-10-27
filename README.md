# Handwritten Digit Recognition using a CNN on MNIST

A simple, beginner-friendly Convolutional Neural Network (CNN) project built with TensorFlow and Keras to classify handwritten digits (0 to 9) from the MNIST dataset.

---

## 1. Installation

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

---

## 2. How to Run the Project

### Step 1: Train and Evaluate the Model
Run `train.py` to train the CNN on MNIST, evaluate accuracy, and generate all output plots and reports:

```bash
python train.py
```

This will save:
- The trained model to `model/digit_cnn.keras`
- Visualizations and metrics to the `outputs/` folder (`sample_digits.png`, `training_curves.png`, `confusion_matrix.png`, `predictions.png`, `classification_report.txt`)

### Step 2: Test on Your Own Handwritten Digit
Place your digit image in `my_digit/digit.png`, then run:

```bash
python test_my_digit.py
```

This will print the predicted digit and confidence percentage, and save the result to `outputs/my_digit_prediction.png`.

---

## 3. How to Make Your Own Digit Image

You can draw a digit using MS Paint, Photoshop, or on paper and take a photo:
1. **White Background, Dark Digit (Standard Drawing)**: Draw a dark number on a white background (e.g., black ink on paper or default MS Paint). The script automatically detects the white background and inverts it for the model.
2. **Black Background, White Digit (MNIST Style)**: Draw a bright white digit on a solid black background.
3. Save the image as a square PNG named `digit.png` inside the `my_digit/` folder.

---

## 4. How the CNN Works (5-Line Explanation)

1. **Convolutional Layers (Conv2D)**: Scan the input image with small filters to detect basic visual features like edges, corners, and curves.
2. **Activation Function (ReLU)**: Introduces non-linearity so the network can learn complex, curved patterns in handwritten strokes.
3. **Pooling Layers (MaxPooling2D)**: Shrink the spatial dimensions of the feature maps to reduce computation and make detection position-invariant.
4. **Flatten Layer**: Converts the 2D feature maps into a single 1D vector so standard dense layers can process them.
5. **Dense Output Layer (Softmax)**: Computes probability scores across all 10 possible digits (0 through 9) to pick the most confident prediction.

