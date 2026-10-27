"""
Assignment 4: MNIST Handwritten Digit Recognition
Filename: train.py

Tasks Covered:
- Task 1: Load and explore the MNIST dataset
- Task 2: Visualize sample digits from the dataset (outputs/sample_digits.png)
- Task 3: Preprocess dataset (normalization, channel expansion, one-hot encoding)
- Task 4: Split dataset into training and validation sets
- Task 5: Build a deep Convolutional Neural Network (CNN) architecture
- Task 6: Compile the model with optimizer, loss function, and metrics
- Task 7: Train the model with callbacks (EarlyStopping, ReduceLROnPlateau, Checkpoint)
- Task 8: Plot training and validation loss/accuracy curves (outputs/training_curves.png)
- Task 9: Evaluate model on test data and generate classification report (outputs/classification_report.txt)
- Task 10: Compute confusion matrix (outputs/confusion_matrix.png), visualize sample predictions
          (outputs/predictions.png), and save final model (model/digit_cnn.keras)
"""

import os
import random
import numpy as np
from pathlib import Path

# Ensure Matplotlib does not try to open GUI windows in non-interactive environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# Configuration and Reproducibility Setup
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / "digit_cnn.keras"

NUM_CLASSES = 10
IMG_ROWS, IMG_COLS = 28, 28
BATCH_SIZE = 64
EPOCHS = 15


def main():
    print("=" * 70)
    print(" Assignment 4: MNIST Digit Recognition - Model Training Pipeline")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Task 1: Load and Explore MNIST Dataset
    # --------------------------------------------------------------------------
    print("\n[Task 1] Loading MNIST dataset...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

    print(f"  * Training raw shape : {x_train_raw.shape}, labels: {y_train_raw.shape}")
    print(f"  * Test raw shape     : {x_test_raw.shape}, labels: {y_test_raw.shape}")
    print(f"  * Pixel range        : [{x_train_raw.min()}, {x_train_raw.max()}]")
    print(f"  * Unique classes     : {np.unique(y_train_raw)}")

    # --------------------------------------------------------------------------
    # Task 2: Visualize Sample Digits
    # --------------------------------------------------------------------------
    print("\n[Task 2] Visualizing sample digits...")
    sample_path = OUTPUTS_DIR / "sample_digits.png"
    plt.figure(figsize=(10, 5))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_train_raw[i], cmap="gray")
        plt.title(f"Label: {y_train_raw[i]}", fontsize=10)
        plt.axis("off")
    plt.suptitle("Task 2: Sample Digits from MNIST Training Dataset", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(sample_path, dpi=300)
    plt.close()
    print(f"  * Saved sample digits grid to: {sample_path}")

    # --------------------------------------------------------------------------
    # Task 3: Preprocess Dataset
    # --------------------------------------------------------------------------
    print("\n[Task 3] Preprocessing dataset (normalizing, reshaping, one-hot encoding)...")
    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    x_train_norm = x_train_raw.astype("float32") / 255.0
    x_test_norm = x_test_raw.astype("float32") / 255.0

    # Expand channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train_reshaped = np.expand_dims(x_train_norm, axis=-1)
    x_test_reshaped = np.expand_dims(x_test_norm, axis=-1)

    # Convert labels to one-hot encoded vectors
    y_train_cat = keras.utils.to_categorical(y_train_raw, NUM_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test_raw, NUM_CLASSES)

    print(f"  * Preprocessed X_train shape: {x_train_reshaped.shape}")
    print(f"  * Preprocessed X_test shape : {x_test_reshaped.shape}")
    print(f"  * One-hot y_train shape     : {y_train_cat.shape}")

    # --------------------------------------------------------------------------
    # Task 4: Train / Validation Split
    # --------------------------------------------------------------------------
    print("\n[Task 4] Splitting training data into train and validation sets...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_reshaped,
        y_train_cat,
        test_size=10000,
        random_state=SEED,
        stratify=y_train_raw
    )
    print(f"  * Training set   : {x_train.shape[0]} samples")
    print(f"  * Validation set : {x_val.shape[0]} samples")
    print(f"  * Test set       : {x_test_reshaped.shape[0]} samples")

    # --------------------------------------------------------------------------
    # Task 5: Build CNN Architecture
    # --------------------------------------------------------------------------
    print("\n[Task 5] Building Convolutional Neural Network (CNN) architecture...")
    model = keras.Sequential([
        # Convolutional Block 1
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(IMG_ROWS, IMG_COLS, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Convolutional Block 2
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Classification Dense Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="MNIST_CNN_Classifier")

    model.summary()

    # --------------------------------------------------------------------------
    # Task 6: Compile Model
    # --------------------------------------------------------------------------
    print("\n[Task 6] Compiling the model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("  * Optimizer: Adam (lr=0.001)")
    print("  * Loss: categorical_crossentropy")
    print("  * Metrics: ['accuracy']")

    # --------------------------------------------------------------------------
    # Task 7: Train Model with Callbacks
    # --------------------------------------------------------------------------
    print("\n[Task 7] Training model with callbacks...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------------------------------
    # Task 8: Plot Training Curves
    # --------------------------------------------------------------------------
    print("\n[Task 8] Generating training & validation curves...")
    curves_path = OUTPUTS_DIR / "training_curves.png"
    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history["accuracy"], "o-", label="Train Accuracy", linewidth=2)
    plt.plot(epochs_range, history.history["val_accuracy"], "s--", label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy vs Epochs", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history["loss"], "o-", label="Train Loss", linewidth=2)
    plt.plot(epochs_range, history.history["val_loss"], "s--", label="Validation Loss", linewidth=2)
    plt.title("Model Loss vs Epochs", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    plt.suptitle("Task 8: CNN Training Curves (Loss & Accuracy)", fontsize=14)
    plt.tight_layout()
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"  * Saved training curves to: {curves_path}")

    # --------------------------------------------------------------------------
    # Task 9: Evaluate Model on Test Set & Classification Report
    # --------------------------------------------------------------------------
    print("\n[Task 9] Evaluating model on unseen test set...")
    test_loss, test_acc = model.evaluate(x_test_reshaped, y_test_cat, verbose=0)
    print(f"  * Test Loss     : {test_loss:.4f}")
    print(f"  * Test Accuracy : {test_acc * 100:.2f}%")

    # Predict test probabilities and labels
    y_pred_probs = model.predict(x_test_reshaped, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test_raw

    # Generate classification report
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=[f"Digit {i}" for i in range(NUM_CLASSES)],
        digits=4
    )
    print("\n--- Classification Report ---")
    print(report_str)

    report_path = OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Assignment 4: MNIST Digit Recognition Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc * 100:.2f}%\n\n")
        f.write("Per-Class Classification Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(report_str)
        f.write("\n" + "=" * 60 + "\n")
    print(f"  * Saved classification report to: {report_path}")

    # --------------------------------------------------------------------------
    # Task 10: Confusion Matrix, Visual Predictions & Save Model
    # --------------------------------------------------------------------------
    print("\n[Task 10] Generating confusion matrix and sample test predictions...")
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(i) for i in range(NUM_CLASSES)],
        yticklabels=[str(i) for i in range(NUM_CLASSES)],
        cbar=True
    )
    plt.title("Task 10: MNIST Confusion Matrix", fontsize=14, pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  * Saved confusion matrix to: {cm_path}")

    # Sample Predictions Grid
    pred_path = OUTPUTS_DIR / "predictions.png"
    plt.figure(figsize=(12, 8))
    # Pick a fixed selection of test samples
    sample_indices = np.linspace(0, len(x_test_raw) - 1, 24, dtype=int)
    for idx, sample_idx in enumerate(sample_indices):
        plt.subplot(4, 6, idx + 1)
        plt.imshow(x_test_raw[sample_idx], cmap="gray")
        pred_lbl = y_pred[sample_idx]
        true_lbl = y_true[sample_idx]
        confidence = y_pred_probs[sample_idx][pred_lbl] * 100

        is_correct = (pred_lbl == true_lbl)
        color = "green" if is_correct else "red"
        plt.title(f"Pred: {pred_lbl} ({confidence:.0f}%)\nTrue: {true_lbl}", color=color, fontsize=9)
        plt.axis("off")

    plt.suptitle("Task 10: Sample Test Predictions (Green=Correct, Red=Incorrect)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"  * Saved test predictions visualization to: {pred_path}")

    # Save final model explicitly
    model.save(MODEL_SAVE_PATH)
    print(f"\n[Model Saved] Trained model successfully saved to: {MODEL_SAVE_PATH}")
    print("\n" + "=" * 70)
    print(" Pipeline completed successfully! Ready for Task 11 (test_my_digit.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()

