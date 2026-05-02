"""Train a digit-only CNN on a mixed symbols dataset.

This script extracts only digit classes (0-9) from an archive or directory-based
image dataset, trains a CNN classifier, saves the best model, and reports final
metrics on a held-out test set.
"""

from __future__ import annotations

import json
import random
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_PATH = Path("archive.zip")
EXTRACTED_DATASET_DIR = Path("symbols")
OUTPUT_MODEL_PATH = Path("digit_model.keras")
CLASS_MAPPING_PATH = Path("class_mapping.json")
TRAINING_HISTORY_PLOT_PATH = Path("training_history.png")
RANDOM_SEED = 42
IMAGE_SIZE = (28, 28)
INPUT_SHAPE = (28, 28, 1)
DIGIT_CLASSES = [str(i) for i in range(10)]
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load a grayscale image, resize it if needed, and return a float array."""
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("L")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)
    return image_array


def _load_image_from_path(image_path: Path) -> np.ndarray:
    """Load a grayscale image from disk and return a float array."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)
    return image_array


def _load_from_zip(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Load only digit images from a zip archive containing split/class/image files."""
    images: List[np.ndarray] = []
    labels: List[int] = []
    class_mapping = {index: label for index, label in enumerate(DIGIT_CLASSES)}

    with zipfile.ZipFile(dataset_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            parts = Path(member).parts
            if len(parts) < 4:
                continue

            class_name = parts[-2]
            if class_name not in DIGIT_CLASSES:
                continue

            with archive.open(member) as image_file:
                image_array = _load_image_from_bytes(image_file.read())

            images.append(image_array)
            labels.append(int(class_name))

    if not images:
        raise ValueError(f"No digit images were found in archive: {dataset_path}")

    x_data = np.array(images, dtype=np.float32)
    y_data = np.array(labels, dtype=np.int32)
    return x_data, y_data, class_mapping


def _load_from_directory(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Load only digit images from a directory containing split/class/image files."""
    images: List[np.ndarray] = []
    labels: List[int] = []
    class_mapping = {index: label for index, label in enumerate(DIGIT_CLASSES)}

    for image_path in dataset_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        class_name = image_path.parent.name
        if class_name not in DIGIT_CLASSES:
            continue

        image_array = _load_image_from_path(image_path)
        images.append(image_array)
        labels.append(int(class_name))

    if not images:
        raise ValueError(f"No digit images were found in directory: {dataset_dir}")

    x_data = np.array(images, dtype=np.float32)
    y_data = np.array(labels, dtype=np.int32)
    return x_data, y_data, class_mapping


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """Load, filter, normalize, and split digit data into train/validation/test sets."""
    if DATASET_PATH.exists():
        x_data, y_data, class_mapping = _load_from_zip(DATASET_PATH)
    elif EXTRACTED_DATASET_DIR.exists():
        x_data, y_data, class_mapping = _load_from_directory(EXTRACTED_DATASET_DIR)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Expected either archive at {DATASET_PATH} "
            f"or extracted directory at {EXTRACTED_DATASET_DIR}."
        )

    x_data = (x_data / 255.0).astype(np.float32)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = tf.keras.utils.to_categorical(y_data, num_classes=len(DIGIT_CLASSES))

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_data,
        y_data,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=np.argmax(y_data, axis=1),
    )

    validation_fraction_of_temp = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1.0 - validation_fraction_of_temp,
        random_state=RANDOM_SEED,
        stratify=np.argmax(y_temp, axis=1),
    )

    return x_train, y_train, x_val, y_val, x_test, y_test, class_mapping


def build_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> tf.keras.Model:
    """Build and compile the CNN model for digit classification."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_initializer=HeNormal(),
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_initializer=HeNormal(),
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tf.keras.callbacks.History:
    """Train the CNN with augmentation and standard convergence callbacks."""
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
    )
    train_datagen.fit(x_train)

    model_callbacks: List[callbacks.Callback] = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=model_callbacks,
        verbose=1,
    )
    return history


def _plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Plot training and validation accuracy/loss curves and save them to disk."""
    history_data = history.history
    epochs_range = range(1, len(history_data["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_data["loss"], label="Train Loss")
    plt.plot(epochs_range, history_data["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_data["accuracy"], label="Train Accuracy")
    plt.plot(epochs_range, history_data["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PLOT_PATH, dpi=300)
    plt.close()


def _save_class_mapping(class_mapping: Dict[int, str]) -> None:
    """Save the class index to digit-label mapping as JSON."""
    serializable_mapping = {str(index): label for index, label in class_mapping.items()}
    with CLASS_MAPPING_PATH.open("w", encoding="utf-8") as mapping_file:
        json.dump(serializable_mapping, mapping_file, indent=2)


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_mapping: Dict[int, str],
    history: tf.keras.callbacks.History,
) -> None:
    """Evaluate the trained model, save artifacts, and print a final report."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = [class_mapping[index] for index in range(len(class_mapping))]
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    _plot_training_history(history)
    _save_class_mapping(class_mapping)

    print("\n=== Final Test Metrics ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\n=== Per-Class Classification Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(matrix)
    print("\nSaved artifacts:")
    print(f"- Best model: {OUTPUT_MODEL_PATH}")
    print(f"- Class mapping: {CLASS_MAPPING_PATH}")
    print(f"- Training history plot: {TRAINING_HISTORY_PLOT_PATH}")


def main() -> None:
    """Run the end-to-end digit-only training pipeline."""
    set_seed(RANDOM_SEED)
    x_train, y_train, x_val, y_val, x_test, y_test, class_mapping = load_data()

    print("Loaded digit-only dataset.")
    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Classes: {class_mapping}")

    model = build_model()
    model.summary()

    history = train_model(model, x_train, y_train, x_val, y_val)

    best_model_path = OUTPUT_MODEL_PATH
    if best_model_path.exists():
        best_model = tf.keras.models.load_model(best_model_path)
    else:
        best_model = model

    evaluate_model(best_model, x_test, y_test, class_mapping, history)


if __name__ == "__main__":
    main()
