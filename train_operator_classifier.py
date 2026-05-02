"""Train an operator-only CNN on a mixed symbols dataset."""

from __future__ import annotations

import json
import random
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_PATH = Path("archive.zip")
EXTRACTED_DATASET_DIR = Path("symbols")
OUTPUT_MODEL_PATH = Path("operator_model.keras")
CLASS_MAPPING_PATH = Path("operator_class_mapping.json")
TRAINING_HISTORY_PLOT_PATH = Path("operator_training_history.png")
RANDOM_SEED = 42
IMAGE_SIZE = (28, 28)
INPUT_SHAPE = (28, 28, 1)

OPERATOR_FOLDER_TO_LABEL = {
    "plus": "+",
    "minus": "-",
    "dot": "·",
    "slash": "÷",
}
CLASS_ORDER = ["+", "-", "·", "÷"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_ORDER)}

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
    """Load a grayscale image, resize if needed, and return a float array."""
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
    """Load only allowed operator images from a zip archive."""
    images: List[np.ndarray] = []
    labels: List[int] = []

    with zipfile.ZipFile(dataset_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            parts = Path(member).parts
            if len(parts) < 4:
                continue

            folder_name = parts[-2]
            if folder_name not in OPERATOR_FOLDER_TO_LABEL:
                continue

            operator_label = OPERATOR_FOLDER_TO_LABEL[folder_name]
            label_index = LABEL_TO_INDEX[operator_label]

            with archive.open(member) as image_file:
                image_array = _load_image_from_bytes(image_file.read())

            images.append(image_array)
            labels.append(label_index)

    if not images:
        raise ValueError(f"No allowed operator images were found in archive: {dataset_path}")

    x_data = np.array(images, dtype=np.float32)
    y_data = np.array(labels, dtype=np.int32)
    class_mapping = {idx: label for idx, label in enumerate(CLASS_ORDER)}
    return x_data, y_data, class_mapping


def _load_from_directory(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Load only allowed operator images from an extracted directory."""
    images: List[np.ndarray] = []
    labels: List[int] = []

    for image_path in dataset_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        folder_name = image_path.parent.name
        if folder_name not in OPERATOR_FOLDER_TO_LABEL:
            continue

        operator_label = OPERATOR_FOLDER_TO_LABEL[folder_name]
        label_index = LABEL_TO_INDEX[operator_label]

        image_array = _load_image_from_path(image_path)
        images.append(image_array)
        labels.append(label_index)

    if not images:
        raise ValueError(f"No allowed operator images were found in directory: {dataset_dir}")

    x_data = np.array(images, dtype=np.float32)
    y_data = np.array(labels, dtype=np.int32)
    class_mapping = {idx: label for idx, label in enumerate(CLASS_ORDER)}
    return x_data, y_data, class_mapping


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], Dict[int, float]]:
    """Load, filter, normalize, split operator data, and compute class weights."""
    if DATASET_PATH.exists():
        x_data, y_indices, class_mapping = _load_from_zip(DATASET_PATH)
    elif EXTRACTED_DATASET_DIR.exists():
        x_data, y_indices, class_mapping = _load_from_directory(EXTRACTED_DATASET_DIR)
    else:
        raise FileNotFoundError(
            f"Dataset not found. Expected either archive at {DATASET_PATH} "
            f"or extracted directory at {EXTRACTED_DATASET_DIR}."
        )

    x_data = (x_data / 255.0).astype(np.float32)
    x_data = np.expand_dims(x_data, axis=-1)

    x_train, x_temp, y_train_idx, y_temp_idx = train_test_split(
        x_data,
        y_indices,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_indices,
    )

    validation_fraction_of_temp = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    x_val, x_test, y_val_idx, y_test_idx = train_test_split(
        x_temp,
        y_temp_idx,
        test_size=1.0 - validation_fraction_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp_idx,
    )

    y_train = tf.keras.utils.to_categorical(y_train_idx, num_classes=len(CLASS_ORDER))
    y_val = tf.keras.utils.to_categorical(y_val_idx, num_classes=len(CLASS_ORDER))
    y_test = tf.keras.utils.to_categorical(y_test_idx, num_classes=len(CLASS_ORDER))

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_ORDER)),
        y=y_train_idx,
    )
    class_weights = {int(idx): float(weight) for idx, weight in enumerate(class_weights_array)}

    return x_train, y_train, x_val, y_val, x_test, y_test, class_mapping, class_weights


def build_model(input_shape: Tuple[int, int, int] = INPUT_SHAPE) -> tf.keras.Model:
    """Build and compile the CNN model for operator classification."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer=HeNormal(), padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer=HeNormal(), padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer=HeNormal(), padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(4, activation="softmax"),
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
    class_weights: Dict[int, float],
) -> tf.keras.callbacks.History:
    """Train the CNN with augmentation, class weights, and callbacks."""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
    )
    train_datagen.fit(x_train)

    model_callbacks: List[callbacks.Callback] = [
        callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath=str(OUTPUT_MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=model_callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    return history


def _plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Plot and save training/validation loss and accuracy."""
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
    """Save class index to operator symbol mapping as JSON."""
    serializable_mapping = {str(index): label for index, label in class_mapping.items()}
    with CLASS_MAPPING_PATH.open("w", encoding="utf-8") as mapping_file:
        json.dump(serializable_mapping, mapping_file, indent=2)


def _print_common_confusions(matrix: np.ndarray, class_mapping: Dict[int, str]) -> None:
    """Print counts for specific confusion pairs requested for operator analysis."""
    inverse = {label: idx for idx, label in class_mapping.items()}

    pairs = [
        ("+", "·"),
        ("-", "÷"),
        ("·", "÷"),
    ]

    print("\n=== Common Confusion Checks ===")
    for a, b in pairs:
        ia, ib = inverse[a], inverse[b]
        a_to_b = int(matrix[ia, ib])
        b_to_a = int(matrix[ib, ia])
        print(f"{a} -> {b}: {a_to_b}, {b} -> {a}: {b_to_a}")


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_mapping: Dict[int, str],
    history: tf.keras.callbacks.History,
) -> None:
    """Evaluate the model, save artifacts, and print summary metrics."""
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
    _print_common_confusions(matrix, class_mapping)
    print("\nSaved artifacts:")
    print(f"- Best model: {OUTPUT_MODEL_PATH}")
    print(f"- Class mapping: {CLASS_MAPPING_PATH}")
    print(f"- Training history plot: {TRAINING_HISTORY_PLOT_PATH}")


def main() -> None:
    """Run the end-to-end operator-only training pipeline."""
    set_seed(RANDOM_SEED)
    x_train, y_train, x_val, y_val, x_test, y_test, class_mapping, class_weights = load_data()

    print("Loaded operator-only dataset (plus, minus, dot, slash folders).")
    print("Interpreting classes as: plus=+, minus=-, dot=·, slash=÷")
    print("Excluded all digits and excluded x (cross multiply).")
    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Classes: {class_mapping}")
    print(f"Class weights: {class_weights}")

    model = build_model()
    model.summary()

    history = train_model(model, x_train, y_train, x_val, y_val, class_weights)

    if OUTPUT_MODEL_PATH.exists():
        best_model = tf.keras.models.load_model(OUTPUT_MODEL_PATH)
    else:
        best_model = model

    evaluate_model(best_model, x_test, y_test, class_mapping, history)


if __name__ == "__main__":
    main()
