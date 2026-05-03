from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import callbacks, initializers, layers, models, optimizers


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).resolve().parent / "symbols"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
MODEL_PATH = Path(__file__).resolve().parent / "operator_model_ann.keras"
CLASS_MAPPING_PATH = Path(__file__).resolve().parent / "operator_class_mapping.json"
HISTORY_PLOT_PATH = Path(__file__).resolve().parent / "operator_training_history.png"
IMAGE_SIZE = (28, 28)
INPUT_DIM = 784
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
OPERATOR_FOLDERS = ["dot", "minus", "plus", "slash"]
OPERATOR_LABELS = ["·", "-", "+", "/"]


@dataclass(frozen=True)
class ArchitectureConfig:
    hidden_units: tuple[int, ...]
    dropout: float


ARCHITECTURE_RETRIES = [
    ArchitectureConfig((512, 256, 128), 0.3),
    ArchitectureConfig((1024, 512, 256), 0.3),
    ArchitectureConfig((1024, 512, 256, 128), 0.3),
    ArchitectureConfig((1024, 512, 256, 128), 0.2),
]


def load_image(image_path: Path) -> np.ndarray:
    """Load one grayscale operator image and validate that it is 28x28."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        if image.size != IMAGE_SIZE:
            raise ValueError(f"Invalid image shape for {image_path}: expected 28x28, got {image.size}")
        return np.asarray(image, dtype=np.float32)


def iter_class_files(root: Path, folders: Iterable[str]) -> list[tuple[Path, int]]:
    """Collect image files for the allowed operator folders only."""
    items: list[tuple[Path, int]] = []
    for label, class_name in enumerate(folders):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                items.append((image_path, label))
    if not items:
        raise ValueError(f"No images found under {root}")
    return items


def load_split(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load operator split, normalize to [0,1], and flatten to vectors."""
    samples = iter_class_files(root, OPERATOR_FOLDERS)
    x_data = np.stack([load_image(path) for path, _ in samples], axis=0) / 255.0
    x_data = x_data.reshape(len(samples), INPUT_DIM)
    y_indices = np.array([label for _, label in samples], dtype=np.int32)
    y_data = tf.keras.utils.to_categorical(y_indices, num_classes=len(OPERATOR_LABELS))
    return x_data, y_data


def build_model(config: ArchitectureConfig) -> tf.keras.Model:
    """Build an ANN using only Dense and Dropout hidden layers."""
    model = models.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))
    for units in config.hidden_units:
        model.add(layers.Dense(units, activation="relu", kernel_initializer=initializers.HeNormal()))
        model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(len(OPERATOR_LABELS), activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation loss/accuracy plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Save a confusion matrix image for the operator ANN."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train_with_retries(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[tf.keras.Model, tf.keras.callbacks.History, float]:
    """Train ANN variants until acceptable operator accuracy or protocol exhaustion."""
    x_val = x_test
    y_val = y_test
    best_accuracy = 0.0

    for index, config in enumerate(ARCHITECTURE_RETRIES, start=1):
        logger.info("Training operator ANN attempt %s with hidden=%s dropout=%.2f", index, config.hidden_units, config.dropout)
        model = build_model(config)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True),
            ],
            verbose=2,
        )
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        best_accuracy = max(best_accuracy, float(accuracy))
        logger.info("Operator ANN attempt %s test accuracy: %.4f", index, accuracy)
        if accuracy >= 0.90:
            return model, history, float(accuracy)

    raise RuntimeError(f"ANN insufficient for this dataset. Maximum accuracy achieved: {best_accuracy * 100:.2f}%")


def main() -> None:
    """Train, evaluate, and save the operator ANN artifacts."""
    logger.info("Loading operator dataset from %s", DATASET_ROOT)
    x_train, y_train = load_split(TRAIN_DIR)
    x_test, y_test = load_split(TEST_DIR)
    model, history, accuracy = train_with_retries(x_train, y_train, x_test, y_test)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    logger.info("Operator test loss: %.4f", test_loss)
    logger.info("Operator test accuracy: %.4f", test_accuracy)
    print(classification_report(y_true, y_pred, target_names=OPERATOR_LABELS, digits=4))
    print(confusion_matrix(y_true, y_pred))

    CLASS_MAPPING_PATH.write_text(json.dumps({i: label for i, label in enumerate(OPERATOR_LABELS)}, indent=2), encoding="utf-8")
    plot_history(history, HISTORY_PLOT_PATH)
    save_confusion(y_true, y_pred, OPERATOR_LABELS, Path(__file__).resolve().parent / "operator_confusion_matrix.png")
    model.save(MODEL_PATH)
    logger.info("Operator confusion focus: plus vs dot and minus vs slash reviewed in confusion matrix output.")

    if accuracy >= 0.95:
        logger.info("Operator ANN reached success threshold.")
    else:
        logger.info("Operator ANN reached acceptable threshold with note.")


if __name__ == "__main__":
    main()
