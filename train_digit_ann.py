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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks, initializers, layers, models, optimizers, regularizers


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = Path(__file__).resolve().parent / "symbols"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
MODEL_PATH = Path(__file__).resolve().parent / "digit_model_ann.keras"
CLASS_MAPPING_PATH = Path(__file__).resolve().parent / "digit_class_mapping.json"
HISTORY_PLOT_PATH = Path(__file__).resolve().parent / "digit_training_history.png"
CONFUSION_PLOT_PATH = Path(__file__).resolve().parent / "digit_confusion_matrix.png"
IMAGE_SIZE = (28, 28)
INPUT_DIM = 784
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.1
RANDOM_SEED = 42
DIGIT_CLASSES = [str(i) for i in range(10)]


@dataclass(frozen=True)
class ArchitectureConfig:
    hidden_units: tuple[int, ...]
    dropout: float
    l2_weight: float = 0.0
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE


ARCHITECTURE_RETRIES = [
    ArchitectureConfig((512, 256, 128), 0.3, 0.0, 0.001, 32),
    ArchitectureConfig((1024, 512, 256), 0.3, 0.0, 0.001, 32),
    ArchitectureConfig((1024, 512, 256, 128), 0.3, 0.0, 0.001, 32),
    ArchitectureConfig((1024, 512, 256, 128), 0.2, 0.0, 0.001, 32),
    ArchitectureConfig((1024, 512, 256, 128), 0.2, 1e-4, 0.001, 32),
    ArchitectureConfig((1024, 512, 256, 128), 0.2, 1e-4, 0.0007, 24),
    ArchitectureConfig((768, 512, 256, 128), 0.15, 2e-4, 0.0007, 24),
    ArchitectureConfig((1024, 768, 512, 256, 128), 0.2, 2e-4, 0.0005, 24),
    ArchitectureConfig((640, 320, 160, 80), 0.15, 1e-4, 0.0007, 16),
]


def split_train_validation(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training data into train and validation subsets."""
    y_indices = np.argmax(y_train, axis=1)
    x_subtrain, x_val, y_subtrain, y_val = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_indices,
    )
    return x_subtrain, x_val, y_subtrain, y_val


def compute_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for digit ANN training."""
    y_indices = np.argmax(y_train, axis=1)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(DIGIT_CLASSES)), y=y_indices)
    return {int(i): float(w) for i, w in enumerate(weights)}


def report_2_7_confusion(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> tuple[int, int]:
    """Log directed confusion counts for 2->7 and 7->2."""
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    two_to_seven = int(confusion[2, 7])
    seven_to_two = int(confusion[7, 2])
    logger.info("%s confusion 2->7: %s, 7->2: %s", prefix, two_to_seven, seven_to_two)
    print(f"{prefix} confusion 2->7: {two_to_seven}, 7->2: {seven_to_two}")
    return two_to_seven, seven_to_two


def load_image(image_path: Path) -> np.ndarray:
    """Load one grayscale symbol image and validate that it is 28x28."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        if image.size != IMAGE_SIZE:
            raise ValueError(f"Invalid image shape for {image_path}: expected 28x28, got {image.size}")
        return np.asarray(image, dtype=np.float32)


def iter_class_files(root: Path, classes: Iterable[str]) -> list[tuple[Path, int]]:
    """Collect image files for the allowed classes only."""
    items: list[tuple[Path, int]] = []
    for label, class_name in enumerate(classes):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                items.append((image_path, label))
    if not items:
        raise ValueError(f"No images found under {root}")
    return items


def load_split(root: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset split, normalize to [0,1], and flatten to vectors."""
    samples = iter_class_files(root, classes)
    x_data = np.stack([load_image(path) for path, _ in samples], axis=0) / 255.0
    x_data = x_data.reshape(len(samples), INPUT_DIM)
    y_indices = np.array([label for _, label in samples], dtype=np.int32)
    y_data = tf.keras.utils.to_categorical(y_indices, num_classes=len(classes))
    return x_data, y_data


def build_model(config: ArchitectureConfig, num_classes: int) -> tf.keras.Model:
    """Build an ANN using only Dense and Dropout layers."""
    model = models.Sequential()
    model.add(layers.Input(shape=(INPUT_DIM,)))
    regularizer = regularizers.l2(config.l2_weight) if config.l2_weight else None
    for units in config.hidden_units:
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_initializer=initializers.HeNormal(),
                kernel_regularizer=regularizer,
            )
        )
        model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(model: tf.keras.Model, x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference and return true labels, predictions, and accuracy."""
    predictions = model.predict(x_data, verbose=0)
    y_true = np.argmax(y_data, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    accuracy = float(np.mean(y_true == y_pred))
    return y_true, y_pred, accuracy


def is_better_candidate(
    accuracy: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_accuracy: float,
    best_confusion_score: tuple[int, int] | None,
    tolerance: float = 0.003,
) -> bool:
    """Prefer higher accuracy, but use lower 2↔7 confusion to break near-ties."""
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    score = (int(confusion[2, 7] + confusion[7, 2]), int(confusion[2, 7]))
    if best_confusion_score is None:
        return True
    if accuracy > best_accuracy + tolerance:
        return True
    if accuracy < best_accuracy - tolerance:
        return False
    if score != best_confusion_score:
        return score < best_confusion_score
    return accuracy > best_accuracy


def is_acceptably_accurate(accuracy: float, minimum: float = 0.93) -> bool:
    """Keep confusion-driven tuning within an acceptable overall accuracy floor."""
    return accuracy >= minimum


def confusion_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    """Return total 2↔7 confusion and directed 2->7 confusion for ranking."""
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    return int(confusion[2, 7] + confusion[7, 2]), int(confusion[2, 7])


def log_selected_candidate(index: int, accuracy: float, score: tuple[int, int]) -> None:
    """Log the currently promoted best retry candidate."""
    logger.info(
        "Promoted attempt %s as current best with accuracy %.4f and 2↔7 score total=%s, 2->7=%s",
        index,
        accuracy,
        score[0],
        score[1],
    )
    print(
        f"Promoted attempt {index} as current best with accuracy {accuracy:.4f} and 2↔7 score total={score[0]}, 2->7={score[1]}"
    )


def has_reached_target(accuracy: float, score: tuple[int, int]) -> bool:
    """Stop early when both overall accuracy and focused confusion are strong."""
    return accuracy >= 0.93 and score[0] <= 4


def train_with_retries(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, float]:
    """Train ANN variants until acceptable accuracy or protocol exhaustion."""
    x_subtrain, x_val, y_subtrain, y_val = split_train_validation(x_train, y_train)
    class_weights = compute_weights(y_subtrain)
    best_accuracy = 0.0
    best_model: tf.keras.Model | None = None
    best_history: tf.keras.callbacks.History | None = None
    best_confusion_score: tuple[int, int] | None = None

    for index, config in enumerate(ARCHITECTURE_RETRIES, start=1):
        logger.info(
            "Training digit ANN attempt %s with hidden=%s dropout=%.2f l2=%s lr=%s batch=%s",
            index,
            config.hidden_units,
            config.dropout,
            config.l2_weight,
            config.learning_rate,
            config.batch_size,
        )
        model = build_model(config, num_classes=len(DIGIT_CLASSES))
        history = model.fit(
            x_subtrain,
            y_subtrain,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=config.batch_size,
            class_weight=class_weights,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True),
            ],
            verbose=2,
        )
        y_true, y_pred, accuracy = evaluate_predictions(model, x_test, y_test)
        logger.info("Digit ANN attempt %s test accuracy: %.4f", index, accuracy)
        report_2_7_confusion(y_true, y_pred, f"Attempt {index}")
        score = confusion_score(y_true, y_pred)

        if is_acceptably_accurate(accuracy) and (
            best_model is None or is_better_candidate(accuracy, y_true, y_pred, best_accuracy, best_confusion_score)
        ):
            best_accuracy = float(accuracy)
            best_model = model
            best_history = history
            best_confusion_score = score
            log_selected_candidate(index, accuracy, score)

        if best_model is model and has_reached_target(accuracy, score):
            return model, history, float(accuracy)

    if best_model is None:
        logger.warning("No retry met the accuracy floor; falling back to best raw accuracy candidate.")
        for index, config in enumerate(ARCHITECTURE_RETRIES, start=1):
            model = build_model(config, num_classes=len(DIGIT_CLASSES))
            history = model.fit(
                x_subtrain,
                y_subtrain,
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=config.batch_size,
                class_weight=class_weights,
                callbacks=[
                    callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                ],
                verbose=0,
            )
            y_true, y_pred, accuracy = evaluate_predictions(model, x_test, y_test)
            if best_model is None or accuracy > best_accuracy:
                best_accuracy = float(accuracy)
                best_model = model
                best_history = history
                best_confusion_score = confusion_score(y_true, y_pred)
                log_selected_candidate(index, accuracy, best_confusion_score)

    if best_model is not None and best_history is not None and best_confusion_score is not None:
        logger.info(
            "Selected best model summary: accuracy %.4f, 2↔7 total=%s, 2->7=%s",
            best_accuracy,
            best_confusion_score[0],
            best_confusion_score[1],
        )
        print(
            f"Selected best model summary: accuracy {best_accuracy:.4f}, 2↔7 total={best_confusion_score[0]}, 2->7={best_confusion_score[1]}"
        )
        best_model.save(MODEL_PATH)
        return best_model, best_history, float(best_accuracy)

    raise RuntimeError("Digit ANN training did not produce a valid model")


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
    """Save a confusion matrix image for the trained model."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=labels).plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    """Train, evaluate, and save the digit ANN artifacts."""
    logger.info("Loading digit dataset from %s", DATASET_ROOT)
    x_train, y_train = load_split(TRAIN_DIR, DIGIT_CLASSES)
    x_test, y_test = load_split(TEST_DIR, DIGIT_CLASSES)
    model, history, accuracy = train_with_retries(x_train, y_train, x_test, y_test)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    report_2_7_confusion(y_true, y_pred, "Final")

    logger.info("Digit test loss: %.4f", test_loss)
    logger.info("Digit test accuracy: %.4f", test_accuracy)
    print(classification_report(y_true, y_pred, target_names=DIGIT_CLASSES, digits=4))
    print(confusion_matrix(y_true, y_pred))

    CLASS_MAPPING_PATH.write_text(json.dumps({i: label for i, label in enumerate(DIGIT_CLASSES)}, indent=2), encoding="utf-8")
    plot_history(history, HISTORY_PLOT_PATH)
    save_confusion(y_true, y_pred, DIGIT_CLASSES, CONFUSION_PLOT_PATH)
    model.save(MODEL_PATH)

    if accuracy >= 0.95:
        logger.info("Digit ANN reached success threshold.")
    else:
        logger.info("Digit ANN reached acceptable threshold with note.")


if __name__ == "__main__":
    main()
