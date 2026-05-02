from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, render_template, request
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
DIGIT_MODEL_PATH = Path(os.getenv("DIGIT_MODEL_PATH", BASE_DIR / "digit_model.keras"))
OPERATOR_MODEL_PATH = Path(os.getenv("OPERATOR_MODEL_PATH", BASE_DIR / "operator_model.keras"))
DIGIT_MAPPING_PATH = Path(os.getenv("DIGIT_MAPPING_PATH", BASE_DIR / "class_mapping.json"))
OPERATOR_MAPPING_PATH = Path(os.getenv("OPERATOR_MAPPING_PATH", BASE_DIR / "operator_class_mapping.json"))
INPUT_SIZE = 28
CONTENT_SIZE = 20
LOW_CONFIDENCE_THRESHOLD = 0.7

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


def load_mapping(mapping_path: Path) -> Dict[int, str]:
    """Load a class mapping JSON file and convert keys to integers."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        raw_mapping = json.load(mapping_file)
    return {int(key): value for key, value in raw_mapping.items()}


def validate_model_artifacts() -> None:
    """Ensure required model and mapping artifacts exist before startup."""
    required_paths = [DIGIT_MODEL_PATH, OPERATOR_MODEL_PATH, DIGIT_MAPPING_PATH, OPERATOR_MAPPING_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {', '.join(missing)}")


def load_models() -> Tuple[tf.keras.Model, tf.keras.Model, Dict[int, str], Dict[int, str]]:
    """Load trained models and class mappings once at startup."""
    validate_model_artifacts()
    digit_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
    operator_model = tf.keras.models.load_model(OPERATOR_MODEL_PATH)
    digit_mapping = load_mapping(DIGIT_MAPPING_PATH)
    operator_mapping = load_mapping(OPERATOR_MAPPING_PATH)
    logger.info("Loaded digit model from %s", DIGIT_MODEL_PATH)
    logger.info("Loaded operator model from %s", OPERATOR_MODEL_PATH)
    return digit_model, operator_model, digit_mapping, operator_mapping


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode a base64 image payload into a PIL image."""
    if not image_data:
        raise ValueError("Please draw or upload an image")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))
        return image.convert("L")
    except Exception as exc:
        raise ValueError("Invalid image data") from exc


def ensure_white_background(image: Image.Image) -> Image.Image:
    """Normalize image polarity so background is white and foreground is dark."""
    image_array = np.asarray(image, dtype=np.uint8)
    mean_intensity = float(image_array.mean())
    if mean_intensity < 127:
        image = ImageOps.invert(image)
    return image


def preprocess_to_mnist_tensor(image: Image.Image) -> np.ndarray:
    """Convert a raw image into a centered MNIST-style model input tensor."""
    image = ensure_white_background(image)
    image_array = np.asarray(image, dtype=np.uint8)

    ink_mask = image_array < 245
    if not np.any(ink_mask):
        raise ValueError("Please draw or upload an image")

    ys, xs = np.where(ink_mask)
    top, bottom = ys.min(), ys.max() + 1
    left, right = xs.min(), xs.max() + 1
    cropped = image.crop((left, top, right, bottom))

    scale = min(CONTENT_SIZE / cropped.width, CONTENT_SIZE / cropped.height)
    resized_width = max(1, int(round(cropped.width * scale)))
    resized_height = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    centered = Image.new("L", (INPUT_SIZE, INPUT_SIZE), color=255)
    offset_x = (INPUT_SIZE - resized_width) // 2
    offset_y = (INPUT_SIZE - resized_height) // 2
    centered.paste(resized, (offset_x, offset_y))

    final_array = np.asarray(centered, dtype=np.float32) / 255.0
    final_array = np.expand_dims(final_array, axis=(0, -1))
    return final_array


def predict_with_model(model: tf.keras.Model, mapping: Dict[int, str], image_data: str) -> Dict[str, Any]:
    """Run preprocessing and inference for a single image input."""
    image = decode_base64_image(image_data)
    model_input = preprocess_to_mnist_tensor(image)
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_label = mapping[predicted_index]

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": [float(value) for value in probabilities],
        "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        "warning": "Low confidence — please redraw more clearly" if confidence < LOW_CONFIDENCE_THRESHOLD else None,
    }


def calculate_result(d1: int, op: str, d2: int) -> Dict[str, Any]:
    """Apply the predicted operator to the predicted digits."""
    equation = f"{d1} {op} {d2}"

    if op == "+":
        result: Any = d1 + d2
    elif op == "-":
        result = d1 - d2
    elif op == "·":
        result = d1 * d2
    elif op == "÷":
        if d2 == 0:
            return {"result": "Error: Cannot divide by zero", "equation": equation}
        result = d1 / d2
    else:
        return {"result": "Could not recognize symbol", "equation": equation}

    return {"result": result, "equation": equation}


try:
    digit_model, operator_model, digit_mapping, operator_mapping = load_models()
except Exception as exc:
    logger.exception("Failed to initialize models: %s", exc)
    raise


@app.get("/")
def index() -> str:
    """Render the visual calculator UI."""
    return render_template("index.html")


@app.post("/predict_digit")
def predict_digit() -> Any:
    """Predict a digit from an uploaded or drawn image."""
    try:
        payload = request.get_json(silent=True) or {}
        result = predict_with_model(digit_model, digit_mapping, payload.get("image", ""))
        return jsonify(
            {
                "digit": int(result["prediction"]),
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "low_confidence": result["low_confidence"],
                "warning": result["warning"],
            }
        )
    except ValueError as exc:
        logger.warning("Digit prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Digit prediction failed: %s", exc)
        return jsonify({"error": "Could not recognize symbol"}), 500


@app.post("/predict_operator")
def predict_operator() -> Any:
    """Predict an operator from an uploaded or drawn image."""
    try:
        payload = request.get_json(silent=True) or {}
        result = predict_with_model(operator_model, operator_mapping, payload.get("image", ""))
        return jsonify(
            {
                "operator": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "low_confidence": result["low_confidence"],
                "warning": result["warning"],
            }
        )
    except ValueError as exc:
        logger.warning("Operator prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Operator prediction failed: %s", exc)
        return jsonify({"error": "Could not recognize symbol"}), 500


@app.post("/calculate")
def calculate() -> Any:
    """Calculate the arithmetic result for predicted symbols."""
    try:
        payload = request.get_json(silent=True) or {}
        d1 = int(payload["d1"])
        d2 = int(payload["d2"])
        op = str(payload["op"])
        return jsonify(calculate_result(d1, op, d2))
    except KeyError:
        return jsonify({"error": "Missing calculation inputs"}), 400
    except ValueError:
        return jsonify({"error": "Invalid calculation inputs"}), 400
    except Exception as exc:
        logger.exception("Calculation failed: %s", exc)
        return jsonify({"error": "Calculation failed"}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", os.getenv("PORT", "5000")))
    debug = os.getenv("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
