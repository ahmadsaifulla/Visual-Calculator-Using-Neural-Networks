from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from flask import Flask, jsonify, render_template, request


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DIGIT_MODEL_PATH = BASE_DIR / "digit_model_ann.keras"
OPERATOR_MODEL_PATH = BASE_DIR / "operator_model_ann.keras"
DIGIT_MAPPING_PATH = BASE_DIR / "digit_class_mapping.json"
OPERATOR_MAPPING_PATH = BASE_DIR / "operator_class_mapping.json"
IMAGE_SIZE = (28, 28)
INPUT_DIM = 784

app = Flask(__name__)


def load_mapping(mapping_path: Path) -> dict[int, str]:
    """Load class mapping JSON and convert keys to integers."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in mapping.items()}


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image payload into a grayscale PIL image."""
    if not image_data:
        raise ValueError("Could not recognize symbol")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:
        raise ValueError("Invalid image data") from exc
    return image


def preprocess_for_ann(image: Image.Image) -> np.ndarray:
    """Normalize an image to a centered flattened ANN input vector of length 784."""
    image = image.convert("L")
    image_array = np.asarray(image, dtype=np.float32)
    mean_value = float(image_array.mean())
    if mean_value < 127.5:
        image = ImageOps.invert(image)
        image_array = np.asarray(image, dtype=np.float32)

    ink_mask = image_array < 245
    if not np.any(ink_mask):
        raise ValueError("Could not recognize symbol")

    ys, xs = np.where(ink_mask)
    top, bottom = ys.min(), ys.max() + 1
    left, right = xs.min(), xs.max() + 1
    cropped = image.crop((left, top, right, bottom))

    scale = min(20 / cropped.width, 20 / cropped.height)
    resized_width = max(1, int(round(cropped.width * scale)))
    resized_height = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((resized_width, resized_height), Image.Resampling.NEAREST)

    centered = Image.new("L", IMAGE_SIZE, color=255)
    offset_x = (IMAGE_SIZE[0] - resized_width) // 2
    offset_y = (IMAGE_SIZE[1] - resized_height) // 2
    centered.paste(resized, (offset_x, offset_y))

    normalized = np.asarray(centered, dtype=np.float32) / 255.0
    flattened = normalized.reshape(1, INPUT_DIM)
    return flattened


def predict_label(model: tf.keras.Model, mapping: dict[int, str], image_data: str) -> dict[str, Any]:
    """Run ANN inference and return prediction, confidence, and probabilities."""
    image = decode_base64_image(image_data)
    model_input = preprocess_for_ann(image)
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    prediction = mapping[predicted_index]
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": [float(value) for value in probabilities],
    }


def calculate_result(d1: int, op: str, d2: int) -> dict[str, Any]:
    """Calculate arithmetic result for predicted symbols."""
    equation = f"{d1} {op} {d2}"
    if op == "+":
        return {"result": d1 + d2, "equation": equation}
    if op == "-":
        return {"result": d1 - d2, "equation": equation}
    if op == "·":
        return {"result": d1 * d2, "equation": equation}
    if op == "÷":
        if d2 == 0:
            return {"result": "Error: Cannot divide by zero", "equation": f"{d1} ÷ 0"}
        return {"result": d1 / d2, "equation": equation}
    return {"result": "Could not recognize symbol", "equation": equation}


def load_models() -> tuple[tf.keras.Model, tf.keras.Model, dict[int, str], dict[int, str]]:
    """Load both ANN models and their class mappings once at startup."""
    if not DIGIT_MODEL_PATH.exists() or not OPERATOR_MODEL_PATH.exists():
        raise FileNotFoundError("Model files not found. Train ANN models first.")
    digit_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
    operator_model = tf.keras.models.load_model(OPERATOR_MODEL_PATH)
    digit_mapping = load_mapping(DIGIT_MAPPING_PATH)
    operator_mapping = load_mapping(OPERATOR_MAPPING_PATH)
    return digit_model, operator_model, digit_mapping, operator_mapping


digit_model, operator_model, digit_mapping, operator_mapping = load_models()


@app.get("/")
def index() -> str:
    """Render the visual calculator page."""
    return render_template("index.html")


@app.post("/predict_digit")
def predict_digit() -> Any:
    """Predict a digit from a base64 image payload."""
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_label(digit_model, digit_mapping, str(payload.get("image", "")))
        return jsonify(
            {
                "digit": int(result["prediction"]),
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Digit prediction failed: %s", exc)
        return jsonify({"error": "Could not recognize symbol"}), 500


@app.post("/predict_operator")
def predict_operator() -> Any:
    """Predict an operator symbol from a base64 image payload."""
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_label(operator_model, operator_mapping, str(payload.get("image", "")))
        return jsonify(
            {
                "operator": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Operator prediction failed: %s", exc)
        return jsonify({"error": "Could not recognize symbol"}), 500


@app.post("/calculate")
def calculate() -> Any:
    """Calculate the expression result from predicted values."""
    payload = request.get_json(silent=True) or {}
    try:
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
    app.run(host="127.0.0.1", port=5050, debug=False)
