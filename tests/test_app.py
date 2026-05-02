from __future__ import annotations

import base64
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.array([probabilities], dtype=np.float32)

    def predict(self, model_input: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self._probabilities


with mock.patch("tensorflow.keras.models.load_model", side_effect=[FakeModel([1.0] + [0.0] * 9), FakeModel([1.0, 0.0, 0.0, 0.0])]):
    import app


class AppWhiteBoxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.app.test_client()

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def make_line_image(*, bg: int = 255, fg: int = 0) -> Image.Image:
        image = Image.new("L", (280, 280), bg)
        draw = ImageDraw.Draw(image)
        draw.line((140, 40, 140, 240), fill=fg, width=24)
        return image

    def test_load_mapping_converts_keys_to_ints(self) -> None:
        mapping = app.load_mapping(PROJECT_ROOT / "class_mapping.json")
        self.assertEqual(mapping[0], "0")
        self.assertTrue(all(isinstance(key, int) for key in mapping))

    def test_decode_base64_image_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "Please draw or upload an image"):
            app.decode_base64_image("")

    def test_decode_base64_image_rejects_invalid_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid image data"):
            app.decode_base64_image("data:image/png;base64,not-valid")

    def test_ensure_white_background_inverts_dark_image(self) -> None:
        image = self.make_line_image(bg=0, fg=255)
        normalized = app.ensure_white_background(image)
        normalized_array = np.asarray(normalized)
        self.assertGreater(normalized_array.mean(), 127)

    def test_preprocess_to_mnist_tensor_returns_expected_shape_and_range(self) -> None:
        tensor = app.preprocess_to_mnist_tensor(self.make_line_image())
        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_preprocess_to_mnist_tensor_rejects_blank_image(self) -> None:
        blank = Image.new("L", (280, 280), 255)
        with self.assertRaisesRegex(ValueError, "Please draw or upload an image"):
            app.preprocess_to_mnist_tensor(blank)

    def test_predict_with_model_sets_low_confidence_warning(self) -> None:
        fake_model = FakeModel([0.4, 0.3, 0.2, 0.1])
        mapping = {0: "+", 1: "-", 2: "·", 3: "÷"}
        result = app.predict_with_model(fake_model, mapping, self.image_to_base64(self.make_line_image()))
        self.assertEqual(result["prediction"], "+")
        self.assertTrue(result["low_confidence"])
        self.assertEqual(result["warning"], "Low confidence — please redraw more clearly")

    def test_predict_with_model_high_confidence_has_no_warning(self) -> None:
        fake_model = FakeModel([0.95, 0.02, 0.02, 0.01])
        mapping = {0: "+", 1: "-", 2: "·", 3: "÷"}
        result = app.predict_with_model(fake_model, mapping, self.image_to_base64(self.make_line_image()))
        self.assertFalse(result["low_confidence"])
        self.assertIsNone(result["warning"])

    def test_calculate_result_all_supported_operators(self) -> None:
        self.assertEqual(app.calculate_result(7, "+", 3)["result"], 10)
        self.assertEqual(app.calculate_result(7, "-", 3)["result"], 4)
        self.assertEqual(app.calculate_result(7, "·", 3)["result"], 21)
        self.assertEqual(app.calculate_result(8, "÷", 2)["result"], 4.0)

    def test_calculate_result_divide_by_zero(self) -> None:
        result = app.calculate_result(8, "÷", 0)
        self.assertEqual(result["result"], "Error: Cannot divide by zero")

    def test_calculate_result_invalid_operator(self) -> None:
        result = app.calculate_result(8, "x", 2)
        self.assertEqual(result["result"], "Could not recognize symbol")

    def test_index_route_renders_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NeuroCalc", response.data)
        self.assertIn(b"AI-Powered Visual Arithmetic", response.data)

    def test_predict_digit_route_success(self) -> None:
        payload = {"image": self.image_to_base64(self.make_line_image())}
        with mock.patch.object(app, "predict_with_model", return_value={
            "prediction": "7",
            "confidence": 0.99,
            "probabilities": [0.0] * 10,
            "low_confidence": False,
            "warning": None,
        }):
            response = self.client.post("/predict_digit", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["digit"], 7)
        self.assertFalse(data["low_confidence"])

    def test_predict_digit_route_value_error(self) -> None:
        with mock.patch.object(app, "predict_with_model", side_effect=ValueError("Please draw or upload an image")):
            response = self.client.post("/predict_digit", json={"image": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Please draw or upload an image")

    def test_predict_digit_route_unexpected_error(self) -> None:
        with mock.patch.object(app, "predict_with_model", side_effect=RuntimeError("boom")):
            response = self.client.post("/predict_digit", json={"image": "x"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Could not recognize symbol")

    def test_predict_operator_route_success(self) -> None:
        payload = {"image": self.image_to_base64(self.make_line_image())}
        with mock.patch.object(app, "predict_with_model", return_value={
            "prediction": "÷",
            "confidence": 0.95,
            "probabilities": [0.0, 0.0, 0.0, 1.0],
            "low_confidence": False,
            "warning": None,
        }):
            response = self.client.post("/predict_operator", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["operator"], "÷")

    def test_predict_operator_route_value_error(self) -> None:
        with mock.patch.object(app, "predict_with_model", side_effect=ValueError("Invalid image data")):
            response = self.client.post("/predict_operator", json={"image": "bad"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid image data")

    def test_predict_operator_route_unexpected_error(self) -> None:
        with mock.patch.object(app, "predict_with_model", side_effect=RuntimeError("boom")):
            response = self.client.post("/predict_operator", json={"image": "x"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Could not recognize symbol")

    def test_calculate_route_success(self) -> None:
        response = self.client.post("/calculate", json={"d1": 7, "op": "·", "d2": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["result"], 21)

    def test_calculate_route_missing_inputs(self) -> None:
        response = self.client.post("/calculate", json={"d1": 7, "op": "+"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Missing calculation inputs")

    def test_calculate_route_invalid_inputs(self) -> None:
        response = self.client.post("/calculate", json={"d1": "bad", "op": "+", "d2": 2})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid calculation inputs")

    def test_calculate_route_handles_internal_error(self) -> None:
        with mock.patch.object(app, "calculate_result", side_effect=RuntimeError("boom")):
            response = self.client.post("/calculate", json={"d1": 1, "op": "+", "d2": 2})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "Calculation failed")


if __name__ == "__main__":
    unittest.main()
