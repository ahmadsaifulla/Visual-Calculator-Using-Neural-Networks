from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import requests
from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:5050"


@dataclass
class Case:
    endpoint: str
    expected: str | int
    kind: str


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def blank() -> Image.Image:
    return Image.new("L", (28, 28), 255)


def digit_zero() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.ellipse((7, 4, 21, 24), outline=0, width=3)
    return image


def digit_one() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.line((14, 4, 14, 24), fill=0, width=3)
    return image


def digit_two() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.arc((7, 4, 21, 15), start=180, end=360, fill=0, width=3)
    draw.line((21, 14, 7, 24), fill=0, width=3)
    draw.line((7, 24, 21, 24), fill=0, width=3)
    return image


def digit_three() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.arc((7, 4, 21, 15), start=200, end=20, fill=0, width=3)
    draw.arc((7, 13, 21, 24), start=200, end=20, fill=0, width=3)
    return image


def digit_seven() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.line((7, 5, 21, 5), fill=0, width=3)
    draw.line((21, 5, 10, 24), fill=0, width=3)
    return image


def op_plus() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.line((14, 6, 14, 22), fill=0, width=3)
    draw.line((6, 14, 22, 14), fill=0, width=3)
    return image


def op_minus() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.line((6, 14, 22, 14), fill=0, width=3)
    return image


def op_dot() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 12, 16, 16), outline=0, fill=0)
    return image


def op_division_slash() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.line((7, 22, 21, 6), fill=0, width=3)
    return image


def op_division_obelus() -> Image.Image:
    image = blank()
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 5, 16, 9), outline=0, fill=0)
    draw.line((7, 14, 21, 14), fill=0, width=3)
    draw.ellipse((12, 19, 16, 23), outline=0, fill=0)
    return image


def predict(endpoint: str, image: Image.Image) -> dict:
    payload = {"image": to_data_url(image)}
    response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def run() -> None:
    cases = [
        (Case("/predict_digit", 0, "digit"), digit_zero()),
        (Case("/predict_digit", 1, "digit"), digit_one()),
        (Case("/predict_digit", 2, "digit"), digit_two()),
        (Case("/predict_digit", 3, "digit"), digit_three()),
        (Case("/predict_digit", 7, "digit"), digit_seven()),
        (Case("/predict_operator", "+", "operator"), op_plus()),
        (Case("/predict_operator", "-", "operator"), op_minus()),
        (Case("/predict_operator", "·", "operator"), op_dot()),
        (Case("/predict_operator", "÷", "operator"), op_division_slash()),
        (Case("/predict_operator", "÷", "operator"), op_division_obelus()),
    ]

    passed = 0
    for idx, (case, image) in enumerate(cases, start=1):
        data = predict(case.endpoint, image)
        field = "digit" if case.kind == "digit" else "operator"
        prediction = data[field]
        confidence = data.get("confidence", 0.0)
        ok = prediction == case.expected
        passed += int(ok)
        print(
            f"{idx:02d}. {case.endpoint} expected={case.expected!r} got={prediction!r} "
            f"confidence={confidence:.4f} {'PASS' if ok else 'FAIL'}"
        )

    print(f"\nSummary: {passed}/{len(cases)} exact matches")


if __name__ == "__main__":
    run()
