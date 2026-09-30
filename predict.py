import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mushroom_image_model.h5"
CLASSES_PATH = BASE_DIR / "models" / "class_indices.npy"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5


class PredictionError(Exception):
    """Raised when the image cannot be analyzed safely."""


_TF = None


def get_tensorflow():
    global _TF
    if _TF is None:
        import tensorflow as tf
        from absl import logging as absl_logging

        tf.get_logger().setLevel("ERROR")
        absl_logging.set_verbosity(absl_logging.ERROR)
        _TF = tf
    return _TF


def load_assets():
    if not MODEL_PATH.exists():
        raise PredictionError(f"Model file not found: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise PredictionError(f"Class map file not found: {CLASSES_PATH}")

    tf = get_tensorflow()
    model = tf.keras.models.load_model(MODEL_PATH)
    class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
    labels_map = {value: key for key, value in class_indices.items()}
    return model, labels_map


def preprocess_image(image_path):
    image_path = Path(image_path).expanduser().resolve()

    if not image_path.exists():
        raise PredictionError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise PredictionError(f"Image path is not a file: {image_path}")

    try:
        tf = get_tensorflow()
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
    except Exception as exc:
        raise PredictionError(
            "The uploaded file is not a readable image. Use JPG, PNG, or JPEG."
        ) from exc

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0), image_path


def predict_mushroom(image_path):
    image_batch, resolved_image_path = preprocess_image(image_path)
    model, labels_map = load_assets()

    predictions = model.predict(image_batch, verbose=0)[0]  # shape (3,) — softmax probs
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    label = labels_map.get(predicted_index, "unknown")

    return {
        "ok": True,
        "image": str(resolved_image_path),
        "label": label,
        "is_poisonous": label.lower() == "poisonous",
        "is_mushroom": label.lower() != "not_mushroom",
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "raw_score": [round(float(p), 6) for p in predictions],
        "threshold": THRESHOLD,
        "warning": "Do not eat wild mushrooms based only on AI prediction.",
    }


def print_human_result(result):
    label = result["label"].upper()
    confidence = result["confidence_percent"]

    print("=" * 45)
    print(f"RESULT: This looks like {label}.")
    print(f"CONFIDENCE: {confidence:.2f}%")
    print(f"RAW SCORE: {result['raw_score']}")
    print(result["warning"])
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(
        description="Predict if an image is an edible mushroom, poisonous mushroom, or not a mushroom."
    )
    parser.add_argument("image", nargs="?", default=str(BASE_DIR / "test_mushroom.jpg"))
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    args = parser.parse_args()

    try:
        result = predict_mushroom(args.image)
    except PredictionError as exc:
        error = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(error))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result))
    else:
        print_human_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())