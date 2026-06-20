import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from predict import predict_mushroom, PredictionError

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_webp(file) -> str:
    """Save uploaded file as a temp .webp and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp:
        tmp_path = tmp.name

    image = Image.open(file).convert(
        "RGB"
    )  # convert to RGB first (handles PNG transparency)
    image.save(
        tmp_path, "WEBP", quality=85
    )  # quality 85 = good balance of size vs clarity
    return tmp_path


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"message": "No image provided."}), 400

    file = request.files["image"]

    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type. Use JPG, PNG, or WEBP."}), 422

    tmp_path = None

    try:
        # Convert to WebP regardless of original format
        tmp_path = convert_to_webp(file)

        result = predict_mushroom(tmp_path)

    except PredictionError as exc:
        return jsonify({"message": str(exc)}), 422
    except Exception as exc:
        return jsonify({"message": "Classification failed.", "detail": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)  # always clean up temp file

    return jsonify(
        {
            "result_name": result["label"].capitalize(),
            "result_classification": result["label"],
            "confidence_level": result["confidence_percent"],
            "is_poisonous": result["is_poisonous"],
            "raw_score": result["raw_score"],
            "warning": result["warning"],
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
