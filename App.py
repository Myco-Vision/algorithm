import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

from flask import Flask, request, jsonify
from PIL import Image

from predict import predict_mushroom, PredictionError

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ── Retrain state (shared between threads) ─────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
LOG_PATH   = BASE_DIR / "retrain_log.json"
RETRAIN_KEY = os.environ.get("RETRAIN_API_KEY", "myco-retrain-secret")

_retrain_lock   = Lock()
_retrain_status = {
    "state":       "idle",          # idle | running | done | failed
    "started_at":  None,
    "finished_at": None,
    "error":       None,
    "last_result": None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_webp(file) -> str:
    """Save uploaded file as a temp .webp and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp:
        tmp_path = tmp.name

    image = Image.open(file).convert("RGB")
    image.save(tmp_path, "WEBP", quality=85)
    return tmp_path


# ── Classify endpoints ─────────────────────────────────────────────────────────

@app.route("/classify", methods=["POST"])
@app.route("/predict", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"message": "No image provided."}), 400

    file = request.files["image"]

    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type. Use JPG, PNG, or WEBP."}), 422

    tmp_path = None

    try:
        tmp_path = convert_to_webp(file)
        result   = predict_mushroom(tmp_path)

    except PredictionError as exc:
        return jsonify({"message": str(exc)}), 422
    except Exception as exc:
        return jsonify({"message": "Classification failed.", "detail": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return jsonify(
        {
            "species":               result["label"],
            "result_name":           result["label"].capitalize(),
            "result_classification": result["label"],
            "confidence":            result["confidence_percent"],
            "confidence_level":      result["confidence_percent"],
            "is_poisonous":          result["is_poisonous"],
            "raw_score":             result["raw_score"],
            "warning":               result["warning"],
        }
    ), 200


# ── Retrain endpoints ──────────────────────────────────────────────────────────

def _verify_retrain_key():
    """Returns None if auth is OK, or a (response, status_code) tuple if not."""
    key = request.headers.get("X-Retrain-Key") or request.args.get("key")
    if key != RETRAIN_KEY:
        return jsonify({"message": "Unauthorized."}), 401
    return None


def _run_retrain_in_background(manifest_path: str):
    """Executed in a background thread — calls retrain.py as a subprocess."""
    global _retrain_status

    retrain_script = BASE_DIR / "retrain.py"

    try:
        result = subprocess.run(
            [sys.executable, str(retrain_script), "--manifest", manifest_path],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )

        if result.returncode == 0:
            # Pull the last log entry written by retrain.py
            last_log = None
            if LOG_PATH.exists():
                try:
                    entries = json.loads(LOG_PATH.read_text())
                    last_log = entries[0] if entries else None
                except Exception:
                    pass

            with _retrain_lock:
                _retrain_status.update({
                    "state":       "done",
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                    "error":       None,
                    "last_result": last_log,
                })
        else:
            with _retrain_lock:
                _retrain_status.update({
                    "state":       "failed",
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                    "error":       result.stderr[-1000:] or "Unknown error",
                    "last_result": None,
                })

    except Exception as exc:
        with _retrain_lock:
            _retrain_status.update({
                "state":       "failed",
                "finished_at": datetime.utcnow().isoformat() + "Z",
                "error":       str(exc),
                "last_result": None,
            })
    finally:
        # Clean up temp manifest
        try:
            os.unlink(manifest_path)
        except Exception:
            pass


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Trigger model retraining.
    Expects: multipart/form-data with pairs of image files and their labels.

    Body fields:
        images[]   — one or more image files
        labels[]   — matching label for each image ("edible" or "poisonous")

    Header:
        X-Retrain-Key: <RETRAIN_API_KEY>
    """
    auth_error = _verify_retrain_key()
    if auth_error:
        return auth_error

    with _retrain_lock:
        if _retrain_status["state"] == "running":
            return jsonify({
                "message": "A retrain job is already in progress.",
                "state":   "running",
            }), 409

    # ── Build manifest from uploaded files ─────────────────────────────────
    images = request.files.getlist("images[]")
    labels = request.form.getlist("labels[]")

    if not images:
        return jsonify({"message": "No images provided."}), 400
    if len(images) != len(labels):
        return jsonify({"message": "images[] and labels[] counts must match."}), 400

    # Save uploaded images to a temp directory
    tmp_dir      = tempfile.mkdtemp(prefix="myco_retrain_")
    manifest     = []
    saved_counts = {"edible": 0, "poisonous": 0}

    for img_file, label in zip(images, labels):
        label = label.lower().strip()
        if label not in ("edible", "poisonous"):
            continue
        if not allowed_file(img_file.filename):
            continue

        dest = os.path.join(tmp_dir, img_file.filename)
        img_file.save(dest)
        manifest.append({"path": dest, "label": label})
        saved_counts[label] += 1

    if not manifest:
        return jsonify({"message": "No valid images after filtering."}), 400

    # Write manifest JSON to a temp file that retrain.py will read
    manifest_file = os.path.join(tmp_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)

    # ── Launch background thread ────────────────────────────────────────────
    with _retrain_lock:
        _retrain_status.update({
            "state":       "running",
            "started_at":  datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
            "error":       None,
            "last_result": None,
        })

    thread = Thread(target=_run_retrain_in_background, args=(manifest_file,), daemon=True)
    thread.start()

    return jsonify({
        "message":         "Retraining started.",
        "state":           "running",
        "edible_queued":   saved_counts["edible"],
        "poisonous_queued": saved_counts["poisonous"],
        "total_queued":    len(manifest),
    }), 202


@app.route("/retrain/status", methods=["GET"])
def retrain_status():
    """
    Return current retrain state and the last completed run's log from retrain_log.json.
    """
    auth_error = _verify_retrain_key()
    if auth_error:
        return auth_error

    with _retrain_lock:
        status = dict(_retrain_status)

    # Also pull full log history
    history = []
    if LOG_PATH.exists():
        try:
            history = json.loads(LOG_PATH.read_text())
        except Exception:
            history = []

    return jsonify({
        "current":  status,
        "history":  history,
    }), 200


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
