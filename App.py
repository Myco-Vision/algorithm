import hmac
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

from predict import predict_mushroom, PredictionError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # reject oversized uploads (100 MB)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ── Retrain state (shared between threads) ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "retrain_log.json"

_DEFAULT_RETRAIN_KEY = "myco-retrain-secret"
RETRAIN_KEY = os.environ.get("RETRAIN_API_KEY", _DEFAULT_RETRAIN_KEY)
if RETRAIN_KEY == _DEFAULT_RETRAIN_KEY:
    print("WARNING: RETRAIN_API_KEY is not set; using the built-in default key. "
          "Set it before deploying.", file=sys.stderr)

RETRAIN_TIMEOUT = int(os.environ.get("RETRAIN_TIMEOUT_SECONDS", "3600"))

_retrain_lock = Lock()
_retrain_status = {
    "state":       "idle",          # idle | running | done | failed
    "started_at":  None,
    "finished_at": None,
    "error":       None,
    "last_result": None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def allowed_file(filename):
    return bool(filename) and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_webp(file) -> str:
    """Save uploaded file as a temp .webp and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp:
        tmp_path = tmp.name

    image = Image.open(file).convert("RGB")
    image.save(tmp_path, "WEBP", quality=85)
    return tmp_path


# ── Classify endpoint ──────────────────────────────────────────────────────────

@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"message": "No image provided."}), 400

    file = request.files["image"]

    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type. Use JPG, PNG, or WEBP."}), 422

    tmp_path = None

    try:
        tmp_path = convert_to_webp(file)
        result = predict_mushroom(tmp_path)

    except PredictionError as exc:
        return jsonify({"message": str(exc)}), 422
    except Exception as exc:
        return jsonify({"message": "Classification failed.", "detail": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not result["is_mushroom"]:
        return jsonify(
            {
                "result_name":           "Not a mushroom",
                "result_classification": "not_mushroom",
                "confidence":            result["confidence_percent"],
                "confidence_level":      result["confidence_percent"],
                "warning":               "No mushroom detected in the image.",
            }
        ), 200

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
    key = request.headers.get("X-Retrain-Key") or request.args.get("key") or ""
    if not hmac.compare_digest(key.encode(), RETRAIN_KEY.encode()):
        return jsonify({"message": "Unauthorized."}), 401
    return None


def _run_retrain_in_background(tmp_dir: str, manifest_path: str):
    """Executed in a background thread — calls retrain.py as a subprocess."""
    retrain_script = BASE_DIR / "retrain.py"

    try:
        result = subprocess.run(
            [sys.executable, str(retrain_script), "--manifest", manifest_path],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=RETRAIN_TIMEOUT,
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
                    "finished_at": now_iso(),
                    "error":       None,
                    "last_result": last_log,
                })
        else:
            with _retrain_lock:
                _retrain_status.update({
                    "state":       "failed",
                    "finished_at": now_iso(),
                    "error":       (result.stderr or "")[-1000:] or "Unknown error",
                    "last_result": None,
                })

    except Exception as exc:
        with _retrain_lock:
            _retrain_status.update({
                "state":       "failed",
                "finished_at": now_iso(),
                "error":       str(exc),
                "last_result": None,
            })
    finally:
        # Remove the uploaded images and manifest
        shutil.rmtree(tmp_dir, ignore_errors=True)


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

    images = request.files.getlist("images[]")
    labels = request.form.getlist("labels[]")

    if not images:
        return jsonify({"message": "No images provided."}), 400
    if len(images) != len(labels):
        return jsonify({"message": "images[] and labels[] counts must match."}), 400

    # Reserve the "running" slot atomically so two requests can't both start a job
    with _retrain_lock:
        if _retrain_status["state"] == "running":
            return jsonify({
                "message": "A retrain job is already in progress.",
                "state":   "running",
            }), 409
        previous_status = dict(_retrain_status)
        _retrain_status.update({
            "state":       "running",
            "started_at":  now_iso(),
            "finished_at": None,
            "error":       None,
            "last_result": None,
        })

    # ── Save uploaded images and build the manifest ────────────────────────
    tmp_dir      = tempfile.mkdtemp(prefix="myco_retrain_")
    manifest     = []
    saved_counts = {"edible": 0, "poisonous": 0}

    try:
        for idx, (img_file, label) in enumerate(zip(images, labels)):
            label = label.lower().strip()
            if label not in ("edible", "poisonous"):
                continue
            if not allowed_file(img_file.filename):
                continue

            # secure_filename blocks path traversal; the index prefix avoids name clashes
            safe_name = f"{idx}_{secure_filename(img_file.filename)}"
            dest = os.path.join(tmp_dir, safe_name)
            img_file.save(dest)
            manifest.append({"path": dest, "label": label})
            saved_counts[label] += 1

        if not manifest:
            raise ValueError("No valid images after filtering.")

        manifest_file = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f)

        thread = Thread(
            target=_run_retrain_in_background,
            args=(tmp_dir, manifest_file),
            daemon=True,
        )
        thread.start()

    except Exception as exc:
        # Nothing was started: clean up and restore the previous status
        shutil.rmtree(tmp_dir, ignore_errors=True)
        with _retrain_lock:
            _retrain_status.update(previous_status)
        status_code = 400 if isinstance(exc, ValueError) else 500
        return jsonify({"message": str(exc)}), status_code

    return jsonify({
        "message":          "Retraining started.",
        "state":            "running",
        "edible_queued":    saved_counts["edible"],
        "poisonous_queued": saved_counts["poisonous"],
        "total_queued":     len(manifest),
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
        "current": status,
        "history": history,
    }), 200


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Debug mode exposes an interactive debugger, so it's opt-in: set FLASK_DEBUG=1 for local dev.
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host="0.0.0.0",
        port=5000,
    )