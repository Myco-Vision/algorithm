import hmac
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from flask import Flask, request, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

# Ensure the algorithm directory is in sys.path so imports work regardless of CWD
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load .env file if present
def _load_env_file(path: Path):
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env_file(BASE_DIR / ".env")
_load_env_file(BASE_DIR.parent / ".env")

from predict import predict_mushroom, PredictionError, unload_assets
from retrain import run_retrain

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # reject oversized uploads (100 MB)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ── Retrain state (shared between threads) ─────────────────────────────────────
LOG_PATH = BASE_DIR / "retrain_log.json"

_DEFAULT_RETRAIN_KEY = "myco-retrain-secret"
RETRAIN_KEY = os.environ.get("RETRAIN_API_KEY", _DEFAULT_RETRAIN_KEY)
if RETRAIN_KEY == _DEFAULT_RETRAIN_KEY:
    is_prod = (
        os.environ.get("RENDER") == "true"
        or os.environ.get("FLASK_ENV") == "production"
        or os.environ.get("APP_ENV") == "production"
        or os.environ.get("ENVIRONMENT") == "production"
    )
    if is_prod:
        # Fail fast in production: the default key is public, so /retrain would be open to anyone.
        raise RuntimeError(
            "RETRAIN_API_KEY is not set (or is the built-in default). "
            "Set a strong random value, e.g. `openssl rand -hex 24`."
        )
    else:
        print("WARNING: RETRAIN_API_KEY is not set; using the built-in default development key.", file=sys.stderr)

RETRAIN_TIMEOUT = int(os.environ.get("RETRAIN_TIMEOUT_SECONDS", "3600"))

_retrain_lock = Lock()
_tf_lock = Lock()
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


# ── Health check ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Classify endpoint ──────────────────────────────────────────────────────────

@app.route("/classify", methods=["POST"])
def classify():
    with _retrain_lock:
        if _retrain_status["state"] == "running":
            return jsonify({
                "message": "AI model retraining is in progress. Please try scanning again in a minute."
            }), 503

    if "image" not in request.files:
        return jsonify({"message": "No image provided."}), 400

    file = request.files["image"]

    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type. Use JPG, PNG, or WEBP."}), 422

    tmp_path = None

    try:
        tmp_path = convert_to_webp(file)
        with _tf_lock:
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
            "species":               result["species"],
            "scientific_name":       result["scientific_name"],
            "common_name":           result["common_name"],
            "result_name":           result["result_name"],
            "result_classification": result["label"],
            "confidence":            result["confidence_percent"],
            "confidence_level":      result["confidence_percent"],
            "species_confidence":    result["species_confidence_percent"],
            "is_poisonous":          result["is_poisonous"],
            "raw_score":             result["raw_score"],
            "species_raw_score":     result["species_raw_score"],
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


def _record_log_start(started_at: str, saved_counts: dict):
    """Prepend a 'running' entry to retrain_log.json as soon as a job is queued."""
    try:
        entries = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
        if not isinstance(entries, list):
            entries = []
    except Exception:
        entries = []

    entry = {
        "started_at":       started_at,
        "status":           "running",
        "error":            None,
        "new_edible":       saved_counts.get("edible", 0),
        "new_poisonous":    saved_counts.get("poisonous", 0),
        "new_not_mushroom": saved_counts.get("not_mushroom", 0),
    }
    entries.insert(0, entry)
    try:
        LOG_PATH.write_text(json.dumps(entries[:20], indent=2))
    except Exception:
        pass


def _finalize_failed_log(error_msg: str):
    """Ensure killed/crashed subprocesses still update the 'running' log row to 'failed'."""
    try:
        entries = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
        if not isinstance(entries, list) or not entries:
            return
        if isinstance(entries[0], dict) and entries[0].get("status") == "running":
            entries[0]["status"] = "failed"
            entries[0]["finished_at"] = now_iso()
            entries[0]["error"] = error_msg
            LOG_PATH.write_text(json.dumps(entries[:20], indent=2))
    except Exception:
        pass


def _run_retrain_in_background(tmp_dir: str, manifest: list, started_at: str):
    """Executed in a background thread — unloads inference models and retrains in-process."""
    try:
        with _tf_lock:
            unload_assets()
            last_log = run_retrain(manifest=manifest, started_at=started_at)

        with _retrain_lock:
            _retrain_status.update({
                "state":       "done",
                "finished_at": now_iso(),
                "error":       None,
                "last_result": last_log,
            })

    except Exception as exc:
        err_detail = str(exc)[-1000:] or "Unknown error"
        _finalize_failed_log(err_detail)
        with _retrain_lock:
            _retrain_status.update({
                "state":       "failed",
                "finished_at": now_iso(),
                "error":       err_detail,
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
        labels[]   — matching label for each image ("edible", "poisonous", or "not_mushroom")

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
    saved_counts = {"edible": 0, "poisonous": 0, "not_mushroom": 0}

    try:
        for idx, (img_file, label) in enumerate(zip(images, labels)):
            label = label.lower().strip()
            if label not in ("edible", "poisonous", "not_mushroom"):
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

        _record_log_start(_retrain_status["started_at"], saved_counts)

        thread = Thread(
            target=_run_retrain_in_background,
            args=(tmp_dir, manifest, _retrain_status["started_at"]),
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
        "not_mushroom_queued":  saved_counts["not_mushroom"],
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
