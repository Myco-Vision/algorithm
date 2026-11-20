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
from PIL import Image, ImageOps
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

from predict import predict_mushroom, PredictionError, unload_assets, load_assets, load_species_assets
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
    """Save uploaded file as a temp .webp and return the path, transposing EXIF orientation if present."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp:
        tmp_path = tmp.name

    image = Image.open(file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.save(tmp_path, "WEBP", quality=85)
    return tmp_path


def _verify_retrain_key():
    """Returns None if auth is OK, or a (response, status_code) tuple if not."""
    key = request.headers.get("X-Retrain-Key") or request.args.get("key") or ""
    if not hmac.compare_digest(key.encode(), RETRAIN_KEY.encode()):
        return jsonify({"message": "Unauthorized."}), 401
    return None


# ── Health check ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    if request.args.get("warm") in ("1", "true", "yes"):
        with _retrain_lock:
            is_retraining = _retrain_status["state"] == "running"
        if is_retraining:
            return jsonify({"status": "retraining", "warmed": False}), 200
        try:
            with _tf_lock:
                load_assets()
                species_loaded = load_species_assets() is not None
            return jsonify({"status": "ok", "warmed": True, "species_loaded": species_loaded}), 200
        except Exception as exc:
            return jsonify({"status": "error", "warmed": False, "detail": str(exc)}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/refresh", methods=["POST"])
def refresh_models():
    """Force-reload Stage-1 and Stage-2 inference models on demand."""
    auth_error = _verify_retrain_key()
    if auth_error:
        return auth_error

    with _retrain_lock:
        if _retrain_status["state"] == "running":
            return jsonify({"message": "Retraining is currently in progress."}), 503

    try:
        with _tf_lock:
            unload_assets()
            load_assets()
            species_loaded = load_species_assets() is not None
        return jsonify({
            "status":         "ready",
            "message":        "ML inference models refreshed and ready for new scans.",
            "stage1_loaded":  True,
            "species_loaded": species_loaded,
        }), 200
    except Exception as exc:
        return jsonify({"message": f"Failed to refresh ML models: {exc}"}), 500


# ── Classify endpoint ──────────────────────────────────────────────────────────

def _is_truthy(val) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _run_image_inference(detect_only: bool = False):
    """Handle image upload validation and run either detection-only or full classification."""
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
            result = predict_mushroom(tmp_path, detect_only=detect_only)

    except PredictionError as exc:
        return jsonify({"message": str(exc)}), 422
    except Exception as exc:
        return jsonify({"message": "Classification failed.", "detail": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if detect_only:
        return jsonify(
            {
                "is_mushroom":  bool(result["is_mushroom"]),
                "status_label": result["status_label"],
            }
        ), 200

    if not result["is_mushroom"]:
        return jsonify(
            {
                "result_name":           result.get("result_name", "Not a mushroom"),
                "result_classification": "not_mushroom",
                "confidence":            result["confidence_percent"],
                "confidence_level":      result["confidence_percent"],
                "warning":               result.get("warning", "No mushroom detected in the image."),
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
            "is_confident_species":  result.get("is_confident_species", False),
            "is_uncatalogued":       result.get("is_uncatalogued", False),
            "closest_species":       result.get("closest_species"),
            "species_threshold":     result.get("species_threshold"),
            "is_poisonous":          result["is_poisonous"],
            "raw_score":             result["raw_score"],
            "species_raw_score":     result["species_raw_score"],
            "warning":               result["warning"],
        }
    ), 200


@app.route("/detect", methods=["POST"])
def detect():
    """Check whether the uploaded image subject is a mushroom without running species/edibility classification."""
    return _run_image_inference(detect_only=True)


@app.route("/classify", methods=["POST"])
def classify():
    detect_only = _is_truthy(request.args.get("detect_only")) or _is_truthy(request.form.get("detect_only"))
    return _run_image_inference(detect_only=detect_only)


# ── Retrain endpoints ──────────────────────────────────────────────────────────

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


def _get_modal_module():
    """Retrieve modal_app module if running inside Modal deployment."""
    try:
        import modal_app
        return modal_app
    except Exception:
        return sys.modules.get("modal_app")


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
    annotation_statuses = request.form.getlist("annotation_statuses[]")
    verified_species_list = request.form.getlist("verified_species[]")

    if not images:
        return jsonify({"message": "No images provided."}), 400
    if len(images) != len(labels):
        return jsonify({"message": "images[] and labels[] counts must match."}), 400

    modal_mod = _get_modal_module()
    is_modal = modal_mod is not None and hasattr(modal_mod, "retrain_on_gpu") and hasattr(modal_mod, "retrain_state")

    if is_modal:
        # Atomic lock check via Modal shared Dict
        current = modal_mod.retrain_state.get("current") or {}
        if current.get("state") == "running":
            return jsonify({
                "message": "A retrain job is already in progress on GPU.",
                "state":   "running",
            }), 409

        manifest_items = []
        saved_counts = {"edible": 0, "poisonous": 0, "not_mushroom": 0}

        for idx, (img_file, label) in enumerate(zip(images, labels)):
            # Enforce human-in-the-loop approved status for crowd contributions
            status = (annotation_statuses[idx] if idx < len(annotation_statuses) else "").lower().strip()
            if status and status != "approved":
                continue

            # Prioritize verified species classification to eliminate label noise
            verified_sp = verified_species_list[idx] if idx < len(verified_species_list) else ""
            if verified_sp:
                try:
                    from species_catalog import get_species_info
                    info = get_species_info(str(verified_sp))
                    if info and info.get("classification") in ("edible", "poisonous", "not_mushroom"):
                        label = info["classification"]
                except Exception:
                    pass

            label = label.lower().strip()
            if label not in ("edible", "poisonous", "not_mushroom"):
                continue
            if not allowed_file(img_file.filename):
                continue

            orig_name = img_file.filename or "upload.jpg"
            stem, ext = os.path.splitext(orig_name)
            ext = ext.lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            safe_stem = secure_filename(stem) or "img"
            safe_name = f"{idx}_{safe_stem}{ext}"

            img_bytes = img_file.read()
            manifest_items.append({
                "filename": safe_name,
                "bytes":    img_bytes,
                "label":    label,
            })
            saved_counts[label] += 1

        if not manifest_items:
            return jsonify({"message": "No valid images after filtering."}), 400

        started_at = now_iso()
        modal_mod.retrain_state["current"] = {
            "state":               "running",
            "started_at":          started_at,
            "finished_at":         None,
            "error":               None,
            "last_result":         None,
            "new_edible":          saved_counts["edible"],
            "new_poisonous":       saved_counts["poisonous"],
            "new_not_mushroom":    saved_counts["new_not_mushroom"] if "new_not_mushroom" in saved_counts else saved_counts["not_mushroom"],
        }

        # Dispatch asynchronous GPU training task on Modal
        modal_mod.retrain_on_gpu.spawn(manifest_items, started_at)

        return jsonify({
            "message":          "Retraining dispatched to Modal GPU worker.",
            "state":            "running",
            "edible_queued":    saved_counts["edible"],
            "poisonous_queued": saved_counts["poisonous"],
            "not_mushroom_queued": saved_counts["not_mushroom"],
            "total_queued":     len(manifest_items),
        }), 202

    # ── Fallback for local development / non-Modal environments ────────────
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

    # Save uploaded images and build the manifest
    tmp_dir      = tempfile.mkdtemp(prefix="myco_retrain_")
    manifest     = []
    saved_counts = {"edible": 0, "poisonous": 0, "not_mushroom": 0}

    try:
        for idx, (img_file, label) in enumerate(zip(images, labels)):
            # Enforce human-in-the-loop approved status for crowd contributions
            status = (annotation_statuses[idx] if idx < len(annotation_statuses) else "").lower().strip()
            if status and status != "approved":
                continue

            # Prioritize verified species classification to eliminate label noise
            verified_sp = verified_species_list[idx] if idx < len(verified_species_list) else ""
            if verified_sp:
                try:
                    from species_catalog import get_species_info
                    info = get_species_info(str(verified_sp))
                    if info and info.get("classification") in ("edible", "poisonous", "not_mushroom"):
                        label = info["classification"]
                except Exception:
                    pass

            label = label.lower().strip()
            if label not in ("edible", "poisonous", "not_mushroom"):
                continue
            if not allowed_file(img_file.filename):
                continue

            orig_name = img_file.filename or "upload.jpg"
            stem, ext = os.path.splitext(orig_name)
            ext = ext.lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                ext = ".jpg"
            safe_stem = secure_filename(stem) or "img"
            safe_name = f"{idx}_{safe_stem}{ext}"
            dest = os.path.join(tmp_dir, safe_name)
            img_file.save(dest)
            manifest.append({"path": dest, "label": label, "annotation_status": status, "verified_species": verified_sp})
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

    modal_mod = _get_modal_module()
    if modal_mod and hasattr(modal_mod, "retrain_state"):
        current = modal_mod.retrain_state.get("current") or {"state": "idle"}
        history = modal_mod.retrain_state.get("history")

        # When a training run completes, sync newly trained weights from persistent volume
        if current.get("state") == "done":
            try:
                modal_mod.models_volume.reload()
                persisted_dir = Path("/app/models_persisted")
                target_dir = BASE_DIR / "models"
                if persisted_dir.exists():
                    for f in persisted_dir.glob("*"):
                        if f.is_file() and f.name != "retrain_log.json":
                            shutil.copy2(f, target_dir / f.name)
                unload_assets()
            except Exception:
                pass

        if not history:
            log_candidates = [
                Path("/app/models_persisted/retrain_log.json"),
                LOG_PATH,
            ]
            for lp in log_candidates:
                if lp.exists():
                    try:
                        history = json.loads(lp.read_text())
                        if isinstance(history, list):
                            break
                    except Exception:
                        pass
            history = history if isinstance(history, list) else []

        return jsonify({
            "current": current,
            "history": history[:20],
            "last_result": current.get("last_result"),
        }), 200

    with _retrain_lock:
        status = dict(_retrain_status)

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
