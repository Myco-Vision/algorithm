"""
modal_app.py — Serverless Modal Deployment for MycoVision ML
=============================================================
Deploys MycoVision Flask ML service to Modal (https://modal.com/)
with automatic scale-to-zero, fast cold-starts, and dedicated GPU retraining.

Architecture:
  1. flask_app (@modal.wsgi_app):
     - Fast CPU server for inference (/predict, /classify) and job dispatching.
     - Never runs training in-process to avoid serverless timeouts.
  2. retrain_on_gpu (@app.function):
     - Dedicated worker function running on NVIDIA T4 GPU.
     - Billed only for active training seconds (~10-15s per run).
     - Persists new model weights and logs directly to Modal Volume.
  3. retrain_state (modal.Dict):
     - Real-time atomic state shared between web endpoints and GPU worker.
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import modal

# ── 1. App Definition ──────────────────────────────────────────────────────────
app = modal.App("mycovision-ml")

BASE_DIR = Path(__file__).resolve().parent

# ── 2. Storage & State ─────────────────────────────────────────────────────────
models_volume = modal.Volume.from_name("mycovision-models-volume", create_if_missing=True)
retrain_state = modal.Dict.from_name("mycovision-retrain-state", create_if_missing=True)

# ── 3. Container Images ────────────────────────────────────────────────────────
# A. Fast CPU image for Flask web app (fast cold starts, inference only)
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow-cpu==2.16.2",
        "numpy<2.0.0",
        "pillow>=10.0.0",
        "scipy>=1.11.0",
        "absl-py",
        "flask>=3.0.0",
        "gunicorn>=21.0.0",
    )
    .add_local_file(BASE_DIR / "App.py", remote_path="/app/App.py", copy=True)
    .add_local_file(BASE_DIR / "predict.py", remote_path="/app/predict.py", copy=True)
    .add_local_file(BASE_DIR / "retrain.py", remote_path="/app/retrain.py", copy=True)
    .add_local_file(BASE_DIR / "augmentations.py", remote_path="/app/augmentations.py", copy=True)
    .add_local_file(BASE_DIR / "species_catalog.py", remote_path="/app/species_catalog.py", copy=True)
    .add_local_file(BASE_DIR / "modal_app.py", remote_path="/app/modal_app.py", copy=True)
    .add_local_dir(BASE_DIR / "models", remote_path="/app/models", copy=True)
)

# B. GPU training image with CUDA acceleration for fine-tuning
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow[and-cuda]==2.16.2",
        "numpy<2.0.0",
        "pillow>=10.0.0",
        "scipy>=1.11.0",
        "absl-py",
    )
    .add_local_file(BASE_DIR / "retrain.py", remote_path="/app/retrain.py", copy=True)
    .add_local_file(BASE_DIR / "train_species.py", remote_path="/app/train_species.py", copy=True)
    .add_local_file(BASE_DIR / "augmentations.py", remote_path="/app/augmentations.py", copy=True)
    .add_local_file(BASE_DIR / "species_catalog.py", remote_path="/app/species_catalog.py", copy=True)
    .add_local_file(BASE_DIR / "modal_app.py", remote_path="/app/modal_app.py", copy=True)
    .add_local_dir(BASE_DIR / "models", remote_path="/app/models", copy=True)
    .add_local_dir(BASE_DIR / "mushroom_dataset", remote_path="/app/mushroom_dataset", copy=True)
)

# ── 4. Dedicated On-Demand GPU Retraining Function ─────────────────────────────
@app.function(
    image=gpu_image,
    gpu="T4",               # NVIDIA T4: fast fine-tuning (~10-15s)
    timeout=600,            # 10 minutes maximum runtime
    memory=4096,            # 4 GB RAM
    volumes={"/app/models_persisted": models_volume},
    secrets=[
        modal.Secret.from_dict({
            "RETRAIN_API_KEY": os.environ.get("RETRAIN_API_KEY", "myco-retrain-secret"),
            "MUSHROOM_CONFIDENCE_THRESHOLD": "0.70",
            "OOD_MIN_CONFIDENCE_THRESHOLD": "0.60",
            "SPECIES_CONFIDENCE_THRESHOLD": "0.65",
            "CALIBRATION_TEMPERATURE": "1.4",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "MPLCONFIGDIR": "/tmp",
            "KERAS_HOME": "/tmp",
        })
    ],
)
def retrain_on_gpu(manifest_items: list, started_at: str | None = None):
    """
    Executes MobileNetV2 retraining on NVIDIA T4 GPU and persists new weights to the Modal Volume.
    manifest_items: list of dicts [{"filename": str, "bytes": bytes, "label": str}]
    """
    os.chdir("/app")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    run_started = started_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    retrain_state["current"] = {
        "state": "running",
        "started_at": run_started,
        "finished_at": None,
        "error": None,
        "last_result": None,
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="gpu_retrain_"))
    try:
        # Reload volume so we have latest baseline logs and models
        try:
            models_volume.reload()
            persisted_log = Path("/app/models_persisted/retrain_log.json")
            if persisted_log.exists():
                shutil.copy2(persisted_log, Path("/app/retrain_log.json"))
        except Exception as vol_err:
            print(f"[WARN] Volume reload note: {vol_err}", flush=True)

        # Write uploaded images to temporary folder and construct manifest
        file_manifest = []
        for idx, item in enumerate(manifest_items):
            fn = item.get("filename") or f"img_{idx}.jpg"
            dest = tmp_dir / f"{idx}_{fn}"
            dest.write_bytes(item["bytes"])
            file_manifest.append({"path": str(dest), "label": item["label"]})

        from retrain import run_retrain
        result = run_retrain(manifest=file_manifest, started_at=run_started)

        # Persist new weights and log to the volume
        persisted_dir = Path("/app/models_persisted")
        persisted_dir.mkdir(parents=True, exist_ok=True)
        for model_file in Path("/app/models").glob("*"):
            if model_file.is_file():
                shutil.copy2(model_file, persisted_dir / model_file.name)
        if Path("/app/retrain_log.json").exists():
            shutil.copy2(Path("/app/retrain_log.json"), persisted_dir / "retrain_log.json")

        models_volume.commit()

        # Update shared state
        finish_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        retrain_state["current"] = {
            "state": "done",
            "started_at": run_started,
            "finished_at": finish_time,
            "error": None,
            "last_result": result,
        }

        # Update history cache
        if Path("/app/retrain_log.json").exists():
            try:
                retrain_state["history"] = json.loads(Path("/app/retrain_log.json").read_text())[:20]
            except Exception:
                pass

        return result

    except Exception as exc:
        finish_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        err_msg = str(exc)
        retrain_state["current"] = {
            "state": "failed",
            "started_at": run_started,
            "finished_at": finish_time,
            "error": err_msg,
            "last_result": None,
        }
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── 5. Web Endpoint (Inference / Classification API) ───────────────────────────
@app.function(
    image=web_image,
    cpu=2,                  # 2 vCPUs for fast inference
    memory=2048,           # 2 GB RAM
    timeout=600,            # 10 minutes
    scaledown_window=15,    # Scale to zero after 15s of inactivity (saves Modal credits)
    volumes={"/app/models_persisted": models_volume},
    secrets=[
        modal.Secret.from_dict({
            "RETRAIN_API_KEY": os.environ.get("RETRAIN_API_KEY", "myco-retrain-secret"),
            "MUSHROOM_CONFIDENCE_THRESHOLD": os.environ.get("MUSHROOM_CONFIDENCE_THRESHOLD", "0.70"),
            "OOD_MIN_CONFIDENCE_THRESHOLD": os.environ.get("OOD_MIN_CONFIDENCE_THRESHOLD", "0.60"),
            "SPECIES_CONFIDENCE_THRESHOLD": os.environ.get("SPECIES_CONFIDENCE_THRESHOLD", "0.65"),
            "CALIBRATION_TEMPERATURE": os.environ.get("CALIBRATION_TEMPERATURE", "1.4"),
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "MPLCONFIGDIR": "/tmp",
            "KERAS_HOME": "/tmp",
        })
    ],
)
@modal.wsgi_app(label="mycovision-ml")
def flask_app():
    """Mounts the Flask application from App.py as a serverless WSGI app."""
    os.chdir("/app")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # If retrained weights exist in the persistent volume, sync them to /app/models
    persisted_dir = Path("/app/models_persisted")
    target_dir = Path("/app/models")
    if persisted_dir.exists():
        try:
            models_volume.reload()
        except Exception:
            pass
        for f in persisted_dir.glob("*"):
            if f.is_file():
                if f.name == "retrain_log.json":
                    shutil.copy2(f, Path("/app/retrain_log.json"))
                else:
                    shutil.copy2(f, target_dir / f.name)

    from App import app as myco_flask
    return myco_flask
