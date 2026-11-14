"""
modal_app.py — Serverless Modal Deployment for MycoVision ML
=============================================================
Deploys MycoVision Flask ML service to Modal (https://modal.com/)
with automatic scale-to-zero, fast cold-starts, and GPU support.

Deployment:
    1. pip install modal
    2. modal setup
    3. modal deploy modal_app.py

After deployment, update Laravel .env (and Vercel):
    ML_API_URL=https://<workspace>--mycovision-ml-flask-app.modal.run
"""

import os
import sys
from pathlib import Path
import modal

# ── 1. App Definition ──────────────────────────────────────────────────────────
app = modal.App("mycovision-ml")

BASE_DIR = Path(__file__).resolve().parent

# ── 2. Container Image Specification ───────────────────────────────────────────
image = (
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
    # Copy application code into container /app
    .add_local_file(BASE_DIR / "App.py", remote_path="/app/App.py")
    .add_local_file(BASE_DIR / "predict.py", remote_path="/app/predict.py")
    .add_local_file(BASE_DIR / "retrain.py", remote_path="/app/retrain.py")
    .add_local_file(BASE_DIR / "train_species.py", remote_path="/app/train_species.py")
    .add_local_file(BASE_DIR / "augmentations.py", remote_path="/app/augmentations.py")
    .add_local_file(BASE_DIR / "species_catalog.py", remote_path="/app/species_catalog.py")
    # Bake the trained model weights (~21 MB) directly into /app/models
    .add_local_dir(BASE_DIR / "models", remote_path="/app/models")
    # Bake the base mushroom dataset (127 MB) so incremental retraining has full baseline data
    .add_local_dir(BASE_DIR / "mushroom_dataset", remote_path="/app/mushroom_dataset")
)

# ── 3. Persistent Volume (for Retrained Weights & User Datasets) ────────────────
# Retraining updates model weights; persisting them across serverless container
# recycles is handled via a Modal Volume mounted alongside the models.
models_volume = modal.Volume.from_name("mycovision-models-volume", create_if_missing=True)


# ── 4. Web Endpoint (Inference / Classification API) ───────────────────────────
@app.function(
    image=image,
    cpu=2,
    memory=2048,           # 2 GB RAM (comfortably handles TensorFlow without OOM)
    timeout=600,            # 10 minutes (ensures retraining jobs finish cleanly without timeout)
    volumes={"/app/models_persisted": models_volume},
    secrets=[
        modal.Secret.from_dict({
            "RETRAIN_API_KEY": os.environ.get("RETRAIN_API_KEY", "myco-retrain-secret"),
            "MUSHROOM_CONFIDENCE_THRESHOLD": os.environ.get("MUSHROOM_CONFIDENCE_THRESHOLD", "0.75"),
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
        for f in persisted_dir.glob("*"):
            if f.is_file():
                import shutil
                shutil.copy2(f, target_dir / f.name)

    from App import app as myco_flask
    return myco_flask


# ── 5. Dedicated On-Demand GPU Retraining Function ─────────────────────────────
@app.function(
    image=image,
    gpu="T4",               # NVIDIA T4 GPU: fast fine-tuning, billed only for active seconds
    timeout=600,
    memory=4096,
    volumes={"/app/models_persisted": models_volume},
    secrets=[
        modal.Secret.from_dict({
            "RETRAIN_API_KEY": os.environ.get("RETRAIN_API_KEY", "myco-retrain-secret"),
            "MUSHROOM_CONFIDENCE_THRESHOLD": "0.75",
            "SPECIES_CONFIDENCE_THRESHOLD": "0.65",
            "CALIBRATION_TEMPERATURE": "1.4",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "MPLCONFIGDIR": "/tmp",
            "KERAS_HOME": "/tmp",
        })
    ],
)
def retrain_on_gpu(manifest: list):
    """
    Executes MobileNetV2 retraining on GPU and persists new weights to the Modal Volume.
    """
    import shutil
    os.chdir("/app")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    from retrain import run_retrain

    models_volume.reload()
    result = run_retrain(manifest=manifest)

    # Copy new model outputs to persistent volume and commit
    persisted_dir = Path("/app/models_persisted")
    persisted_dir.mkdir(parents=True, exist_ok=True)
    for model_file in Path("/app/models").glob("*"):
        if model_file.is_file():
            shutil.copy2(model_file, persisted_dir / model_file.name)

    models_volume.commit()
    return result

