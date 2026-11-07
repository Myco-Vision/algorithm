FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Render sets $PORT; default to 5000 for docker compose / local use.
EXPOSE 5000

# 1 worker on purpose: retrain status is held in memory and TensorFlow is RAM-heavy.
CMD ["sh", "-c", "exec gunicorn -w 1 --threads 2 --timeout 300 -b 0.0.0.0:${PORT:-5000} App:app"]
