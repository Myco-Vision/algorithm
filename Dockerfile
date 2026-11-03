FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# System libs needed by opencv / mediapipe
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

# 1 worker on purpose: retrain status is held in memory and TensorFlow is RAM-heavy.
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "300", "-b", "0.0.0.0:5000", "App:app"]
