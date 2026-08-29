# ============================================================
# ArtJudge Backend — Docker image
# ============================================================
# Build:   docker build -t artjudge-backend .
# Run:     docker run -p 8000:8000 --env-file .env artjudge-backend
# ============================================================

FROM python:3.11-slim

# System deps for Pillow image processing.
RUN apt-get update && \
    (apt-get install -y --no-install-recommends --fix-missing \
    libglib2.0-0 \
    || (sleep 5 && apt-get update && apt-get install -y -f --no-install-recommends \
    libglib2.0-0)) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching).
# --retries and --timeout handle transient pypi.org connectivity issues in Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
