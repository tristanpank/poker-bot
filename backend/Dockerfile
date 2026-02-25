FROM python:3.12-slim

WORKDIR /app

# System deps for aiortc (libavformat, libsrtp, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev \
        libsrtp2-dev libopus-dev libvpx-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (separate step for Docker layer caching)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source
COPY backend/ /app/backend/

# Copy training model class definitions (needed for torch.load)
# Checkpoints (.pt files) are mounted at runtime via volumes
COPY training/src/models/ /app/training/src/models/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
