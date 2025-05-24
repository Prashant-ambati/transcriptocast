FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated cache directory in /opt
RUN mkdir -p /opt/cache/huggingface /opt/cache/whisper && \
    chown -R appuser:appuser /opt/cache && \
    chmod -R 777 /opt/cache

# Create a symlink from /.cache to /opt/cache
RUN ln -sf /opt/cache /.cache && \
    chown -h appuser:appuser /.cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/opt/cache/huggingface
ENV HF_HOME=/opt/cache/huggingface
ENV HF_DATASETS_CACHE=/opt/cache/huggingface
ENV HF_HUB_CACHE=/opt/cache/huggingface
ENV XDG_CACHE_HOME=/opt/cache
ENV HF_CACHE_HOME=/opt/cache/huggingface

# Switch to non-root user
USER appuser
WORKDIR /home/appuser/app

# Install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser app.py .

EXPOSE 7860

# Run with explicit cache directory
CMD ["sh", "-c", "mkdir -p /opt/cache/huggingface /opt/cache/whisper && chmod -R 777 /opt/cache && uvicorn app:app --host 0.0.0.0 --port 7860"] 