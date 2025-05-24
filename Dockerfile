FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /home/appuser/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories in user's home directory
RUN mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/whisper && \
    chown -R appuser:appuser /home/appuser/.cache && \
    chmod -R 755 /home/appuser/.cache

# Switch to non-root user
USER appuser

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV HF_DATASETS_CACHE=/home/appuser/.cache/huggingface
ENV XDG_CACHE_HOME=/home/appuser/.cache

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"] 