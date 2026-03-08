# MedGemma Medical Image Analysis Web App
# Optimized for Hugging Face Spaces free CPU tier

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only app code (NOT model weights - they download at runtime via HF_TOKEN)
COPY app.py .
COPY static/ ./static/

EXPOSE 7860

# HF Spaces uses port 7860 by default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "300"]
