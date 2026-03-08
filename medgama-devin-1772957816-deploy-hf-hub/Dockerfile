# MedGemma Medical Image Analysis Web App
# Image is large (~12GB+) because it includes the model. GPU recommended at runtime.

FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional, for some transformers deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model (entire repo; model files are in same dir as app.py)
COPY . .

EXPOSE 8000

# Run the app (use --gpus all when running if you have NVIDIA GPU)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
