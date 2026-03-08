# MedGemma Medical Image Analysis Web App

A web application that uses the MedGemma AI model to analyze medical images for disease detection and treatment suggestions. Supports chest X-rays, dermatology, ophthalmology, pathology, and other medical imaging types.

## Features

- **Photo Upload** – Drag & drop or click to upload medical images (JPEG, PNG, WebP, BMP)
- **Disease Detection** – AI analyzes images and identifies potential conditions
- **Treatment Suggestions** – Provides recommended treatments and next steps
- **Modern UI** – Clean, responsive interface with dark theme

## Requirements

- Python 3.10+
- **GPU recommended** (NVIDIA CUDA) for reasonable inference speed (~30–60 seconds per image). CPU is supported but will be slower.
- ~10GB disk space for model files (already in this repo)

**No API keys required.** The app runs fully locally using the model files in this repo.

---

## Step-by-step: Run locally

1. **Open a terminal** and go to the project folder:
   ```bash
   cd path\to\medgemma-1.5-4b-it
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```
   - **Windows:** `venv\Scripts\activate`
   - **Linux/Mac:** `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (First run may take a few minutes; no API keys needed.)

4. **Start the app:**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Open in browser:** go to **http://localhost:8000** (or http://127.0.0.1:8000).

6. **Use the app:** upload a medical image → click **Analyze Image** → wait for the result (30–60 s on GPU, longer on CPU).

---

## Step-by-step: Deploy (Docker)

1. **Install Docker** (and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) if you want GPU support).

2. **Open a terminal** in the project folder:
   ```bash
   cd path\to\medgemma-1.5-4b-it
   ```

3. **Build the image** (large, ~12GB+; may take 10–20 minutes):
   ```bash
   docker build -t medgemma-web .
   ```

4. **Run the container:**
   - **With GPU:**
     ```bash
     docker run -p 8000:8000 --gpus all medgemma-web
     ```
   - **Without GPU (CPU only, slower):**
     ```bash
     docker run -p 8000:8000 medgemma-web
     ```

5. **Open in browser:** **http://localhost:8000** (or your server’s IP/hostname if deployed on a VM).

6. **Production tip:** Put a reverse proxy (e.g. Nginx) in front with HTTPS and keep the app behind it.

---

## Step-by-step: Deploy on a cloud VM (e.g. GCP / AWS / Azure)

1. **Create a VM** with:
   - Ubuntu 22.04 (or similar)
   - NVIDIA GPU (e.g. T4 on GCP, g4dn on AWS) for faster inference
   - At least 16 GB RAM and ~20 GB disk

2. **SSH into the VM** and install Python 3.10+ and (optional) Docker.

3. **Copy the project** to the VM (e.g. `git clone` or `scp` the folder).

4. **On the VM**, follow “Step-by-step: Run locally” (venv, `pip install -r requirements.txt`, then run uvicorn).  
   Or use Docker: build the image and run the container as in “Step-by-step: Deploy (Docker)”.

5. **Open port 8000** in the cloud firewall/security group so the app is reachable.

6. **Start the app** with:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   (Use `--reload` only for development.)

7. **Open in browser:** `http://<your-vm-public-ip>:8000`.

8. **Production:** Use Nginx (or Caddy) as a reverse proxy and add HTTPS (e.g. Let’s Encrypt).

---

## Setup (reference)

- **Virtual environment:** `python -m venv venv` then activate (see Step 2 above).
- **Dependencies:** `pip install -r requirements.txt`.
- **Model weights (if you cloned from GitHub):** This repo does not include the large `.safetensors` model files (GitHub size limit). Download them from [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) (accept the license), then place `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors` in this project folder. If you already have the full model from Hugging Face, you can skip this.

## Run the Web App (reference)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

## Usage

1. Upload a medical image (X-ray, skin condition, fundus, etc.)
2. Optionally toggle "Include disease detection & treatment suggestions"
3. Click **Analyze Image**
4. Wait 30–60 seconds (depending on GPU/CPU)
5. Review the analysis and suggested treatments

## API Endpoints

- `GET /` – Web interface
- `POST /api/analyze` – Analyze an image (multipart form: `file`, query: `include_cure=true|false`)
- `GET /api/health` – Check model status

## Deployment

The app runs fully locally (no API keys). To expose it on a server or cloud:

### Option 1: Docker (recommended)

Build and run (image is large, ~12GB+, because it includes the model):

```bash
docker build -t medgemma-web .
docker run -p 8000:8000 --gpus all medgemma-web
```

Without GPU (CPU only, slower):

```bash
docker run -p 8000:8000 medgemma-web
```

Then open `http://localhost:8000` (or your server’s hostname).

### Option 2: Cloud VM (GPU for speed)

- **GCP**: Create a VM with an NVIDIA GPU (e.g. T4), install Docker or Python, clone repo, run `uvicorn app:app --host 0.0.0.0 --port 8000`. Open port 8000 in firewall.
- **AWS**: Same idea on an EC2 instance with GPU (e.g. g4dn), or use SageMaker for a more managed setup.
- **Azure**: GPU VM (e.g. NC-series), same steps.

Use a process manager (e.g. systemd, or Docker) and HTTPS (reverse proxy like Nginx + Let’s Encrypt) for production.

### Option 3: Hugging Face Spaces

You can deploy the app as a [Gradio](https://gradio.app) or [Streamlit](https://streamlit.io) Space. You’d need to adapt the UI to Gradio/Streamlit and load the model from the Hugging Face Hub (with token for the gated model). GPU Spaces are available for faster inference.

### Important when deploying

- **GPU**: Strongly recommended; CPU inference can take several minutes per image.
- **Memory**: Plan for ~8–16GB RAM (and VRAM if using GPU).
- **HTTPS**: Use a reverse proxy (Nginx, Caddy) with TLS in front of the app.
- **No API keys**: The model runs on your server; no external API keys are required.

---

## Disclaimer

This tool is for **informational purposes only**. MedGemma provides preliminary analysis. Always consult a qualified healthcare professional for diagnosis and treatment. Do not rely on this tool for clinical decisions.

## License

MedGemma use is governed by [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms).
