"""
MedGemma Medical Image Analysis Web Application
Analyzes medical images (X-rays, dermatology, etc.) for disease detection and treatment suggestions.

Fixes applied:
  - Model pre-warmed on startup (no cold-start delay on first request)
  - File size limit enforced (max 10MB)
  - Rate limiting via slowapi (5 requests/min per IP)
  - Async inference timeout (120s) to prevent hung requests
  - include_cure param handled correctly (query string only)
  - Graceful error messages for all failure modes
"""

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-1.5-4b-it")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_PATH = Path(__file__).parent
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024  # default 10 MB
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT_SEC", "120"))        # default 120 s
RATE_LIMIT = os.environ.get("RATE_LIMIT", "5/minute")                          # per IP

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """Analyze this medical image carefully. Please provide:
1. **Observations**: Describe what you see in the image (anatomy, abnormalities, notable findings)
2. **Potential Conditions**: List any diseases, conditions, or abnormalities you detect
3. **Recommended Actions**: Suggest appropriate next steps, potential treatments, or therapies
4. **Important Note**: Mention if a healthcare professional consultation is advised

Be specific and thorough. Format your response clearly with each section labeled."""

CURE_PROMPT = """Based on your analysis of this medical image, provide:
1. **Identified Conditions**: What disease(s) or condition(s) are present?
2. **Treatment Recommendations**: What are the suggested treatments or cures for each condition?
3. **Lifestyle/Preventive Advice**: Any supportive care or preventive measures?
4. **When to Seek Help**: Urgency level and when to consult a healthcare provider

Remember: This is for informational purposes only. Always consult a qualified healthcare professional for diagnosis and treatment."""

# ---------------------------------------------------------------------------
# Model (lazy singleton)
# ---------------------------------------------------------------------------
pipe = None


def load_model():
    """Load MedGemma model — tries local weights first, falls back to HF Hub."""
    global pipe
    if pipe is not None:
        return pipe

    try:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        local_safetensors = list(MODEL_PATH.glob("*.safetensors"))
        if local_safetensors:
            model_source = str(MODEL_PATH)
            print(f"[startup] Loading MedGemma from local path on {device} …")
        else:
            model_source = MODEL_ID
            print(f"[startup] Downloading MedGemma from HF Hub ({MODEL_ID}) on {device} …")
            if not HF_TOKEN:
                raise RuntimeError(
                    "HF_TOKEN env var is required to download the model from Hugging Face. "
                    "Set it and restart, or place the .safetensors files next to app.py."
                )

        pipe = pipeline(
            "image-text-to-text",
            model=model_source,
            torch_dtype=torch_dtype,
            device=device,
            token=HF_TOKEN,
        )
        print("[startup] Model loaded successfully ✓")
        return pipe

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


# ---------------------------------------------------------------------------
# Lifespan — pre-warm model at startup so first request is fast
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the model when the server starts."""
    print("[startup] Pre-warming MedGemma model …")
    try:
        await asyncio.get_event_loop().run_in_executor(None, load_model)
    except Exception as exc:
        print(f"[startup] WARNING: Model pre-warm failed — {exc}")
        print("[startup] The model will be loaded on the first request instead.")
    yield
    # Nothing to clean up on shutdown


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MedGemma Medical Image Analyzer",
    description="Analyze medical images for disease detection and treatment suggestions",
    version="1.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalysisResponse(BaseModel):
    success: bool
    analysis: str
    prompt_used: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>MedGemma API</h1><p>Frontend not found. Use /api/analyze endpoint.</p>"
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
@limiter.limit(RATE_LIMIT)
async def analyze_image(
    request: Request,                        # required by slowapi
    file: UploadFile = File(...),
    include_cure: bool = True,
):
    """
    Analyze an uploaded medical image for disease detection and treatment suggestions.
    Supports: X-rays, dermatology, ophthalmology, pathology images.

    - Max file size: 10 MB (configurable via MAX_UPLOAD_MB env var)
    - Rate limit: 5 requests / minute per IP (configurable via RATE_LIMIT env var)
    - Inference timeout: 120 s (configurable via INFERENCE_TIMEOUT_SEC env var)
    """
    # --- Validate MIME type ---
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: JPEG, PNG, WebP, BMP.",
        )

    # --- Enforce file size limit BEFORE reading the whole file into RAM ---
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model_pipe = load_model()

        prompt = CURE_PROMPT if include_cure else ANALYSIS_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # --- Run inference with a timeout so hung requests don't block the server ---
        loop = asyncio.get_event_loop()
        try:
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: model_pipe(text=messages, max_new_tokens=2048),
                ),
                timeout=INFERENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return AnalysisResponse(
                success=False,
                analysis="",
                prompt_used="",
                error=f"Inference timed out after {INFERENCE_TIMEOUT}s. "
                      "Try a smaller image or check GPU availability.",
            )

        analysis_text = output[0]["generated_text"][-1]["content"]

        return AnalysisResponse(
            success=True,
            analysis=analysis_text,
            prompt_used="Disease detection + Treatment suggestions" if include_cure else "General analysis",
        )

    except HTTPException:
        raise
    except Exception as exc:
        return AnalysisResponse(
            success=False,
            analysis="",
            prompt_used="",
            error=str(exc),
        )


@app.get("/api/health")
async def health_check():
    """Check if the API and model are ready."""
    model_ready = pipe is not None
    return {
        "status": "ok" if model_ready else "degraded",
        "model_loaded": model_ready,
        "model_id": MODEL_ID,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "rate_limit": RATE_LIMIT,
        "inference_timeout_sec": INFERENCE_TIMEOUT,
    }


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
