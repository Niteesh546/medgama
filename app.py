"""
MedGemma Medical Image Analysis Web Application
Analyzes medical images (X-rays, dermatology, etc.) for disease detection and treatment suggestions.
"""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Model loading - lazy load to allow server to start
pipe = None
MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-1.5-4b-it")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL_PATH = Path(__file__).parent

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


def load_model():
    """Load MedGemma model - supports both GPU and CPU. Downloads from HF Hub if needed."""
    global pipe
    if pipe is not None:
        return pipe

    try:
        import torch
        from transformers import pipeline
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Try loading from local path first, fall back to HF Hub
        local_safetensors = list(MODEL_PATH.glob("*.safetensors"))
        if local_safetensors:
            model_source = str(MODEL_PATH)
            print(f"Loading MedGemma model from local path {MODEL_PATH} on {device}...")
        else:
            model_source = MODEL_ID
            print(f"Loading MedGemma model from HF Hub ({MODEL_ID}) on {device}...")

        pipe = pipeline(
            "image-text-to-text",
            model=model_source,
            torch_dtype=torch_dtype,
            device=device,
            token=HF_TOKEN,
        )
        print("Model loaded successfully!")
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


app = FastAPI(
    title="MedGemma Medical Image Analyzer",
    description="Analyze medical images for disease detection and treatment suggestions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    success: bool
    analysis: str
    prompt_used: str
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>MedGemma API</h1><p>Frontend not found. Use /api/analyze endpoint.</p>")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    include_cure: bool = True,
):
    """
    Analyze an uploaded medical image for disease detection and treatment suggestions.
    Supports: X-rays, dermatology, ophthalmology, pathology images.
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: JPEG, PNG, WebP, BMP. Got: {file.content_type}",
        )

    try:
        from PIL import Image

        contents = await file.read()
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

        output = model_pipe(text=messages, max_new_tokens=2048)
        analysis_text = output[0]["generated_text"][-1]["content"]

        return AnalysisResponse(
            success=True,
            analysis=analysis_text,
            prompt_used="Disease detection + Treatment suggestions" if include_cure else "General analysis",
        )

    except Exception as e:
        return AnalysisResponse(
            success=False,
            analysis="",
            prompt_used="",
            error=str(e),
        )


@app.get("/api/health")
async def health_check():
    """Check if the API and model are ready."""
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "model_loaded": False, "error": str(e)}


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
