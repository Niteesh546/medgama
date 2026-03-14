"""
MedGamma Medical Image Analysis — FastAPI + Gemini API
Multi-turn agent with conversation history
"""

import os
import base64
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from PIL import Image
import uvicorn

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are MedGamma, an expert AI medical imaging assistant trained to analyze medical images and provide detailed clinical insights.

You can analyze:
- Chest X-rays (pneumonia, cardiomegaly, pleural effusion, fractures)
- Dermatology images (skin lesions, rashes, melanoma)
- Ophthalmology images (fundus, diabetic retinopathy)
- Pathology slides (histopathology, biopsy)
- MRI and CT scans (brain, spine, abdomen)

When analyzing images, provide structured reports with observations, potential conditions, treatment recommendations, urgency level, and follow-up suggestions.
You remember conversation context and answer follow-up questions about the same image.
Always remind users this is for informational purposes only and to consult a healthcare professional."""

app = FastAPI(title="MedGamma")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    image_data: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    response: str = ""
    error: Optional[str] = None

def get_html():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return "<h1>MedGamma</h1>"

@app.get("/", response_class=HTMLResponse)
async def index():
    return get_html()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        return ChatResponse(success=False, error="Service not configured.")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_PROMPT)
        history = []
        for msg in request.history:
            history.append({"role": msg.role, "parts": [msg.content]})
        chat_session = model.start_chat(history=history)
        content = []
        if request.image_data:
            img_bytes = base64.b64decode(request.image_data)
            content.append({"mime_type": "image/png", "data": base64.b64encode(img_bytes).decode()})
        content.append(request.message)
        response = chat_session.send_message(content)
        return ChatResponse(success=True, response=response.text)
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return ChatResponse(success=False, error="Daily API limit reached. Try again tomorrow.")
        return ChatResponse(success=False, error=f"Error: {error_msg}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_cure: bool = True):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Service not configured.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode()
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SYSTEM_PROMPT)
        prompt = "Please analyze this medical image and provide a comprehensive report including: observations, potential conditions with confidence levels, treatment recommendations, urgency level (Emergency/Urgent/Routine), and follow-up suggestions." if include_cure else "Please analyze this medical image and describe your observations and potential findings."
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_b64}
        ])
        return {"success": True, "analysis": response.text, "image_b64": img_b64}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.0-flash"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
