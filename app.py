"""
MedGemma Medical Image Analysis — Powered by Google Gemini 2.0 Flash API
Fast, free, no GPU needed!
"""

import os
import base64
import gradio as gr
import google.generativeai as genai
from PIL import Image
import io

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ANALYSIS_PROMPT = """You are an expert medical imaging AI assistant. Analyze this medical image carefully and provide:

1. **Observations**: Describe what you see in the image (anatomy, structures, abnormalities, notable findings)
2. **Potential Conditions**: List any diseases, conditions, or abnormalities you detect with confidence levels
3. **Recommended Actions**: Suggest appropriate next steps, potential treatments, or therapies
4. **Important Note**: Mention if a healthcare professional consultation is advised

Be specific, thorough, and use proper medical terminology. Format your response clearly with each section labeled."""

CURE_PROMPT = """You are an expert medical imaging AI assistant. Based on your analysis of this medical image, provide:

1. **Identified Conditions**: What disease(s) or condition(s) are present? Include severity if detectable.
2. **Treatment Recommendations**: What are the suggested treatments or cures for each condition?
3. **Lifestyle/Preventive Advice**: Any supportive care, lifestyle changes, or preventive measures?
4. **When to Seek Help**: Urgency level (Emergency/Urgent/Routine) and when to consult a healthcare provider
5. **Follow-up**: Recommended follow-up tests or imaging

⚠️ Remember: This is for informational purposes only. Always consult a qualified healthcare professional for diagnosis and treatment."""

# ---------------------------------------------------------------------------
# Analysis function
# ---------------------------------------------------------------------------
def analyze_image(image: Image.Image, include_cure: bool, api_key: str) -> str:
    if image is None:
        return "⚠️ Please upload a medical image first."

    key = api_key.strip() if api_key.strip() else GEMINI_API_KEY
    if not key:
        return "❌ Please provide your Gemini API key. Get one free at https://aistudio.google.com/apikey"

    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        prompt = CURE_PROMPT if include_cure else ANALYSIS_PROMPT

        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": base64.b64encode(img_byte_arr).decode()}
        ])

        return response.text

    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return "❌ Invalid API key. Please check your Gemini API key at https://aistudio.google.com/apikey"
        elif "quota" in error_msg.lower():
            return "❌ API quota exceeded. Free tier allows 1500 requests/day. Try again tomorrow."
        else:
            return f"❌ Error: {error_msg}"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="MedGemma | Medical Image Analysis",
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
) as demo:

    gr.Markdown("# 🩺 MedGamma — Medical Image Analysis")
    gr.Markdown("**AI-Powered Disease Detection & Treatment Suggestions — Powered by Google Gemini 2.0 Flash**")

    gr.HTML("""
    <div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.4);
        border-radius:12px;padding:1rem;color:#fbbf24;font-size:0.9rem;margin-bottom:1rem;">
        <strong>⚠️ Disclaimer:</strong> This tool is for informational purposes only.
        Always consult a qualified healthcare professional for diagnosis, treatment, and medical advice.
        Do not rely on this tool for clinical decisions.
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="🔑 Gemini API Key",
                placeholder="Paste your free Gemini API key here (from aistudio.google.com/apikey)",
                type="password",
                info="Get a free API key at https://aistudio.google.com/apikey (1500 free requests/day)"
            )
            image_input = gr.Image(
                label="📤 Upload Medical Image",
                type="pil",
                sources=["upload", "clipboard"],
            )
            include_cure = gr.Checkbox(
                label="Include disease detection & treatment suggestions",
                value=True,
            )
            analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Markdown(
                value="Upload a medical image and click **🔬 Analyze Image** to get AI-powered analysis.",
                label="📋 Analysis Results",
            )

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, include_cure, api_key_input],
        outputs=output_text,
        show_progress=True,
    )

    gr.Markdown("""
    ---
    ### 📋 Supported Image Types
    - **Chest X-rays** — pneumonia, cardiomegaly, pleural effusion, fractures
    - **Dermatology** — skin lesions, rashes, melanoma detection
    - **Ophthalmology** — fundus images, diabetic retinopathy
    - **Pathology** — histopathology slides, biopsy images
    - **MRI / CT scans** — brain, spine, abdomen
    - **General medical imaging** — JPEG, PNG, WebP, BMP

    ### 🆓 Free API Setup
    1. Go to 👉 [Google AI Studio](https://aistudio.google.com/apikey)
    2. Sign in with Google → Click **Create API key**
    3. Paste it in the **API Key** field above
    4. **Free tier**: 1500 requests/day, no credit card needed!

    *Powered by [Google Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/) • Built with ❤️ by Niteesh*
    """)

if __name__ == "__main__":
    demo.launch()
