"""
MedGamma Medical Image Analysis — Powered by Google Gemini 2.0 Flash API
"""

import os
import base64
import io
import google.generativeai as genai
from PIL import Image
import gradio as gr

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ANALYSIS_PROMPT = """You are an expert medical imaging AI assistant. Analyze this medical image carefully and provide:

1. **Observations**: Describe what you see in the image (anatomy, structures, abnormalities, notable findings)
2. **Potential Conditions**: List any diseases, conditions, or abnormalities you detect with confidence levels
3. **Recommended Actions**: Suggest appropriate next steps, potential treatments, or therapies
4. **Important Note**: Mention if a healthcare professional consultation is advised

Be specific, thorough, and use proper medical terminology."""

CURE_PROMPT = """You are an expert medical imaging AI assistant. Based on your analysis of this medical image, provide:

1. **Identified Conditions**: What disease(s) or condition(s) are present? Include severity if detectable.
2. **Treatment Recommendations**: Suggested treatments or cures for each condition.
3. **Lifestyle/Preventive Advice**: Supportive care, lifestyle changes, or preventive measures.
4. **When to Seek Help**: Urgency level (Emergency/Urgent/Routine) and when to consult a healthcare provider.
5. **Follow-up**: Recommended follow-up tests or imaging.

⚠️ This is for informational purposes only. Always consult a qualified healthcare professional."""

def analyze_image(image, include_cure):
    if image is None:
        return "⚠️ Please upload a medical image first."
    if not GEMINI_API_KEY:
        return "❌ Service not configured. Please contact the administrator."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        prompt = CURE_PROMPT if include_cure else ANALYSIS_PROMPT
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": base64.b64encode(img_bytes).decode()}
        ])
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return "❌ Daily limit reached. Please try again tomorrow."
        return f"❌ Error: {error_msg}"

with gr.Blocks(
    title="MedGamma | Medical Image Analysis",
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
) as demo:
    gr.Markdown("# 🩺 MedGamma — Medical Image Analysis")
    gr.Markdown("**AI-Powered Disease Detection & Treatment Suggestions — Powered by Google Gemini 2.0 Flash**")
    gr.HTML("""<div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.4);
        border-radius:12px;padding:1rem;color:#fbbf24;font-size:0.9rem;margin-bottom:1rem;">
        <strong>⚠️ Disclaimer:</strong> For informational purposes only.
        Always consult a qualified healthcare professional. Do not rely on this tool for clinical decisions.
    </div>""")
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="📤 Upload Medical Image", type="pil")
            include_cure = gr.Checkbox(label="Include disease detection & treatment suggestions", value=True)
            analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_text = gr.Markdown(value="Upload a medical image and click **🔬 Analyze Image**.")
    analyze_btn.click(fn=analyze_image, inputs=[image_input, include_cure], outputs=[output_text])
    gr.Markdown("---\n*Powered by [Google Gemini 2.0 Flash](https://deepmind.google/technologies/gemini/) • Built with ❤️ by Niteesh*")

demo.launch()
