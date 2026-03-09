"""
MedGemma Medical Image Analysis — Gradio UI for HF Spaces ZeroGPU (Free GPU)
"""

import os
import gradio as gr
import torch
from transformers import pipeline
from PIL import Image

MODEL_ID = "google/medgemma-1.5-4b-it"
HF_TOKEN = os.environ.get("HF_TOKEN")

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

Remember: This is for informational purposes only. Always consult a qualified healthcare professional."""

pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        torch_dtype=dtype,
        device=device,
        token=HF_TOKEN,
    )
    return pipe

def analyze_image(image, include_cure):
    if image is None:
        return "⚠️ Please upload a medical image first."
    try:
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
        return output[0]["generated_text"][-1]["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(
    title="MedGemma | Medical Image Analysis",
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.Markdown("# 🩺 MedGemma Medical Image Analysis")
    gr.Markdown("**AI-Powered Disease Detection & Treatment Suggestions using Google MedGemma 1.5**")
    gr.HTML("""<div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.4);
        border-radius:12px;padding:1rem;color:#fbbf24;font-size:0.9rem;margin-bottom:1rem;">
        <strong>⚠️ Disclaimer:</strong> For informational purposes only. Always consult a qualified
        healthcare professional. Do not rely on this tool for clinical decisions.</div>""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="📤 Upload Medical Image", type="pil", sources=["upload", "clipboard"])
            include_cure = gr.Checkbox(label="Include disease detection & treatment suggestions", value=True)
            analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
        with gr.Column(scale=1):
            output_text = gr.Markdown(value="Upload a medical image and click **Analyze Image**.")

    analyze_btn.click(fn=analyze_image, inputs=[image_input, include_cure], outputs=output_text, show_progress=True)

    gr.Markdown("---\n*Powered by [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) by Google*")

if __name__ == "__main__":
    demo.launch()
