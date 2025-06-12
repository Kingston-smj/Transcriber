import os
import whisper
import gradio as gr
import torch
import psutil

# --- Check system and choose device ---
def check_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

device = check_device()

# --- Recommend model based on RAM ---
def recommend_model_size():
    mem_gb = psutil.virtual_memory().available / 1e9
    if mem_gb < 4: return "tiny"
    elif mem_gb < 8: return "base"
    elif mem_gb < 16: return "small"
    else: return "medium"

model_size = recommend_model_size()
model = whisper.load_model(model_size, device=device)

# --- Transcription logic ---
def transcribe(audio_path):
    if not audio_path:
        return "No file uploaded", ""

    result = model.transcribe(audio_path)

    text = result["text"].strip()
    lang = result["language"]

    # Format timestamps
    segments = result.get("segments", [])
    timestamped = ""
    for seg in segments:
        s = int(seg['start'])
        e = int(seg['end'])
        txt = seg['text'].strip()
        timestamped += f"[{s//60:02d}:{s%60:02d} - {e//60:02d}:{e%60:02d}] {txt}\n"

    return f"🌍 Language: {lang}\n\n📝 Transcription:\n{text}", timestamped

# --- Gradio UI ---
with gr.Blocks(title="Whisper Transcriber") as app:
    gr.Markdown("# 🎧 Whisper Transcriber")
    gr.Markdown(f"Running on **{device.upper()}**, model: `{model_size}`")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            transcribe_btn = gr.Button("Transcribe")
        with gr.Column():
            out_text = gr.Textbox(label="Transcription")
            out_stamp = gr.Textbox(label="With Timestamps")

    transcribe_btn.click(fn=transcribe, inputs=audio_input, outputs=[out_text, out_stamp])

# --- Entry point for Render ---
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))