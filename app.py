import gradio as gr
import tempfile
import os
from moviepy import VideoFileClip
import speech_recognition as sr

def process_video(video_file):
    if not video_file:
        return "No video file uploaded."

    temp_video_path = None
    temp_audio_path = None

    try:
        if isinstance(video_file, dict) and "name" in video_file:
            video_path = video_file["name"]
        elif isinstance(video_file, str) and os.path.exists(video_file):
            video_path = video_file
        else:
            return "Invalid video input format."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        with VideoFileClip(video_path) as video:
            if not video.audio:
                return "No audio track found in the video."
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service: {e}"

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# Gradio interface
with gr.Blocks(theme="monochrome", css="""
.centered-container {
    width: 80vw;
    min-width: 400px;
    max-width: 1400px;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 2.5em;
    margin-bottom: 2.5em;
    background: var(--block-background-fill);
    border-radius: 1.2em;
    box-shadow: 0 0 16px 0 #0001;
    padding: 2em 2em 2em 2em;
}
@media (max-width: 900px) {
    .centered-container {
        width: 98vw;
        padding: 1em 0.5em 1em 0.5em;
    }
}
.transcribe-btn-center {
    display: flex;
    justify-content: center;
    margin-top: 1em;
}
""") as demo:
    with gr.Column(elem_classes="centered-container"):
        gr.Markdown("# Automatic Video Transcriber")
        gr.Markdown("## Upload a video file and click 'Transcribe' to begin.")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                video_input = gr.Video(
                    label="Input Video File (.mp4)",
                    interactive=True,
                    sources=["upload"],
                )
                with gr.Row(elem_classes="transcribe-btn-center"):
                    transcribe_btn = gr.Button("Transcribe", scale=0)
                gr.Markdown("### An online version of this app is available [here](https://huggingface.co/spaces/sam-mata/Lecture-Transcriber).", elem_id="note")
            with gr.Column(scale=1, min_width=320):
                text_output = gr.Textbox(
                    label="Raw Text Output",
                    show_copy_button=True,
                    lines=14,
                    interactive=False,
                )
        transcribe_btn.click(
            fn=process_video,
            inputs=video_input,
            outputs=text_output
        )

demo.launch(max_file_size="200MB")
