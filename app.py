"""Single-file Gradio app for the Inworld Voice Isolator.

Model: https://huggingface.co/inworld/voice-isolator (Apache-2.0)
Task:  Audio-to-audio voice isolation at 48 kHz (noise removal / background).

All state lives under this project root:
    models/   downloaded checkpoints (inworld__voice-isolator/)
    outputs/  generated voice / background wavs and params sidecars
    cache/    offline caches (HF hub, torch, gradio temp)
    logs/     local logs

The app runs fully offline once the checkpoint is downloaded.
Launch: launch.bat  ->  Gradio at http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

for _folder in (MODELS_DIR, OUTPUTS_DIR, CACHE_DIR, LOGS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR / "hf")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hf" / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["GRADIO_TEMP_DIR"] = str(CACHE_DIR / "gradio")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from inworld import voice_isolation

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
torch.set_num_threads(min(torch.get_num_threads(), 4))

SAMPLE_RATE = 48000
MAX_SECONDS = 600
DEFAULT_LOOKAHEAD_FRAMES = 2
MODEL_SUBDIR = "inworld__voice-isolator"
CHECKPOINT_FILES = {
    "offline": "denoiser.safetensors",
    "streaming": "denoiser_realtime.safetensors",
}
ISOLATORS: dict[str, object] = {}


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_isolator(mode: str, lookahead_frames: int = 0):
    key = mode + ":" + str(int(lookahead_frames or 0))
    cached = ISOLATORS.get(key)
    if cached is not None:
        return cached
    filename = CHECKPOINT_FILES[mode]
    path = MODELS_DIR / MODEL_SUBDIR / filename
    if not path.is_file():
        raise gr.Error(
            "Voice Isolator checkpoint not found: " + str(path)
            + ". Download the weights first: run download_model.bat"
        )
    frames = int(lookahead_frames) if mode == "streaming" else 0
    isolator = voice_isolation.VoiceIsolator(
        str(path),
        device=_device(),
        mode=mode,
        precision="auto",
        lookahead_frames=frames,
    )
    ISOLATORS[key] = isolator
    return isolator


def load_audio_file(path) -> tuple[torch.Tensor, int]:
    source = str(path)
    try:
        wav, sample_rate = voice_isolation.load_audio(source)
        return wav, int(sample_rate)
    except Exception as error:
        logging.info("soundfile read failed, using gradio decoder: %s", error)
    from gradio import processing_utils as _pu

    rate, data = _pu.audio_from_file(source)
    samples = np.asarray(data)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return torch.from_numpy(np.asarray(samples, dtype=np.float32)), int(rate)


def check_duration(source: str, wav: torch.Tensor, sample_rate: int) -> None:
    try:
        info = sf.info(source)
        seconds = float(info.frames) / float(info.samplerate)
    except Exception:
        seconds = float(wav.shape[-1]) / float(sample_rate)
    if seconds > MAX_SECONDS:
        raise gr.Error(
            "Input audio is longer than 600 seconds; keep it under that."
        )


def run_isolation(
    audio_path: str | None,
    mode: str,
    lookahead_frames: int,
) -> list:
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")
    source = str(audio_path)
    wav, sample_rate = load_audio_file(source)
    check_duration(source, wav, sample_rate)
    isolator = get_isolator(mode, int(lookahead_frames or 0))
    started = time.perf_counter()
    voice = isolator.isolate(wav, sample_rate, output_sample_rate=SAMPLE_RATE)
    elapsed = time.perf_counter() - started
    aligned = (
        wav
        if sample_rate == SAMPLE_RATE
        else voice_isolation.resample(wav, sample_rate, SAMPLE_RATE)
    )
    background = aligned - voice
    voice = voice.detach().cpu().float()
    background = background.detach().cpu().float()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    source_stem = Path(source).stem
    voice_path = OUTPUTS_DIR / (stamp + "_" + source_stem + "_voice.wav")
    background_path = OUTPUTS_DIR / (stamp + "_" + source_stem + "_background.wav")
    voice_isolation.save_audio(str(voice_path), voice, SAMPLE_RATE, subtype="FLOAT")
    voice_isolation.save_audio(
        str(background_path), background, SAMPLE_RATE, subtype="FLOAT"
    )
    sidecar = {
        "source": source,
        "mode": mode,
        "lookahead_frames": int(lookahead_frames or 0),
        "checkpoint": CHECKPOINT_FILES[mode],
        "sample_rate": SAMPLE_RATE,
        "duration_s": float(voice.shape[-1]) / float(SAMPLE_RATE),
        "elapsed_s": float(elapsed),
        "device": str(getattr(isolator, "device", "")),
    }
    sidecar_path = OUTPUTS_DIR / (stamp + "_" + source_stem + "_params.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    status = (
        "Isolated voice saved: "
        + voice_path.name
        + " | Background saved: "
        + background_path.name
        + " | Mode: "
        + mode
        + " | Processed in "
        + format(elapsed, ".2f")
        + " s"
    )
    return [str(voice_path), str(background_path), status]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Inworld Voice Isolator", analytics_enabled=False) as demo:
        gr.Markdown("# Voice Isolator")
        gr.Markdown(
            "Offline voice isolation at 48 kHz. Keeps the foreground voice "
            "and removes background noise, music or competing speech. "
            "Model: inworld/voice-isolator (Apache-2.0)."
        )
        with gr.Row():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                format=None,
                label="Input audio",
                elem_id="input",
            )
            with gr.Column():
                mode_input = gr.Dropdown(
                    choices=["offline", "streaming"],
                    value="offline",
                    label="Mode",
                    elem_id="mode",
                )
                lookahead_input = gr.Number(
                    value=DEFAULT_LOOKAHEAD_FRAMES,
                    minimum=0,
                    maximum=8,
                    step=1,
                    label="Look-ahead frames (streaming only)",
                    elem_id="lookahead",
                )
                run_button = gr.Button(
                    "Isolate voice",
                    variant="primary",
                    elem_id="run",
                )
        voice_output = gr.Audio(
            label="Isolated voice",
            interactive=False,
            format="wav",
            buttons=["download"],
            elem_id="voice-out",
        )
        background_output = gr.Audio(
            label="Background (removed)",
            interactive=False,
            format="wav",
            buttons=["download"],
            elem_id="background-out",
        )
        status_markdown = gr.Markdown(elem_id="status")
        run_button.click(
            fn=run_isolation,
            inputs=[audio_input, mode_input, lookahead_input],
            outputs=[voice_output, background_output, status_markdown],
            concurrency_limit=1,
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("APP_PORT", "7860")),
        allowed_paths=[str(OUTPUTS_DIR)],
        max_file_size="300mb",
        inbrowser=True,
    )
