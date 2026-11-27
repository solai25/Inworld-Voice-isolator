# Inworld Voice Isolator — Local Gradio App

**Offline, local, GPU-accelerated voice isolation for noisy speech recordings.**

This project packages the
[Inworld Voice Isolator](https://huggingface.co/inworld/voice-isolator)
(v1.0.0, Apache-2.0) into a single-file Gradio application that runs entirely on
local hardware with no cloud APIs.

The app keeps the foreground voice and removes background noise, music, or
competing speech from audio recordings — and reports the removed background so
it can be inspected or reused.

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%203060%20Ti-76b900)
![License](https://img.shields.io/badge/License-Apache--2.0-green)

## Table of Contents

- [Overview](#overview)
- [Model](#model)
- [Features](#features)
- [Hardware](#hardware)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Download the Model](#download-the-model)
- [Usage](#usage)
- [Offline Behavior](#offline-behavior)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The underlying model performs **audio-to-audio voice isolation** at **48 kHz**.
It is trained to:

- preserve the speaker foreground voice (timbre, level, timing, non-verbal sounds),
- remove surrounding noise, music, crowd sound, and competing talkers,
- not attempt restoration of audio the recording itself lacked.

The release ships **two checkpoints** that share one architecture:

| Checkpoint | Target | Mode | Parameters | File |
|---|---|---|---|---|
| `voice-isolator-v1` | Recorded audio, non-latency-sensitive | Offline, whole 8 s chunks | 64.0 M | `denoiser.safetensors` |
| `voice-isolator-v1-realtime` | Live audio / streams | Causal, look-ahead | 58.2 M | `denoiser_realtime.safetensors` |

## Model

- **Source:** `inworld/voice-isolator` ([Hugging Face](https://huggingface.co/inworld/voice-isolator))
- **Task:** Audio-to-Audio (voice isolation / speech enhancement / denoising)
- **License:** Apache-2.0 (weights + code); third-party notices in repo `NOTICE`
- **Sample rate:** 48 kHz (float32 WAV output, original levels preserved)
- **Weights size:** ~0.49 GB total (two safetensors files)
- **Architecture:** STFT (n_fft 2048, hop 480) → band split → dual-path blocks →
  per-band heads (complex mask + residual) → iSTFT

> Note: version 1 performs denoising only. Restoration is planned for version 2.

## Features

- Single-file Gradio UI (`app.py`) — no multi-module package required
- Offline mode and streaming mode via checkpoint selection
- Look-ahead configuration for streaming (trained positions: 0 / 20 / 40 ms)
- Isolated voice and background (removed) audio outputs, both saved to disk
- JSON sidecar per output (`_params.json`) recording mode, look-ahead, duration,
  timing, and device
- Project-local HF cache, model cache, output folder, and logs
- Offline-first runtime (HF/transformers offline env vars set at startup)

## Hardware

- **Target GPU:** NVIDIA RTX 3060 Ti — 8 GB VRAM
- **System RAM:** ~32 GB
- **Tested setup:** Windows 10/11, CUDA PyTorch (`torch 2.6.0+cu124`)

The released checkpoints are small (~64 M parameters, ~256 MB / ~233 MB files)
and fit comfortably within the 8 GB VRAM budget. Inference was smoke-tested on
CUDA. CPU fallback is supported by the underlying library.

## Quick Start

```bat
git clone <your-fork-url> Inworld-Voice-Isolator
cd Inworld-Voice-Isolator

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

download_model.bat

launch.bat
```

Then open:

```text
http://127.0.0.1:7860
```

## Setup

1. **Clone** this repository.
2. **Create venv** inside the project root:
   `python -m venv .venv`
3. **Activate:**
   `call .venv\Scripts\activate.bat`
4. **Install CUDA torch** (required for the RTX 3060 Ti):
   `python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`
5. **Install remaining dependencies:**
   `python -m pip install -r requirements.txt`

## Download the Model

```bat
download_model.bat
```

This runs `scripts/download_model.py`, which downloads both pinned checkpoints
from `inworld/voice-isolator` into `models\inworld__voice-isolator\` and verifies
their SHA-256 checksums.

Manual equivalent:

```bat
python scripts\download_model.py
```

## Usage

1. Launch the app:
   ```bat
   launch.bat
   ```
2. Open <http://127.0.0.1:7860>.
3. Provide an input audio file (upload or microphone). Supported formats
   include WAV / FLAC / OGG / MP3 via libsndfile, with a Gradio decoder fallback.
4. Select the model mode:
   - `offline` — whole-chunk isolation, best quality for recordings.
   - `streaming` — causal stream with look-ahead; tune look-ahead frames.
5. Click **Isolate voice**.
6. Play the isolated voice and the background (removed) audio. Both are saved
   under `outputs/` with a timestamped naming scheme and a `_params.json` sidecar.

## Offline Behavior

After the initial model download, the application is fully offline:

- `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `GRADIO_TEMP_DIR` are pointed into the
  project-local `cache/` directory.
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` are set.
- Telemetry flags are disabled (HF, Gradio, DoNotTrack).
- Weights are loaded from the local `models/` folder (path-based, not hub-id).
- No inference-time network calls are made.

## Project Structure

```text
Inworld-Voice-Isolator/
├── app.py                        # single-file Gradio app (UI + inference + save)
├── launch.bat                    # launcher -> python app.py
├── download_model.bat            # launcher -> scripts/download_model.py
├── requirements.txt              # pinned-ish runtime dependencies
├── README.md
├── .gitignore
├── models/                       # downloaded checkpoints (inworld__voice-isolator/)
├── outputs/                      # generated voice / background wavs + sidecars
├── cache/                        # offline caches (HF hub, torch, gradio temp)
├── logs/                         # local logs
└── scripts/
    └── download_model.py        # one-time model download + sha256 verification
```

## Configuration

| Variable / argument | Default | Description |
|---|---|---|
| `APP_PORT` | `7860` | Gradio server port |
| Model mode | `offline` | `offline` or `streaming` |
| Look-ahead frames | `2` | Streaming look-ahead (0 / 2 / 4 frames ≈ 0 / 20 / 40 ms) |
| Max input duration | `600` s | Rejected above this limit |

## Troubleshooting

- **Checkpoint missing:** run `download_model.bat`, or place the safetensors
  files in `models\inworld__voice-isolator\`.
- **CUDA unavailable:** install the CUDA build of torch (see Setup) and verify
  with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Long input:** keep recordings under 600 seconds for offline isolation.

## License

- **Code and weights:** Apache-2.0 (see `LICENSE` in the upstream repository).
- **Third-party notices:** upstream `NOTICE` file.

## Acknowledgments

- [Inworld AI](https://github.com/inworld-ai) — model authors and maintainers.
- [Hugging Face](https://huggingface.co) — model hosting and tooling.
