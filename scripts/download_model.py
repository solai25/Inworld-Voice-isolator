"""One-time model download for the Inworld Voice Isolator.

Downloads the pinned checkpoints of inworld/voice-isolator into this
project's models/ folder and verifies their sha256 checksums.

This is the only place the network is used by the project (besides pip).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"
for _folder in (MODELS_DIR, CACHE_DIR):
    _folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hf" / "hub"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPO_ID = "inworld/voice-isolator"
REVISION = "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970"
CHECKPOINTS = {
    "denoiser.safetensors": (
        "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53",
        "voice-isolator-v1",
    ),
    "denoiser_realtime.safetensors": (
        "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f",
        "voice-isolator-v1-realtime",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_model.py",
        description="Download the Inworld Voice Isolator checkpoints locally.",
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()

    dest = MODELS_DIR / "inworld__voice-isolator"
    dest.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
    print(
        args.repo_id
        + " @ "
        + args.revision[:8]
        + " | gated="
        + str(info.gated)
        + " | total files downloaded on first use ~0.49 GB"
    )
    for filename, (expected_sha256, release_name) in CHECKPOINTS.items():
        target = dest / filename
        if target.is_file():
            print("SKIP (already present): " + filename)
            continue
        print("Downloading: " + args.repo_id + "/" + filename)
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(dest),
            allow_patterns=[filename],
        )
        downloaded = dest / filename
        digest = sha256_file(downloaded)
        if digest != expected_sha256:
            raise SystemExit(
                "CHECKSUM MISMATCH for "
                + filename
                + ": expected "
                + expected_sha256
                + ", got "
                + digest
            )
        print("OK: " + filename + " sha256 verified")
    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "checkpoints": {
            release_name: {
                "filename": filename,
                "sha256": expected_sha256,
            }
            for filename, (expected_sha256, release_name) in CHECKPOINTS.items()
        },
    }
    (dest / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("DONE -> " + str(dest))


if __name__ == "__main__":
    sys.exit(main())
