"""Download MiniMax-H3 FL2VA weights and catalogued SafeTensors LoRAs."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import requests

from .config import load_config
from .offline import allow_online_for_download

logger = logging.getLogger(__name__)

# Diffusers FL2VA partition only. Skipping transformer_ref (~62 GB) and the
# original FL2VA/ / Ref2VA/ folders keeps the snapshot on a single H100 box.
FL2VA_ALLOW = (
    "modular_model_index.json",
    "model_index.json",
    "README.md",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
)
FL2VA_IGNORE = (
    "transformer_ref/*",
    "FL2VA/*",
    "Ref2VA/*",
)


def _token(config) -> str | None:
    return config.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def download_base_model(config) -> Path:
    target = config.local_dir
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s FL2VA components → %s", config.model_id, target)
    snapshot_download(
        repo_id=config.model_id,
        local_dir=str(target),
        token=_token(config),
        allow_patterns=list(FL2VA_ALLOW),
        ignore_patterns=list(FL2VA_IGNORE),
        resume_download=True,
    )
    return target


def download_loras(config, *, ids: list[str] | None = None) -> list[Path]:
    downloaded: list[Path] = []
    wanted = set(ids) if ids else None
    for spec in config.catalog:
        if spec.is_base:
            continue
        if wanted is not None and spec.id not in wanted:
            continue
        if not spec.filename or (not spec.repo and not spec.download_url):
            continue
        dest = config.lora_dir / spec.filename
        if dest.exists():
            logger.info("LoRA already present: %s", dest)
            downloaded.append(dest)
            continue
        if spec.download_url:
            logger.info("Downloading %s", spec.download_url)
            with requests.get(spec.download_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                temporary = dest.with_suffix(dest.suffix + ".part")
                digest = hashlib.sha256()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(16 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            digest.update(chunk)
                if spec.sha256 and digest.hexdigest().lower() != spec.sha256.lower():
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"SHA-256 mismatch for {spec.filename}")
                temporary.replace(dest)
            path = dest
        else:
            logger.info("Downloading %s/%s", spec.repo, spec.filename)
            path = hf_hub_download(
                repo_id=spec.repo,
                filename=spec.filename,
                local_dir=str(config.lora_dir),
                token=_token(config),
            )
        downloaded.append(Path(path))
    return downloaded


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="store_true", help="Download MiniMax-H3 FL2VA components")
    parser.add_argument("--loras", action="store_true", help="Download catalogued Turbo LoRAs")
    parser.add_argument("--all", action="store_true", help="Download base model and LoRAs")
    parser.add_argument("--lora-id", action="append", default=[], help="Limit LoRA downloads to these ids")
    args = parser.parse_args(argv)

    allow_online_for_download()
    config = load_config()
    if not (args.base or args.loras or args.all):
        args.all = True
    if args.base or args.all:
        download_base_model(config)
    if args.loras or args.all:
        ids = args.lora_id or None
        if ids is None:
            ids = [spec.id for spec in config.catalog if not spec.is_base]
        download_loras(config, ids=ids)
    print("Done.")


if __name__ == "__main__":
    main()
