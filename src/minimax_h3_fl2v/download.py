"""Download MiniMax-H3 FL2VA weights, catalog LoRAs, and ComfyUI assets."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
import requests

from .assets import (
    DEFAULT_COMFY_ROOT,
    FACE_ASSETS,
    PRUNED_FILES,
    PRUNED_REPO,
    PRUNED_SHA256,
    RIFE_FILE,
    RIFE_REPO,
    RIFE_SHA256,
    UPSCALER_FILE,
    UPSCALER_REPO,
    UPSCALER_SHA256,
)
from .config import LoRASpec, load_config
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

MODEL_TYPES = (
    "pruned",
    "postprocess",
    "backend",
    "base",
    "loras",
    "taomate",
    "silveroxides",
    "dasiwa",
    "lightx2v",
    "12gb",
    "h100",
    "all",
)

LORA_GROUPS = {
    "taomate": ("taomate_fl2va_3step_ema",),
    "silveroxides": ("silveroxides_dareties_pruned_v1",),
    "dasiwa": (
        "dasiwa_multistep_r48_pruned",
        "dasiwa_multistep_r96_pruned",
        "dasiwa_multistep_r144_pruned",
        "dasiwa_multistep_r512_pruned",
    ),
    "lightx2v": (
        "fl2va_turbo_8step_768p",
        "fl2va_turbo_4step_768p",
        "fl2va_turbo_8step",
        "fl2va_turbo_4step_v01",
        "larryvrh_turbo_v4",
    ),
}

TYPE_ALIASES = {
    "silveroxides_dareties_pruned_v1": "silveroxides",
    "taomate_fl2va_3step_ema": "taomate",
}


def _token(config) -> str | None:
    return config.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def download_url_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if expected_size is not None and destination.stat().st_size != expected_size:
            destination.unlink()
        elif expected_sha256:
            try:
                _verify_sha256(destination, expected_sha256)
                logger.info("Already present: %s", destination)
                return destination
            except RuntimeError:
                logger.warning("Checksum mismatch for existing %s; re-downloading", destination)
                destination.unlink()
        else:
            logger.info("Already present: %s", destination)
            return destination

    logger.info("Downloading %s", url)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(16 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    digest.update(chunk)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        actual = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download size mismatch for {destination.name}: expected {expected_size}, got {actual}"
        )
    if expected_sha256 and digest.hexdigest().lower() != expected_sha256.lower():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, got {digest.hexdigest()}"
        )
    temporary.replace(destination)
    return destination


def download_github_file(url: str, destination: Path, expected_size: int) -> Path:
    if destination.is_file() and destination.stat().st_size == expected_size:
        logger.info("Already present: %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, 16 * 1024 * 1024)
    if temporary.stat().st_size != expected_size:
        actual = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download size mismatch for {destination.name}: expected {expected_size}, got {actual}"
        )
    temporary.replace(destination)
    return destination


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
    wanted = set(ids) if ids else {spec.id for spec in config.catalog if not spec.is_base}
    missing: list[str] = []
    for spec in config.catalog:
        if spec.id not in wanted:
            continue
        if spec.is_base:
            continue
        path = _download_lora(config, spec)
        if path is None:
            missing.append(spec.id)
            continue
        downloaded.append(path)
    unknown = wanted - {spec.id for spec in config.catalog}
    if unknown:
        raise KeyError("Unknown LoRA id(s): " + ", ".join(sorted(unknown)))
    if missing:
        raise RuntimeError(
            "No download source for LoRA id(s): "
            + ", ".join(missing)
            + ". Add repo or download_url in configs/loras.yaml."
        )
    return downloaded


def _download_lora(config, spec: LoRASpec) -> Path | None:
    if not spec.filename or (not spec.repo and not spec.download_url):
        logger.warning("Skipping %s: no repo or download_url", spec.id)
        return None
    dest = config.lora_dir / spec.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        _verify_sha256(dest, spec.sha256)
        logger.info("LoRA already present: %s", dest)
        return dest
    if spec.download_url:
        return download_url_file(spec.download_url, dest, expected_sha256=spec.sha256)
    logger.info("Downloading %s/%s", spec.repo, spec.filename)
    path = Path(
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            local_dir=str(config.lora_dir),
            token=_token(config),
        )
    )
    _verify_sha256(path, spec.sha256)
    return path


def download_pruned_models(comfy_root: Path | None = None) -> list[Path]:
    root = (comfy_root or DEFAULT_COMFY_ROOT) / "models"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    downloaded: list[Path] = []
    for filename in PRUNED_FILES:
        destination = root / filename
        if destination.is_file():
            expected = PRUNED_SHA256.get(filename)
            if expected:
                _verify_sha256(destination, expected)
            logger.info("Already present: %s", destination)
            downloaded.append(destination)
            continue
        logger.info("Downloading %s/%s", PRUNED_REPO, filename)
        path = Path(hf_hub_download(repo_id=PRUNED_REPO, filename=filename, local_dir=str(root)))
        expected = PRUNED_SHA256.get(filename)
        _verify_sha256(path, expected)
        downloaded.append(path)
    return downloaded


def download_postprocess_models(comfy_root: Path | None = None) -> list[Path]:
    root = (comfy_root or DEFAULT_COMFY_ROOT) / "models"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    downloaded: list[Path] = []

    upscale_dir = root / "upscale_models"
    upscaler = upscale_dir / UPSCALER_FILE
    if upscaler.is_file():
        _verify_sha256(upscaler, UPSCALER_SHA256)
        logger.info("Already present: %s", upscaler)
    else:
        logger.info("Downloading %s/%s", UPSCALER_REPO, UPSCALER_FILE)
        upscaler = Path(
            hf_hub_download(
                repo_id=UPSCALER_REPO,
                filename=UPSCALER_FILE,
                local_dir=str(upscale_dir),
            )
        )
        _verify_sha256(upscaler, UPSCALER_SHA256)
    downloaded.append(upscaler)

    rife = root / RIFE_FILE
    if rife.is_file():
        _verify_sha256(rife, RIFE_SHA256)
        logger.info("Already present: %s", rife)
    else:
        logger.info("Downloading %s/%s", RIFE_REPO, RIFE_FILE)
        rife = Path(hf_hub_download(repo_id=RIFE_REPO, filename=RIFE_FILE, local_dir=str(root)))
        _verify_sha256(rife, RIFE_SHA256)
    downloaded.append(rife)

    for relative_path, (url, expected_size) in FACE_ASSETS.items():
        downloaded.append(download_github_file(url, root / relative_path, expected_size))
    return downloaded


def known_model_types(config=None) -> tuple[str, ...]:
    cfg = config or load_config()
    catalog_ids = tuple(spec.id for spec in cfg.catalog if not spec.is_base)
    return MODEL_TYPES + catalog_ids


def normalize_type(model_type: str) -> str:
    return TYPE_ALIASES.get(model_type, model_type)


def resolve_types(
    *,
    model_type: str | None,
    base: bool,
    loras: bool,
    all_flag: bool,
    catalog_ids: tuple[str, ...] = (),
) -> list[str]:
    selected: list[str] = []
    if model_type:
        selected.append(normalize_type(model_type))
    if all_flag:
        selected.append("h100")
    if base:
        selected.append("base")
    if loras:
        selected.append("loras")
    if not selected:
        selected.append("12gb")

    expanded: list[str] = []
    for item in selected:
        if item == "backend":
            expanded.extend(["pruned", "postprocess"])
        elif item == "12gb":
            expanded.extend(["pruned", "postprocess", "taomate"])
        elif item == "h100":
            expanded.extend(["base", "loras"])
        elif item == "all":
            expanded.extend(["pruned", "postprocess", "loras"])
        elif item in MODEL_TYPES or item in LORA_GROUPS or item in catalog_ids:
            expanded.append(item)
        else:
            known = ", ".join(MODEL_TYPES + catalog_ids)
            raise ValueError(f"Unknown model type: {item}. Known types: {known}")

    seen: list[str] = []
    for item in expanded:
        if item not in seen:
            seen.append(item)
    return seen


def run_downloads(
    types: list[str],
    *,
    lora_ids: list[str] | None = None,
    config=None,
    comfy_root: Path | None = None,
) -> None:
    cfg = config or load_config()
    catalog_ids = {spec.id for spec in cfg.catalog if not spec.is_base}
    allow_online_for_download()
    for item in types:
        if item == "pruned":
            download_pruned_models(comfy_root)
        elif item == "postprocess":
            download_postprocess_models(comfy_root)
        elif item == "base":
            download_base_model(cfg)
        elif item == "loras":
            download_loras(cfg, ids=lora_ids)
        elif item in LORA_GROUPS:
            download_loras(cfg, ids=list(LORA_GROUPS[item]))
        elif item in catalog_ids:
            download_loras(cfg, ids=[item])
        else:
            raise ValueError(f"Unknown model type: {item}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Download MiniMax-H3 FL2VA models used by this studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model types:
  pruned         FP8 pruned transformer, NVFP4 text encoder, video/audio VAEs
  postprocess    Real-ESRGAN, RIFE 4.25 Lite, CodeFormer, face-detection weights
  backend        pruned + postprocess
  base           full Diffusers FL2VA snapshot (optional, H100 / non-pruned LoRAs)
  loras          every catalogued LoRA with a download source
  taomate        default 12 GB TaoMate Civitai LoRA
  silveroxides   Silveroxides DARE-TIES pruned v1 Hugging Face LoRA
  dasiwa         Dasiwa Civitai ranks 48/96/144/512
  lightx2v       official LightX2V / larryvrh turbo LoRAs
  12gb           pruned + postprocess + TaoMate (default)
  h100           Diffusers base + all catalog LoRAs
  all            pruned + postprocess + all catalog LoRAs

Catalog ids such as taomate_fl2va_3step_ema are also accepted.
""",
    )
    parser.add_argument(
        "--type",
        dest="model_type",
        default=None,
        help="Which model set to download. Defaults to 12gb when no flags are given.",
    )
    parser.add_argument("--base", action="store_true", help="Download MiniMax-H3 FL2VA Diffusers components")
    parser.add_argument("--loras", action="store_true", help="Download catalogued Turbo LoRAs")
    parser.add_argument("--all", action="store_true", help="H100 alias: download Diffusers base and LoRAs")
    parser.add_argument("--lora-id", action="append", default=[], help="Limit LoRA downloads to these catalog ids")
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=None,
        help="ComfyUI root for pruned/postprocess files (default: .runtime/ComfyUI)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    catalog_ids = tuple(spec.id for spec in config.catalog if not spec.is_base)
    types = resolve_types(
        model_type=args.model_type,
        base=args.base,
        loras=args.loras,
        all_flag=args.all,
        catalog_ids=catalog_ids,
    )
    lora_ids = args.lora_id or None
    if lora_ids and "loras" not in types and not any(item in LORA_GROUPS or item in catalog_ids for item in types):
        types.append("loras")
    run_downloads(types, lora_ids=lora_ids, config=config, comfy_root=args.comfy_root)
    print("Done.")


if __name__ == "__main__":
    main()
