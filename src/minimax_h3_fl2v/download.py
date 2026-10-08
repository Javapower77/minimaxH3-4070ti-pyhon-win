"""Download MiniMax-H3 FL2VA weights, catalog LoRAs, and ComfyUI assets."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import hashlib
import logging
import os
import shutil
import sys
import urllib.request
import warnings
from urllib.parse import urlsplit
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download, snapshot_download
import requests

from .assets import (
    CHARACTER_SWAP_LORA_FILE,
    CHARACTER_SWAP_LORA_URL,
    CHARACTER_SWAP_LORA_SHA256,
    CHARACTER_SWAP_LORA_SIZE,
    DEFAULT_COMFY_ROOT,
    ROOT,
    RESTORE_FILE,
    RESTORE_URL,
    RESTORE_SHA256,
    RESTORE_SIZE,
    REF2VA_MODEL_REPO,
    REF2VA_MODEL_REVISION,
    REF2VA_MODEL_FILE,
    REF2VA_MODEL_SHA256,
    REF2VA_MODEL_SIZE,
    REF2VA_TURBO_FILE,
    REF2VA_TURBO_URL,
    REF2VA_TURBO_SHA256,
    REF2VA_TURBO_SIZE,
    LATENT_NODE_BASE_URL,
    LATENT_NODE_FILES,
    LATENT_UPSCALER_FILE,
    LATENT_UPSCALER_URL,
    LATENT_UPSCALER_SHA256,
    LATENT_UPSCALER_SIZE,
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
    "latent_upscaler",
    "restore",
    "restore_base",
    "character_swap",
    "ref2va_turbo",
    "backend",
    "base",
    "loras",
    "taomate",
    "silveroxides",
    "dasiwa",
    "dasiwa_v2",
    "dmad",
    "dmad_hyperflow",
    "dmad_dareties",
    "pdmd_dmad",
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
    "dasiwa_v2": ("dasiwa_multistep_v2_r128_pruned",),
    "dmad": ("dmad_4step_lora_critic",),
    "dmad_hyperflow": ("dasiwa_dmad_hyperflow_4step_r256",),
    "dmad_dareties": ("dmad_full_dareties_v4_step600",),
    "pdmd_dmad": ("dasiwa_pdmd_dmad_4step_r256",),
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
    "dasiwa_multistep_v2_r128_pruned": "dasiwa_v2",
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
    parsed = urlsplit(url)
    token = os.getenv("CIVITAI_API_TOKEN")
    request_options = {}
    if token and parsed.scheme == "https" and parsed.hostname in {"civitai.red", "civitai.com"}:
        # requests strips Authorization when redirecting to a different host.
        request_options["headers"] = {"Authorization": f"Bearer {token}"}
    elif parsed.scheme == "https" and parsed.hostname == "huggingface.co":
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token()
        if hf_token:
            request_options["headers"] = {"Authorization": f"Bearer {hf_token}"}
    with requests.get(url, stream=True, timeout=60, **request_options) as response:
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
    dest = spec.resolved_path(config.lora_dir)
    if dest is not None and dest.is_file():
        _verify_sha256(dest, spec.sha256)
        logger.info("LoRA already present: %s", dest)
        return dest
    if dest is None or not spec.filename or (not spec.repo and not spec.download_url):
        logger.warning("Skipping %s: no repo or download_url", spec.id)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if spec.download_url:
        return download_url_file(spec.download_url, dest, expected_sha256=spec.sha256)
    logger.info("Downloading %s/%s", spec.repo, spec.filename)
    path = Path(
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            local_dir=str(dest.parent),
            token=_token(config),
        )
    )
    _verify_sha256(path, spec.sha256)
    if path.resolve() != dest.resolve():
        shutil.copy2(path, dest)
        return dest
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


def download_latent_upscaler(comfy_root: Path | None = None) -> list[Path]:
    """Install only the optional FP16 checkpoint and pinned MIT inference module."""
    root = comfy_root or DEFAULT_COMFY_ROOT
    model = download_url_file(
        LATENT_UPSCALER_URL, root / "models/latent_upscale_models" / LATENT_UPSCALER_FILE,
        expected_sha256=LATENT_UPSCALER_SHA256, expected_size=LATENT_UPSCALER_SIZE,
    )
    nodes = root / "custom_nodes/minimax_h3_nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "comfy_nodes/minimax_h3_nodes/__init__.py", nodes / "__init__.py")
    paths = [model]
    for filename, (relative, checksum) in LATENT_NODE_FILES.items():
        paths.append(download_url_file(LATENT_NODE_BASE_URL + relative, nodes / filename, expected_sha256=checksum))
    return paths


def download_restore(comfy_root: Path | None = None) -> list[Path]:
    """Download only Nugus's pinned restoration adapter; no base or turbo."""
    root = (comfy_root or DEFAULT_COMFY_ROOT) / "models" / "loras"
    destination = root / RESTORE_FILE
    if destination.exists():
        if destination.stat().st_size != RESTORE_SIZE:
            raise RuntimeError("Existing restoration checkpoint has incorrect size; retained unchanged.")
        _verify_sha256(destination, RESTORE_SHA256)
        return [destination]
    return [download_url_file(RESTORE_URL, root / RESTORE_FILE,
                             expected_sha256=RESTORE_SHA256, expected_size=RESTORE_SIZE)]


def download_character_swap(comfy_root: Path | None = None) -> list[Path]:
    """Download only the pinned character-swap adapter; reuse restore_base separately."""
    root = (comfy_root or DEFAULT_COMFY_ROOT) / "models" / "loras"
    destination = root / CHARACTER_SWAP_LORA_FILE
    if destination.exists():
        if destination.stat().st_size != CHARACTER_SWAP_LORA_SIZE:
            raise RuntimeError("Existing character-swap checkpoint has incorrect size; retained unchanged.")
        _verify_sha256(destination, CHARACTER_SWAP_LORA_SHA256)
        return [destination]
    return [download_url_file(
        CHARACTER_SWAP_LORA_URL, destination,
        expected_sha256=CHARACTER_SWAP_LORA_SHA256,
        expected_size=CHARACTER_SWAP_LORA_SIZE,
    )]


def download_ref2va_turbo(comfy_root: Path | None = None) -> list[Path]:
    """Download only the optional pinned Ref2VA turbo adapter; no base or swap."""
    root = (comfy_root or DEFAULT_COMFY_ROOT) / "models" / "loras"
    destination = root / REF2VA_TURBO_FILE
    if destination.exists():
        if destination.stat().st_size != REF2VA_TURBO_SIZE:
            raise RuntimeError("Existing Ref2VA turbo checkpoint has incorrect size; retained unchanged.")
        _verify_sha256(destination, REF2VA_TURBO_SHA256)
        return [destination]
    return [download_url_file(
        REF2VA_TURBO_URL, destination,
        expected_sha256=REF2VA_TURBO_SHA256,
        expected_size=REF2VA_TURBO_SIZE,
    )]


def download_restore_base(comfy_root: Path | None = None) -> list[Path]:
    """Download only the large pinned Ref2VA FP8-scaled transformer."""
    root = (comfy_root or DEFAULT_COMFY_ROOT) / "models" / "diffusion_models"
    destination = root / REF2VA_MODEL_FILE
    if destination.exists():
        if destination.stat().st_size != REF2VA_MODEL_SIZE:
            raise RuntimeError("Existing Ref2VA checkpoint has incorrect size; retained unchanged.")
        _verify_sha256(destination, REF2VA_MODEL_SHA256)
        return [destination]
    return [download_url_file(
        f"https://huggingface.co/{REF2VA_MODEL_REPO}/resolve/{REF2VA_MODEL_REVISION}"
        f"/diffusion_models/{REF2VA_MODEL_FILE}",
        root / REF2VA_MODEL_FILE,
        expected_sha256=REF2VA_MODEL_SHA256,
        expected_size=REF2VA_MODEL_SIZE,
    )]


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
        elif item == "latent_upscaler":
            download_latent_upscaler(comfy_root)
        elif item == "restore":
            download_restore(comfy_root)
        elif item == "restore_base":
            download_restore_base(comfy_root)
        elif item == "character_swap":
            download_character_swap(comfy_root)
        elif item == "ref2va_turbo":
            download_ref2va_turbo(comfy_root)
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


def _download_providers(types: list[str], config, lora_ids: list[str] | None = None) -> set[str]:
    """Identify providers from the resolved sets and catalog download sources."""
    providers: set[str] = set()
    wanted: set[str] = set()
    for item in types:
        if item == "restore":
            providers.add("civitai")
        elif item in {"base", "pruned", "postprocess", "latent_upscaler", "restore_base", "character_swap", "ref2va_turbo"}:
            providers.add("huggingface")
        elif item == "loras":
            wanted.update(lora_ids if lora_ids else (spec.id for spec in config.catalog if not spec.is_base))
        elif item in LORA_GROUPS:
            wanted.update(LORA_GROUPS[item])
        else:
            wanted.add(item)
    for spec in config.catalog:
        if spec.id not in wanted or spec.is_base:
            continue
        if spec.download_url:
            host = urlsplit(spec.download_url).hostname
            if host in {"civitai.red", "civitai.com"}:
                providers.add("civitai")
            elif host == "huggingface.co":
                providers.add("huggingface")
        elif spec.repo:
            providers.add("huggingface")
    return providers


@contextmanager
def _download_credentials(types: list[str], config, lora_ids=None, *, no_input: bool = False):
    """Prompt once per needed provider; never persist entered credentials."""
    original_hf = config.hf_token
    changed_env: dict[str, str | None] = {}
    try:
        providers = _download_providers(types, config, lora_ids)
        for provider, variable, label in (
            ("huggingface", "HF_TOKEN", "Hugging Face"),
            ("civitai", "CIVITAI_API_TOKEN", "Civitai"),
        ):
            if provider not in providers:
                continue
            existing = (_token(config) or get_token()) if provider == "huggingface" else os.getenv(variable)
            if existing:
                if provider == "huggingface":
                    config.hf_token = existing
                    # URL downloads consume environment/cache credentials, not config.
                    if os.environ.get(variable) != existing:
                        changed_env[variable] = os.environ.get(variable)
                        os.environ[variable] = existing
                continue
            if no_input or not sys.stdin.isatty():
                continue
            # Never fall back to an echoed password prompt on unsupported terminals.
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    token = getpass.getpass(f"{label} token (hidden; Enter for public/anonymous download): ").strip()
                except (EOFError, getpass.GetPassWarning):
                    print(f"Masked {label} input unavailable; set {variable} securely or use an anonymous download.")
                    continue
            if token:
                changed_env[variable] = os.environ.get(variable)
                os.environ[variable] = token
                if provider == "huggingface":
                    config.hf_token = token
        yield
    finally:
        config.hf_token = original_hf
        for variable, previous in changed_env.items():
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Download MiniMax-H3 FL2VA models used by this studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model types:
  pruned         FP8 pruned transformer, NVFP4 text encoder, video/audio VAEs
  postprocess    Real-ESRGAN, RIFE 4.25 Lite, CodeFormer, face-detection weights
    latent_upscaler Optional learned H3 latent upscaler FP16 and pinned node code
    restore        only Nugus Restore / Enhance / Improve rank16 (optional)
    restore_base   only Ref2VA FP8 transformer (~20.96 GB; optional)
        character_swap only pinned Character Swap LoRA (optional; base via restore_base)
    ref2va_turbo   only pinned Ref2VA Acc 8-step LoRA (optional; no base)
  backend        pruned + postprocess
  base           full Diffusers FL2VA snapshot (optional, H100 / non-pruned LoRAs)
  loras          every catalogued LoRA with a download source
  taomate        default 12 GB TaoMate Civitai LoRA
  silveroxides   Silveroxides DARE-TIES pruned v1 Hugging Face LoRA
  dasiwa         Dasiwa Civitai v1 ranks 48/96/144/512
    dasiwa_v2      Dasiwa turbo-multistep-v2 Hyperflow+EMA600 pruned r128
    dmad           Original DMAD 4-step lora_critic (not full_critic)
    dmad_dareties   DMAD Full DARE-TIES v4 step600 Civitai LoRA
    dmad_hyperflow  Dasiwa DMAD + Hyperflow 4-step r256 Civitai blend
    pdmd_dmad      Dasiwa PDMD + DMAD 4-step r256 Civitai blend
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
    parser.add_argument("--no-input", action="store_true", help="Disable masked token prompts; use existing credentials or anonymous downloads")
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
    with _download_credentials(types, config, lora_ids, no_input=args.no_input):
        run_downloads(types, lora_ids=lora_ids, config=config, comfy_root=args.comfy_root)
    print("Done.")


if __name__ == "__main__":
    main()
