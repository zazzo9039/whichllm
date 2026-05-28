"""Local models scanner for discovering local GGUF/model files on disk."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from whichllm.engine.llama_binary import extract_quant
from whichllm.models.grouper import _normalize_name

_ENV_MODELS_DIR = "WHICHLLM_MODELS_DIR"
_WIN_DEFAULT_MODELS_DIR = Path(r"C:\models")

_PACKAGING_PREFIX_RE = re.compile(
    r"^(?:unsloth|bartowski|lmstudio-community)[-._]",
    re.IGNORECASE,
)
_SHARD_SUFFIX_RE = re.compile(
    r"[-._](?:\d+-of-\d+|split-[a-z]+|part\d+)$",
    re.IGNORECASE,
)
_GENERIC_FAMILY_NAMES = frozenset({"model", "ggml", "output", "merged", "final"})


def default_models_dir() -> str:
    """Return the default directory for local GGUF files."""
    override = os.environ.get(_ENV_MODELS_DIR)
    if override:
        return override
    if os.name == "nt" and _WIN_DEFAULT_MODELS_DIR.is_dir():
        return str(_WIN_DEFAULT_MODELS_DIR)
    return str(Path.home() / "models")


@dataclass
class LocalModel:
    """A model found in the local directory."""

    path: Path
    name: str
    size_bytes: int
    is_gguf: bool
    quant_type: str | None = None
    family_id: str | None = None


def local_to_family_id(filename_stem: str) -> str | None:
    """Map a local GGUF filename stem to the ranker's ``family_id`` key."""
    name = filename_stem.strip()
    if not name:
        return None

    name = _PACKAGING_PREFIX_RE.sub("", name)
    name = _SHARD_SUFFIX_RE.sub("", name)

    quant = extract_quant(name)
    if quant:
        name = re.sub(
            rf"[-._](?:UD[-._]?)?{re.escape(quant)}$",
            "",
            name,
            flags=re.IGNORECASE,
        )

    name = name.strip("-._ ")
    if len(name.replace("-", "")) < 3 or name.lower() in _GENERIC_FAMILY_NAMES:
        return None

    family_id = _normalize_name(name)
    if len(family_id.replace("-", "")) < 3 or family_id in _GENERIC_FAMILY_NAMES:
        return None
    return family_id


def scan_local_models(directory: str | None = None) -> list[LocalModel]:
    """Scan a directory for local model files.

    Args:
        directory: Path to scan for models (default: ``~/models`` or
            ``WHICHLLM_MODELS_DIR`` when set)

    Returns:
        List of LocalModel objects found in the directory.
    """
    models: list[LocalModel] = []
    base_path = Path(directory or default_models_dir())

    if not base_path.exists():
        return models

    for entry in base_path.iterdir():
        if entry.is_file():
            model = _process_file(entry)
            if model:
                models.append(model)
        elif entry.is_dir():
            model = _best_model_in_dir(entry)
            if model:
                models.append(model)

    return sorted(models, key=lambda m: m.size_bytes, reverse=True)


def _is_mmproj(path: Path) -> bool:
    """Return True for llama.cpp vision projector sidecars, not chat weights."""
    stem = path.stem.lower()
    return stem.startswith("mmproj") or stem.startswith("mm-proj")


def _best_model_in_dir(dir_path: Path) -> LocalModel | None:
    """Pick the largest GGUF in a folder, excluding mmproj sidecars."""
    ggufs: list[LocalModel] = []
    others: list[LocalModel] = []

    for sub_file in dir_path.iterdir():
        if not sub_file.is_file() or _is_mmproj(sub_file):
            continue
        model = _process_file(sub_file)
        if not model:
            continue
        if model.is_gguf:
            ggufs.append(model)
        else:
            others.append(model)

    if ggufs:
        return max(ggufs, key=lambda m: m.size_bytes)
    if others:
        return max(others, key=lambda m: m.size_bytes)
    return None


def _process_file(path: Path) -> LocalModel | None:
    """Process a single file to extract model info."""
    if _is_mmproj(path):
        return None

    suffix = path.suffix.lower()
    is_gguf = suffix == ".gguf"

    if not is_gguf and suffix not in {".bin", ".safetensors", ".pt", ".pth"}:
        return None

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0

    quant_type = None
    family_id = None
    if is_gguf:
        quant_type = _extract_quant_type(path.stem)
        family_id = local_to_family_id(path.stem)

    return LocalModel(
        path=path,
        name=path.stem,
        size_bytes=size_bytes,
        is_gguf=is_gguf,
        quant_type=quant_type,
        family_id=family_id,
    )


def _extract_quant_type(name: str) -> str | None:
    """Extract quantization type from GGUF filename."""
    return extract_quant(name)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    if size_bytes >= 1024**4:
        return f"{size_bytes / 1024**4:.1f} TB"
    elif size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.1f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024:.1f} KB"
