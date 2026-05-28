"""Native llama.cpp binary launcher for GPU-accelerated local inference.

Finds llama-server and llama-cli (.exe on Windows) on the system and generates
optimised argument lists based on detected hardware and model file size.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    _KNOWN_LLAMA_DIRS = [
        Path(r"C:\tools\llama-cpp"),
        Path(r"C:\tools\llama.cpp"),
        Path(r"C:\llama-cpp"),
        Path(r"C:\llama.cpp"),
    ]
else:
    _KNOWN_LLAMA_DIRS = [
        Path("/usr/local/bin"),
        Path("/usr/bin"),
        Path("/opt/llama.cpp/bin"),
        Path.home() / ".local" / "bin",
        Path.home() / "llama.cpp" / "bin",
        Path.home() / "llama.cpp" / "build" / "bin",
    ]

_ENV_LLAMA_DIR = "WHICHLLM_LLAMA_DIR"
_SERVER_BIN = "llama-server.exe" if _IS_WINDOWS else "llama-server"
_CLI_BIN = "llama-cli.exe" if _IS_WINDOWS else "llama-cli"

# Bytes-per-weight lookup (same source as constants.py)
_BYTES_PER_WEIGHT: dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 1.0625,
    "Q6_K": 0.8125,
    "Q5_K_M": 0.6875,
    "Q5_K_S": 0.6875,
    "Q5_0": 0.625,
    "Q4_K_M": 0.5625,
    "Q4_K_S": 0.5625,
    "Q4_0": 0.5,
    "Q3_K_M": 0.4375,
    "Q3_K_S": 0.4375,
    "Q3_K_L": 0.4375,
    "Q2_K": 0.3125,
    "IQ4_XS": 0.5,
    "IQ4_NL": 0.5,
    "IQ3_XXS": 0.375,
    "IQ2_XXS": 0.25,
    "IQ1_S": 0.21,
    "IQ1_M": 0.22,
}

# KV cache: ~3.5 MB per B-active-param per K-context-token (same as vram.py)
_KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024


@dataclass
class FoundBinaries:
    server: Path | None = None
    cli: Path | None = None


def find_binaries() -> FoundBinaries:
    """Locate llama-server and llama-cli (.exe on Windows) on the system.

    Checks, in order:
      1. WHICHLLM_LLAMA_DIR env var
      2. PATH via shutil.which
      3. Well-known install directories
    """
    result = FoundBinaries()

    candidates: list[Path] = []
    env_dir = os.environ.get(_ENV_LLAMA_DIR)
    if env_dir:
        env_path = Path(env_dir)
        candidates.append(env_path.parent if env_path.is_file() else env_path)
    candidates.extend(_KNOWN_LLAMA_DIRS)

    for base in candidates:
        base = Path(base)
        if not base.is_dir():
            continue
        if not result.server:
            s = base / _SERVER_BIN
            if s.is_file():
                result.server = s
        if not result.cli:
            c = base / _CLI_BIN
            if c.is_file():
                result.cli = c
        if result.server and result.cli:
            return result

    # Fallback: check PATH
    if not result.server:
        s = shutil.which(_SERVER_BIN)
        if s:
            result.server = Path(s)
    if not result.cli:
        c = shutil.which(_CLI_BIN)
        if c:
            result.cli = Path(c)

    return result


def extract_quant(name: str) -> str | None:
    """Extract quantisation type from a GGUF filename stem."""
    for pat in [
        r"IQ[1-4]_[A-Z0-9_]+",
        r"Q[0-9](?:_[A-Z0-9]+)*",
        r"F(?:16|32)",
    ]:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            return m.group().upper()
    return None


def _infer_params_b(file_size_gb: float, quant_type: str | None) -> float:
    """Rough parameter-count estimate from file size + quant."""
    bpw = _BYTES_PER_WEIGHT.get(quant_type, 0.5625) if quant_type else 0.5625
    return file_size_gb / bpw if bpw > 0 else file_size_gb / 0.5625


def estimate_max_context(
    file_size_gb: float,
    vram_gb: float,
    quant_type: str | None = None,
    kv_quant: str = "q8_0",
) -> int:
    """Estimate max context length that fits in VRAM with full GPU offload.

    Formula:
      VRAM_available = VRAM_total - file_size - overhead
      KV_bytes = 3.5 MB * params_b * (ctx / 1024)
      KV with q4_0 KV ≈ KV / 2  (4-bit = half of 8-bit)
    """
    overhead_gb = 0.6
    available_gb = vram_gb - file_size_gb - overhead_gb
    if available_gb <= 0:
        return 4096

    params_b = _infer_params_b(file_size_gb, quant_type)
    kv_div = 2.0 if kv_quant == "q4_0" else 1.0

    ctx = int(
        available_gb * (1024**3) * kv_div
        / (params_b * _KV_BYTES_PER_BPARAM_PER_KCTX / 1024)
    )
    # Clamp to sensible bounds
    return max(4096, min(ctx, 131072))


def auto_params(
    file_size_gb: float,
    vram_gb: float,
    quant_type: str | None = None,
    requested_ctx: int = 4096,
) -> dict:
    """Auto-tune llama.cpp CLI flags based on model size and VRAM."""
    # Decide KV cache quantisation based on VRAM tightness
    overhead_gb = 0.6
    slack_gb = vram_gb - file_size_gb - overhead_gb
    if slack_gb < 2.0:
        kv_quant = "q4_0"
        max_ctx = estimate_max_context(file_size_gb, vram_gb, quant_type, "q4_0")
    elif slack_gb < 4.0:
        kv_quant = "q4_0"
        max_ctx = estimate_max_context(file_size_gb, vram_gb, quant_type, "q4_0")
    else:
        kv_quant = "q8_0"
        max_ctx = estimate_max_context(file_size_gb, vram_gb, quant_type, "q8_0")

    ctx = min(requested_ctx, max_ctx)

    # Batch size scales with available VRAM
    if slack_gb < 1.0:
        batch = 512
    elif slack_gb < 2.5:
        batch = 1024
    else:
        batch = 2048

    return {
        "ngl": 99,
        "ctx_size": ctx,
        "flash_attn": True,
        "cache_type_k": kv_quant,
        "cache_type_v": kv_quant,
        "batch_size": batch,
        "ubatch_size": batch,
    }


def build_server_args(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    context_length: int = 4096,
    gpu_layers: int = 99,
    flash_attn: bool = True,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    batch_size: int = 2048,
    ubatch_size: int = 2048,
    threads: int | None = None,
) -> list[str]:
    """Build argument list for llama-server.exe."""
    args = [
        "-m", model_path,
        "-ngl", str(gpu_layers),
        "--ctx-size", str(context_length),
        "--batch-size", str(batch_size),
        "--ubatch-size", str(ubatch_size),
        "-np", "1",
        "--no-mmap",
        "--jinja",
        "--host", host,
        "--port", str(port),
    ]
    if flash_attn:
        args += ["--flash-attn", "on"]
    if cache_type_k:
        args += ["--cache-type-k", cache_type_k]
    if cache_type_v:
        args += ["--cache-type-v", cache_type_v]
    if threads:
        args += ["-t", str(threads)]
    return args


def build_cli_args(
    model_path: str,
    context_length: int = 4096,
    gpu_layers: int = 99,
    flash_attn: bool = True,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    batch_size: int = 2048,
    ubatch_size: int = 2048,
    threads: int | None = None,
) -> list[str]:
    """Build argument list for llama-cli.exe (interactive mode)."""
    args = [
        "-m", model_path,
        "-ngl", str(gpu_layers),
        "--ctx-size", str(context_length),
        "--batch-size", str(batch_size),
        "--ubatch-size", str(ubatch_size),
        "--no-mmap",
        "--jinja",
    ]
    if flash_attn:
        args += ["--flash-attn", "on"]
    if cache_type_k:
        args += ["--cache-type-k", cache_type_k]
    if cache_type_v:
        args += ["--cache-type-v", cache_type_v]
    if threads:
        args += ["-t", str(threads)]
    return args


def run_server(
    binary: Path,
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    context_length: int = 4096,
    gpu_layers: int = 99,
    **kwargs,
) -> subprocess.Popen:
    """Launch llama-server.exe as a subprocess and return the Popen handle."""
    args = build_server_args(
        model_path=model_path,
        host=host,
        port=port,
        context_length=context_length,
        gpu_layers=gpu_layers,
        **kwargs,
    )
    full_cmd = [str(binary)] + args
    return subprocess.Popen(
        full_cmd,
        cwd=str(binary.parent),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def run_cli(
    binary: Path,
    model_path: str,
    context_length: int = 4096,
    gpu_layers: int = 99,
    **kwargs,
) -> int:
    """Run llama-cli.exe in interactive mode and return its exit code."""
    args = build_cli_args(
        model_path=model_path,
        context_length=context_length,
        gpu_layers=gpu_layers,
        **kwargs,
    )
    full_cmd = [str(binary)] + args
    result = subprocess.run(full_cmd, cwd=str(binary.parent))
    return result.returncode


def launch_params_for_gguf(
    model_path: str | Path,
    quant_type: str | None,
    context_length: int,
    cpu_only: bool,
) -> tuple[dict, int]:
    """Auto-tune llama.cpp flags from an on-disk GGUF file and hardware."""
    from whichllm.hardware.detector import detect_hardware

    path = Path(model_path)
    hw = detect_hardware()
    vram_gb = max((g.vram_bytes for g in hw.gpus), default=0) / (1024**3)
    file_size_gb = path.stat().st_size / (1024**3)
    params = auto_params(file_size_gb, vram_gb, quant_type, context_length)
    ngl = 0 if cpu_only else params["ngl"]
    return params, ngl


def download_hf_gguf(repo_id: str, filename: str) -> Path:
    """Download a GGUF file from HuggingFace using uv + huggingface-hub."""
    import tempfile

    if not shutil.which("uv"):
        raise RuntimeError("uv is required to download HuggingFace models")

    script = f"""\
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id={repo_id!r}, filename={filename!r}))
"""
    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="whichllm_dl_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(script)
        result = subprocess.run(
            ["uv", "run", "--no-project", "--with", "huggingface-hub", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("huggingface-hub download returned no path")
        return Path(lines[-1])
    finally:
        os.unlink(script_path)
