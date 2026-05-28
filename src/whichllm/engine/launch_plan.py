"""Pure launch planning and execution for native llama.cpp / uv fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from whichllm.constants import QUANT_BYTES_PER_WEIGHT
from whichllm.engine.llama_binary import (
    auto_params,
    download_hf_gguf,
    find_binaries,
    run_cli,
    run_server,
)
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo
from whichllm.models.local import LocalModel
from whichllm.models.types import GGUFVariant, ModelInfo

_LOCAL_BINARY_ERROR = (
    "binario llama.cpp non trovato — impossibile servire un modello "
    "locale senza di esso. Imposta WHICHLLM_LLAMA_DIR."
)


@dataclass
class LaunchPlan:
    target: Literal["chat", "server"]
    runtime: Literal["native", "uv_fallback", "error"]
    binary: Path | None
    model_source: Literal["local", "hf_download"]
    model_path: str
    model_id: str
    quant_type: str | None
    args: dict[str, Any]
    ngl: int
    needs_download: bool
    file_size_gb: float
    error: str | None = None


def _estimate_file_size_gb(
    file_size_bytes: int,
    parameter_count: int,
    quant_type: str | None,
) -> float:
    size_gb = file_size_bytes / (1024**3)
    if size_gb > 0:
        return size_gb
    qt = (quant_type or "Q4_K_M").upper()
    bpw = QUANT_BYTES_PER_WEIGHT.get(qt, 0.5625)
    if parameter_count <= 0:
        return 0.0
    return parameter_count * bpw / (1024**3)


def _resolve_model_source(
    result: CompatibilityResult,
    local_models: list[LocalModel],
) -> tuple[Literal["local", "hf_download"], str, str, str | None, float, bool]:
    variant = result.gguf_variant
    quant_type = variant.quant_type if variant else None

    if result.is_local:
        match = [m for m in local_models if m.family_id == result.model.family_id]
        match.sort(key=lambda m: m.size_bytes, reverse=True)
        preferred = next(
            (
                m
                for m in match
                if variant
                and m.quant_type
                and m.quant_type.upper() == variant.quant_type.upper()
            ),
            match[0] if match else None,
        )
        if preferred is not None:
            return (
                "local",
                str(preferred.path),
                preferred.name,
                preferred.quant_type or quant_type,
                preferred.size_bytes / (1024**3),
                False,
            )

    if not variant:
        raise ValueError("GGUF variant required for remote launch plan")

    file_size_gb = _estimate_file_size_gb(
        variant.file_size_bytes,
        result.model.parameter_count,
        variant.quant_type,
    )
    return (
        "hf_download",
        variant.filename,
        result.model.id,
        variant.quant_type,
        file_size_gb,
        True,
    )


def make_launch_plan(
    result: CompatibilityResult,
    hardware: HardwareInfo,
    target: Literal["chat", "server"],
    local_models: list[LocalModel],
    context_length: int,
    cpu_only: bool,
) -> LaunchPlan:
    """Pure launch decision — no download, subprocess, or disk stat()."""
    model_source, model_path, model_id, quant_type, file_size_gb, needs_download = (
        _resolve_model_source(result, local_models)
    )

    vram_gb = max((g.vram_bytes for g in hardware.gpus), default=0) / (1024**3)
    args = auto_params(file_size_gb, vram_gb, quant_type, context_length)
    ngl = 0 if cpu_only else args["ngl"]

    binaries = find_binaries()
    needed = binaries.server if target == "server" else binaries.cli
    if needed:
        runtime: Literal["native", "uv_fallback", "error"] = "native"
        binary = needed
        error = None
    elif model_source == "local":
        runtime = "error"
        binary = None
        error = _LOCAL_BINARY_ERROR
    else:
        runtime = "uv_fallback"
        binary = None
        error = None

    return LaunchPlan(
        target=target,
        runtime=runtime,
        binary=binary,
        model_source=model_source,
        model_path=model_path,
        model_id=model_id,
        quant_type=quant_type,
        args=args,
        ngl=ngl,
        needs_download=needs_download,
        file_size_gb=file_size_gb,
        error=error,
    )


def _run_uv_fallback(
    plan: LaunchPlan,
    host: str,
    port: int,
    context_length: int,
    cpu_only: bool,
) -> int:
    from whichllm.cli import (
        _generate_chat_script,
        _generate_serve_script,
        _resolve_model_deps,
    )

    if not shutil.which("uv"):
        print("[red]uv is required for llama-cpp-python fallback.")
        print("Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1

    if plan.target == "server":
        script = _generate_serve_script(
            model_path=plan.model_path,
            model_id=plan.model_id,
            quant_type=plan.quant_type,
            host=host,
            port=port,
            context_length=context_length,
            cpu_only=cpu_only,
        )
        deps = ["llama-cpp-python", "huggingface-hub", "fastapi", "uvicorn", "pydantic"]
        prefix = "whichllm_serve_"
    else:
        model = ModelInfo(
            id=plan.model_id,
            family_id=plan.model_id,
            name=plan.model_id.split("/")[-1] if "/" in plan.model_id else plan.model_id,
            parameter_count=0,
            downloads=0,
            likes=0,
        )
        variant = GGUFVariant(
            filename=plan.model_path,
            quant_type=plan.quant_type or "Q4_K_M",
            file_size_bytes=max(1, int(plan.file_size_gb * (1024**3))),
        )
        deps, _ = _resolve_model_deps(model, variant)
        script = _generate_chat_script(model, variant, context_length, cpu_only)
        prefix = "whichllm_run_"

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix=prefix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(script)
        cmd = ["uv", "run", "--no-project"]
        for dep in deps:
            cmd.extend(["--with", dep])
        cmd.append(script_path)
        completed = subprocess.run(cmd)
        return completed.returncode
    finally:
        os.unlink(script_path)


def execute_launch_plan(
    plan: LaunchPlan,
    host: str = "localhost",
    port: int = 8000,
    cpu_only: bool = False,
    context_length: int = 4096,
) -> int:
    """Download (if needed) and launch the planned runtime."""
    if plan.runtime == "error":
        print(plan.error or "Launch plan error")
        return 1

    model_path = plan.model_path
    if plan.needs_download:
        print("Downloading GGUF from HuggingFace...")
        try:
            model_path = str(download_hf_gguf(plan.model_id, plan.model_path))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            err = getattr(exc, "stderr", None) or str(exc)
            print(f"Download failed: {err}")
            return 1

    if plan.runtime == "native":
        assert plan.binary is not None
        if plan.target == "server":
            proc = run_server(
                binary=plan.binary,
                model_path=model_path,
                host=host,
                port=port,
                context_length=plan.args["ctx_size"],
                gpu_layers=plan.ngl,
                flash_attn=plan.args["flash_attn"],
                cache_type_k=plan.args["cache_type_k"],
                cache_type_v=plan.args["cache_type_v"],
                batch_size=plan.args["batch_size"],
                ubatch_size=plan.args["ubatch_size"],
            )
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
            return proc.returncode or 0

        return run_cli(
            binary=plan.binary,
            model_path=model_path,
            context_length=plan.args["ctx_size"],
            gpu_layers=plan.ngl,
            flash_attn=plan.args["flash_attn"],
            cache_type_k=plan.args["cache_type_k"],
            cache_type_v=plan.args["cache_type_v"],
            batch_size=plan.args["batch_size"],
            ubatch_size=plan.args["ubatch_size"],
        )

    if plan.runtime == "uv_fallback":
        if plan.target == "chat":
            print(
                "llama-cli.exe not found — falling back to llama-cpp-python "
                "(may be CPU-only on Windows)."
            )
        else:
            print(
                "llama-server.exe not found — falling back to llama-cpp-python "
                "(may be CPU-only on Windows)."
            )
        return _run_uv_fallback(plan, host, port, context_length, cpu_only)

    return 1
