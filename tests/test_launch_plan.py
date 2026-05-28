"""Tests for launch plan construction (no binaries or network)."""

from pathlib import Path

import pytest

from whichllm.engine.launch_plan import LaunchPlan, make_launch_plan
from whichllm.engine.llama_binary import FoundBinaries
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.local import LocalModel
from whichllm.models.types import GGUFVariant, ModelInfo


def _hardware(vram_gb: int = 24) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="Test GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=900.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )


def _result(
    *,
    family_id: str = "qwen3-14b",
    model_id: str = "Qwen/Qwen3-14B",
    is_local: bool = False,
    quant: str = "Q4_K_M",
    file_size_bytes: int = 8_000_000_000,
    parameter_count: int = 14_000_000_000,
) -> CompatibilityResult:
    return CompatibilityResult(
        model=ModelInfo(
            id=model_id,
            family_id=family_id,
            name="Qwen3-14B",
            parameter_count=parameter_count,
            downloads=1000,
            likes=100,
        ),
        gguf_variant=GGUFVariant(
            filename="Qwen3-14B-Q4_K_M.gguf",
            quant_type=quant,
            file_size_bytes=file_size_bytes,
        ),
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        is_local=is_local,
    )


def test_plan_local_skips_download():
    local_path = Path("/models/Qwen3-14B-Q4_K_M.gguf")
    local_models = [
        LocalModel(
            path=local_path,
            name="Qwen3-14B-Q4_K_M",
            size_bytes=8_000_000_000,
            is_gguf=True,
            quant_type="Q4_K_M",
            family_id="qwen3-14b",
        )
    ]
    plan = make_launch_plan(
        _result(is_local=True),
        _hardware(),
        "server",
        local_models,
        context_length=4096,
        cpu_only=False,
    )

    assert plan.model_source == "local"
    assert plan.needs_download is False
    assert plan.model_path == str(local_path)


def test_plan_remote_needs_download():
    plan = make_launch_plan(
        _result(is_local=False),
        _hardware(),
        "server",
        [],
        context_length=4096,
        cpu_only=False,
    )

    assert plan.model_source == "hf_download"
    assert plan.needs_download is True
    assert plan.model_id == "Qwen/Qwen3-14B"
    assert plan.model_path == "Qwen3-14B-Q4_K_M.gguf"


def test_plan_args_computed_without_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = make_launch_plan(
        _result(file_size_bytes=8_000_000_000),
        _hardware(),
        "server",
        [],
        context_length=8192,
        cpu_only=False,
    )

    assert plan.args["ctx_size"] > 0
    assert plan.ngl >= 0
    assert plan.file_size_gb > 0


def test_plan_no_binary_local_is_error(monkeypatch):
    monkeypatch.setattr(
        "whichllm.engine.launch_plan.find_binaries",
        lambda: FoundBinaries(server=None, cli=None),
    )
    plan = make_launch_plan(
        _result(is_local=True),
        _hardware(),
        "server",
        [
            LocalModel(
                path=Path("/models/local.gguf"),
                name="local",
                size_bytes=4_000_000_000,
                is_gguf=True,
                quant_type="Q4_K_M",
                family_id="qwen3-14b",
            )
        ],
        context_length=4096,
        cpu_only=False,
    )

    assert plan.runtime == "error"
    assert plan.error is not None


def test_plan_no_binary_remote_is_uv_fallback(monkeypatch):
    monkeypatch.setattr(
        "whichllm.engine.launch_plan.find_binaries",
        lambda: FoundBinaries(server=None, cli=None),
    )
    plan = make_launch_plan(
        _result(is_local=False),
        _hardware(),
        "server",
        [],
        context_length=4096,
        cpu_only=False,
    )

    assert plan.runtime == "uv_fallback"
    assert plan.binary is None


def test_plan_prefers_matching_quant():
    local_models = [
        LocalModel(
            path=Path("/models/Qwen3-14B-Q8_0.gguf"),
            name="Qwen3-14B-Q8_0",
            size_bytes=12_000_000_000,
            is_gguf=True,
            quant_type="Q8_0",
            family_id="qwen3-14b",
        ),
        LocalModel(
            path=Path("/models/Qwen3-14B-Q4_K_M.gguf"),
            name="Qwen3-14B-Q4_K_M",
            size_bytes=8_000_000_000,
            is_gguf=True,
            quant_type="Q4_K_M",
            family_id="qwen3-14b",
        ),
    ]
    plan = make_launch_plan(
        _result(is_local=True, quant="Q4_K_M"),
        _hardware(),
        "server",
        local_models,
        context_length=4096,
        cpu_only=False,
    )

    assert plan.quant_type == "Q4_K_M"
    assert Path(plan.model_path).name == "Qwen3-14B-Q4_K_M.gguf"


def test_plan_cpu_only_zeroes_ngl():
    plan = make_launch_plan(
        _result(is_local=False),
        _hardware(),
        "chat",
        [],
        context_length=4096,
        cpu_only=True,
    )

    assert plan.ngl == 0
    assert isinstance(plan, LaunchPlan)
