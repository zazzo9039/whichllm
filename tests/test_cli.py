"""Tests for CLI helper logic."""

import pytest
from click.exceptions import Exit

import whichllm.cli as cli_mod
from whichllm.cli import (
    _auto_min_params_for_profile,
    _fill_missing_published_at,
    _generate_chat_script,
    _include_vision_candidates,
    _merge_model_eval_benchmarks,
    _pick_gguf_variant,
    _resolve_ranked_gguf_for_run,
    _resolve_gguf_runtime,
    _resolve_evidence_mode,
    _search_model,
    _validate_evidence,
    app,
)
from whichllm.utils import _current_version
from whichllm.engine.llama_binary import FoundBinaries
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import GPUInfo, HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
import typer
from typer.testing import CliRunner


def _hw_with_gpu(vram_gb: int) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                memory_bandwidth_gbps=1.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=1,
        ram_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )


def test_auto_min_params_general_by_vram():
    # Updated thresholds: tiny GPUs (4-8GB) get a lower floor so they can
    # surface full-GPU 3-4B models instead of being forced into 7B+
    # partial-offload-only candidates.
    assert _auto_min_params_for_profile(_hw_with_gpu(4), "general") == 2.0
    assert _auto_min_params_for_profile(_hw_with_gpu(6), "general") == 3.0
    assert _auto_min_params_for_profile(_hw_with_gpu(8), "general") == 5.0
    assert _auto_min_params_for_profile(_hw_with_gpu(12), "general") == 8.0
    assert _auto_min_params_for_profile(_hw_with_gpu(24), "general") == 10.0
    assert _auto_min_params_for_profile(_hw_with_gpu(32), "general") == 12.0


def test_auto_min_params_non_general_disabled():
    assert _auto_min_params_for_profile(_hw_with_gpu(24), "coding") is None


def test_include_vision_candidates_by_profile():
    assert _include_vision_candidates("vision") is True
    assert _include_vision_candidates("any") is True
    assert _include_vision_candidates("general") is False
    assert _include_vision_candidates("coding") is False


def test_fill_missing_published_at_updates_models():
    model = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
    )

    async def _fake_fetch(ids: list[str]) -> dict[str, str]:
        assert ids == ["Qwen/Qwen3-8B-AWQ"]
        return {"Qwen/Qwen3-8B-AWQ": "2026-03-05T08:00:00.000Z"}

    updated = _fill_missing_published_at([model], [result], _fake_fetch)
    assert updated is True
    assert model.published_at == "2026-03-05T08:00:00.000Z"


def test_version_option_prints_version_and_exits():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert _current_version() in result.stdout


def test_merge_model_eval_benchmarks_is_now_a_noop():
    """As of the self_reported evidence tier, _merge_model_eval_benchmarks
    must NOT mutate the leaderboard scores. Uploader-reported hf_eval values
    are consumed directly by the ranker as a separate, low-trust source.
    """
    model_direct_missing = ModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        family_id="llama-3.1-8b",
        name="Llama-3.1-8B-Instruct",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
        benchmark_scores={"hf_eval": 66.4},
    )
    model_already_present = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        benchmark_scores={"hf_eval": 70.0},
    )
    original = {"Qwen/Qwen2.5-7B-Instruct": 71.2}
    merged, injected = _merge_model_eval_benchmarks(
        [model_direct_missing, model_already_present],
        original,
    )
    # Function is a deprecation no-op now.
    assert injected == 0
    assert merged is original or merged == original
    # Critically, the uploader-reported value MUST NOT have been injected
    # under the model id, because doing so would make it appear as a
    # direct leaderboard hit.
    assert "meta-llama/Llama-3.1-8B-Instruct" not in merged


def test_validate_evidence_accepts_all_modes():
    assert _validate_evidence("strict") == "strict"
    assert _validate_evidence("base") == "base"
    assert _validate_evidence("any") == "any"


def test_validate_evidence_rejects_unknown_mode():
    with pytest.raises(Exit):
        _validate_evidence("foo")


def test_resolve_evidence_mode_direct_alias_wins():
    assert _resolve_evidence_mode("base", direct=True) == "strict"


# --------------- plan command tests ---------------


def test_plan_no_model_found_shows_error():
    runner = CliRunner()
    result = runner.invoke(app, ["plan", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def test_plan_display_plan_renders_tables():
    """display_plan should render model info, VRAM table, and GPU table."""
    from whichllm.output.display import display_plan

    model = ModelInfo(
        id="test-org/Test-Model-7B-GGUF",
        family_id="test-7b",
        name="Test-Model-7B",
        parameter_count=7_000_000_000,
        architecture="llama",
        context_length=4096,
        license="mit",
        downloads=100,
        likes=10,
    )
    # Should not raise
    display_plan(model, context_length=4096, target_quant="Q4_K_M")


def test_plan_display_plan_json_outputs_valid_json():
    """display_plan_json should output valid JSON."""
    import json as json_mod
    from io import StringIO

    from rich.console import Console

    from whichllm.output.display import display_plan_json

    model = ModelInfo(
        id="test-org/Test-Model-7B-GGUF",
        family_id="test-7b",
        name="Test-Model-7B",
        parameter_count=7_000_000_000,
        architecture="llama",
        context_length=4096,
        license="mit",
        downloads=100,
        likes=10,
    )
    # Capture output
    buf = StringIO()
    import whichllm.output.display as disp_mod

    orig_console = disp_mod.console
    disp_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_plan_json(model, context_length=4096, target_quant="Q4_K_M")
    finally:
        disp_mod.console = orig_console
    raw = buf.getvalue().strip()
    data = json_mod.loads(raw)
    assert data["model"]["id"] == "test-org/Test-Model-7B-GGUF"
    assert "vram_by_quant" in data
    assert "gpu_compatibility" in data
    assert data["target_quant"] == "Q4_K_M"


# --------------- helper tests ---------------


def _make_model(model_id="org/Test-7B-GGUF", downloads=100, gguf_variants=None):
    return ModelInfo(
        id=model_id,
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=downloads,
        likes=10,
        gguf_variants=gguf_variants or [],
    )


def test_search_model_exact_match():
    models = [_make_model("org/Llama-8B"), _make_model("org/Qwen-7B")]
    result = _search_model(models, "org/Llama-8B")
    assert result.id == "org/Llama-8B"


def test_search_model_endswith_match():
    models = [_make_model("org/Llama-8B"), _make_model("org/Qwen-7B")]
    result = _search_model(models, "Llama-8B")
    assert result.id == "org/Llama-8B"


def test_search_model_term_match():
    models = [_make_model("org/Llama-3.1-8B-GGUF"), _make_model("org/Qwen-7B")]
    result = _search_model(models, "llama 8b")
    assert result.id == "org/Llama-3.1-8B-GGUF"


def test_search_model_not_found():
    models = [_make_model("org/Llama-8B")]
    with pytest.raises(Exit):
        _search_model(models, "nonexistent_xyz")


def test_pick_gguf_variant_by_preference():
    variants = [
        GGUFVariant(filename="q2.gguf", quant_type="Q2_K", file_size_bytes=1000),
        GGUFVariant(filename="q4km.gguf", quant_type="Q4_K_M", file_size_bytes=2000),
    ]
    model = _make_model(gguf_variants=variants)
    result = _pick_gguf_variant(model)
    assert result.quant_type == "Q4_K_M"


def test_pick_gguf_variant_with_filter():
    variants = [
        GGUFVariant(filename="q2.gguf", quant_type="Q2_K", file_size_bytes=1000),
        GGUFVariant(filename="q4km.gguf", quant_type="Q4_K_M", file_size_bytes=2000),
    ]
    model = _make_model(gguf_variants=variants)
    result = _pick_gguf_variant(model, quant_filter="Q2_K")
    assert result.quant_type == "Q2_K"


def test_pick_gguf_variant_no_variants():
    model = _make_model(gguf_variants=[])
    result = _pick_gguf_variant(model)
    assert result is None


def test_resolve_ranked_synthetic_gguf_to_real_repo():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
    )
    real_gguf = ModelInfo(
        id="unsloth/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=200_000,
        base_model="Qwen/Qwen3.6-27B",
        gguf_variants=[
            GGUFVariant(
                filename="Qwen3.6-27B-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(selected, synthetic, [selected, real_gguf])

    assert resolved is not None
    model, variant = resolved
    assert model.id == "unsloth/Qwen3.6-27B-GGUF"
    assert variant.filename == "Qwen3.6-27B-Q4_K_M.gguf"


def test_resolve_ranked_synthetic_gguf_prefers_exact_quant():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    q5_only = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="q5.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    q4_match = ModelInfo(
        id="smaller/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=10,
        gguf_variants=[
            GGUFVariant(
                filename="q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(
        selected,
        synthetic,
        [selected, q5_only, q4_match],
    )

    assert resolved is not None
    model, variant = resolved
    assert model.id == "smaller/Qwen3.6-27B-GGUF"
    assert variant.quant_type == "Q4_K_M"


def test_resolve_ranked_synthetic_gguf_rejects_quant_mismatch():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    q5_only = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="q5.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(selected, synthetic, [selected, q5_only])

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_without_real_repo_returns_none():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    unrelated = ModelInfo(
        id="other/Model-7B-GGUF",
        family_id="model-7b",
        name="Model-7B-GGUF",
        parameter_count=7_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="other.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    assert (
        _resolve_ranked_gguf_for_run(selected, synthetic, [selected, unrelated]) is None
    )


def test_resolve_ranked_synthetic_gguf_rejects_size_mismatch():
    selected = ModelInfo(
        id="deepseek-ai/DeepSeek-V4-Flash",
        family_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash",
        parameter_count=158_000_000_000,
    )
    mtp_head = ModelInfo(
        id="converter/deepseek-v4-flash-mtp-gguf",
        family_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash-MTP-GGUF",
        parameter_count=6_600_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="mtp.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="DeepSeek-V4-Flash.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=90_000_000_000,
    )

    resolved = _resolve_ranked_gguf_for_run(selected, synthetic, [selected, mtp_head])

    assert resolved is None


# --------------- run/snippet command tests ---------------


def test_resolve_gguf_runtime_finds_gguf_sibling():
    official = ModelInfo(
        id="Qwen/Qwen3-14B",
        family_id="qwen3-14b",
        name="Qwen3-14B",
        parameter_count=14_800_000_000,
        downloads=2_000_000,
    )
    gguf_repo = ModelInfo(
        id="unsloth/Qwen3-14B-GGUF",
        family_id="qwen3-14b",
        name="Qwen3-14B-GGUF",
        parameter_count=14_800_000_000,
        downloads=500_000,
        base_model="Qwen/Qwen3-14B",
        gguf_variants=[
            GGUFVariant(
                filename="Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=9_000_000_000,
            )
        ],
    )

    resolved = _resolve_gguf_runtime(official, [official, gguf_repo], "Q4_K_M")

    assert resolved is not None
    model, variant = resolved
    assert model.id == "unsloth/Qwen3-14B-GGUF"
    assert variant.quant_type == "Q4_K_M"


def test_run_exits_gracefully():
    """run should fail gracefully (uv missing, or no model found)."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "some-model"])
    if result.exit_code != 0:
        assert any(
            msg in result.stdout
            for msg in ("uv is required", "No model found", "llama-cpp-python")
        )


def test_transformers_chat_script_passes_tokenizer_mapping_to_generate():
    model = _make_model(model_id="org/Test-7B")

    script = _generate_chat_script(
        model, variant=None, context_length=4096, cpu_only=False
    )

    assert "return_dict=True" in script
    assert "kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer)" in script
    assert "kwargs=dict(input_ids=inputs" not in script


def test_transformers_chat_script_provides_disk_offload_folder():
    model = _make_model(model_id="org/Test-7B")

    script = _generate_chat_script(
        model, variant=None, context_length=4096, cpu_only=False
    )

    assert 'tempfile.mkdtemp(prefix="whichllm_transformers_offload_")' in script
    assert "offload_folder=offload_folder" in script
    assert "shutil.rmtree(offload_folder, ignore_errors=True)" in script


def test_run_auto_pick_resolves_ranked_gguf_before_launch(monkeypatch):
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
    )
    real_gguf = ModelInfo(
        id="unsloth/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=200_000,
        base_model="Qwen/Qwen3.6-27B",
        gguf_variants=[
            GGUFVariant(
                filename="q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["quant_filter"] = kwargs.get("quant_filter")
        return [
            CompatibilityResult(
                model=selected,
                gguf_variant=synthetic,
                can_run=True,
                vram_required_bytes=0,
                vram_available_bytes=0,
                quality_score=90.0,
            )
        ]

    def fake_execute(plan, **kwargs):
        captured["plan"] = plan
        return 0

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        "whichllm.engine.launch_plan.find_binaries",
        lambda: FoundBinaries(cli=None, server=None),
    )
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [selected, real_gguf])
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(8)
    )
    monkeypatch.setattr("whichllm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichllm.models.local.scan_local_models", lambda _: [])
    monkeypatch.setattr("whichllm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichllm.engine.launch_plan.execute_launch_plan", fake_execute
    )

    result = CliRunner().invoke(app, ["run", "--quant", "Q4_K_M"])

    assert result.exit_code == 0
    assert captured["quant_filter"] == "Q4_K_M"
    plan = captured["plan"]
    assert plan.model_id == "unsloth/Qwen3.6-27B-GGUF"
    assert plan.model_path == "q4.gguf"
    assert plan.runtime == "uv_fallback"


def test_run_hf_gguf_uses_native_llama_cli_when_available(monkeypatch, tmp_path):
    model = ModelInfo(
        id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        family_id="qwen2.5",
        name="Qwen2.5-1.5B",
        parameter_count=1_500_000_000,
        downloads=100_000,
        gguf_variants=[
            GGUFVariant(
                filename="model.q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=1_000_000_000,
            )
        ],
    )
    gguf_file = tmp_path / "model.q4.gguf"
    gguf_file.write_bytes(b"x" * 1024)
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_execute(plan, **kwargs):
        captured["plan"] = plan
        if plan.needs_download:
            captured["download"] = (plan.model_id, plan.model_path)
        return 0

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])
    monkeypatch.setattr("whichllm.models.local.scan_local_models", lambda _: [])
    monkeypatch.setattr(
        "whichllm.engine.launch_plan.find_binaries",
        lambda: FoundBinaries(cli=tmp_path / "llama-cli.exe", server=None),
    )
    monkeypatch.setattr(
        "whichllm.engine.launch_plan.execute_launch_plan", fake_execute
    )

    result = CliRunner().invoke(app, ["run", "qwen 2.5 1.5b gguf"])

    assert result.exit_code == 0
    assert captured["download"] == ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "model.q4.gguf")
    plan = captured["plan"]
    assert plan.runtime == "native"
    assert plan.ngl == 99
    assert plan.model_id == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"


def test_snippet_no_model_found():
    runner = CliRunner()
    result = runner.invoke(app, ["snippet", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def test_json_output_includes_benchmark_source_and_confidence():
    """display_json should include benchmark_source and benchmark_confidence."""
    import json as json_mod
    from io import StringIO

    from rich.console import Console

    from whichllm.output.display import display_json

    model = ModelInfo(
        id="test-org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=100,
        likes=10,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=8_000_000_000,
        vram_available_bytes=24_000_000_000,
        quality_score=55.0,
        benchmark_status="estimated",
        benchmark_source="line_interp",
        benchmark_confidence=0.34,
    )
    hw = HardwareInfo(
        gpus=[],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )

    buf = StringIO()
    import whichllm.output.display as disp_mod

    orig_console = disp_mod.console
    disp_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_json([result], hw)
    finally:
        disp_mod.console = orig_console

    data = json_mod.loads(buf.getvalue().strip())
    entry = data["models"][0]
    assert entry["benchmark_status"] == "estimated"
    assert entry["benchmark_source"] == "line_interp"
    assert entry["benchmark_confidence"] == 0.34


def test_serve_default_context_is_32k(monkeypatch):
    captured: dict[str, int] = {}
    model = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
        context_length=131072,
        downloads=1000,
        likes=100,
        gguf_variants=[
            GGUFVariant(
                filename="qwen3-8b-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_500_000_000,
            )
        ],
    )

    def fake_launch(*args, **kwargs):
        captured["context_length"] = kwargs["context_length"]
        raise typer.Exit(0)

    monkeypatch.setattr(cli_mod, "_launch_from_plan", fake_launch)
    monkeypatch.setattr(cli_mod, "_warn_tool_calling_if_unverified", lambda _r: None)
    monkeypatch.setattr(cli_mod, "_load_models", lambda refresh: [model])
    monkeypatch.setattr("whichllm.models.local.scan_local_models", lambda _: [])
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: _hw_with_gpu(24)
    )

    result = CliRunner().invoke(app, ["serve", "qwen3 8b", "--cpu-only"])

    assert result.exit_code == 0
    assert captured["context_length"] == 32768
