"""CLI entry point using typer."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console

from whichllm.hardware.types import HardwareInfo
from whichllm.models.types import GGUFVariant, ModelInfo
from whichllm.utils import _current_version, CONTEXT_LENGTH

app = typer.Typer(
    name="llm-checker",
    help="Find the best LLM that runs on your hardware.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def _run_async(coro):
    """Run async coroutine from sync context."""
    return asyncio.run(coro)


def _print_version(value: bool) -> None:
    """Print version and exit when --version is requested."""
    if value:
        console.print(_current_version())
        raise typer.Exit()


def _validate_gpu_flags(
    cpu_only: bool,
    gpu: str | None,
    vram: float | None,
) -> None:
    """Validate mutual exclusivity of GPU-related flags."""
    if cpu_only and gpu:
        console.print("[red]Error:[/] --cpu-only and --gpu are mutually exclusive.")
        raise typer.Exit(code=1)
    if vram is not None and not gpu:
        console.print("[red]Error:[/] --vram requires --gpu.")
        raise typer.Exit(code=1)


def _validate_profile(profile: str) -> str:
    """Validate ranking profile option."""
    valid = {"general", "coding", "vision", "math", "any"}
    p = profile.lower()
    if p not in valid:
        console.print(
            "[red]Error:[/] --profile must be one of: general, coding, vision, math, any."
        )
        raise typer.Exit(code=1)
    return p


def _validate_evidence(evidence: str) -> str:
    """Validate evidence mode option."""
    valid = {"strict", "base", "any"}
    mode = evidence.lower()
    if mode not in valid:
        console.print("[red]Error:[/] --evidence must be one of: strict, base, any.")
        raise typer.Exit(code=1)
    return mode


def _resolve_evidence_mode(evidence: str, direct: bool) -> str:
    """Resolve final evidence mode, keeping --direct as strict alias."""
    mode = _validate_evidence(evidence)
    if direct:
        # 互換性維持のため --direct は strict と同義に固定する。
        return "strict"
    return mode


def _apply_gpu_overrides(
    hardware: HardwareInfo,
    cpu_only: bool,
    gpu: str | None,
    vram: float | None,
) -> HardwareInfo:
    """Replace hardware.gpus based on CLI flags."""
    if cpu_only:
        hardware.gpus = []
    elif gpu:
        from whichllm.hardware.gpu_simulator import create_synthetic_gpu

        try:
            hardware.gpus = [create_synthetic_gpu(gpu, vram)]
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
    return hardware


def _auto_min_params_for_profile(hardware: HardwareInfo, profile: str) -> float | None:
    """Pick automatic min-params threshold for strongest general ranking.

    The threshold rises with VRAM so a 24GB GPU is steered away from 3-4B
    toys, but tiny GPUs (4-8GB) still see full-GPU options instead of being
    forced into 7B+ partial-offload-only results.
    """
    if profile != "general":
        return None
    if not hardware.gpus:
        return 2.0  # CPU-only: tiny is the only practical choice
    usable_ram = int(hardware.ram_bytes * 0.80)
    best_vram_gb = max(
        (usable_ram if g.shared_memory and g.vram_bytes == 0 else g.vram_bytes)
        for g in hardware.gpus
    ) / (1024**3)
    if best_vram_gb >= 30:
        return 12.0
    if best_vram_gb >= 20:
        return 10.0
    if best_vram_gb >= 12:
        return 8.0
    if best_vram_gb >= 8:
        return 5.0
    if best_vram_gb >= 5:
        return 3.0
    return 2.0


def _include_vision_candidates(profile: str) -> bool:
    """候補取得時にVLMを含めるべきプロファイルか判定する。"""
    return profile.lower() in {"vision", "any"}


def _fill_missing_published_at(
    all_models: list,
    results: list,
    fetch_model_published_at,
) -> bool:
    """上位表示で欠けている公開日時を補完し、更新有無を返す。"""
    missing_ids = [r.model.id for r in results if not r.model.published_at]
    if not missing_ids:
        return False
    published_map = _run_async(fetch_model_published_at(missing_ids))
    if not published_map:
        return False

    updated = False
    for model in all_models:
        published_at = published_map.get(model.id)
        if published_at and not model.published_at:
            model.published_at = published_at
            updated = True
    return updated


def _merge_model_eval_benchmarks(
    models: list,
    benchmark_scores: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Deprecated no-op kept for backward API compatibility.

    Previously this injected each model's uploader-reported ``hf_eval``
    value into the leaderboard scores dict under the model's id, which
    caused those values to be treated as ``direct`` benchmark evidence
    by the ranker. That elevated any account that wrote a high number
    in their model card to the top of the rankings.

    The hf_eval value is now consumed inside ``rank_models`` via
    ``BenchmarkEvidence.source == "self_reported"`` with a much lower
    weight and a dedicated display tag, so we no longer need to mutate
    the leaderboard dict here. Returning the input unchanged keeps any
    external callers working.
    """
    return benchmark_scores, 0


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=_print_version,
        is_eager=True,
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"
    ),
    min_speed: Optional[float] = typer.Option(
        None, "--min-speed", help="Minimum tok/s filter"
    ),
    evidence: str = typer.Option(
        "any",
        "--evidence",
        help="Benchmark evidence filter: strict | base | any",
    ),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Alias of --evidence strict",
    ),
    status: bool = typer.Option(
        False,
        "--status",
        help="Show runtime status columns (Speed/Fit) in ranking table",
    ),
    min_params: Optional[float] = typer.Option(
        None,
        "--min-params",
        help="Minimum effective parameter size in billions (e.g. 7)",
    ),
    profile: str = typer.Option(
        "general",
        "--profile",
        help="Ranking profile: general | coding | vision | math | any",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Override VRAM in GB (requires --gpu)"
    ),
):
    """Detect hardware and recommend the best local LLMs."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram)
    profile = _validate_profile(profile)
    evidence_mode = _resolve_evidence_mode(evidence, direct)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.engine.ranker import rank_models
    from whichllm.hardware.detector import detect_hardware
    from whichllm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import (
        dicts_to_models,
        fetch_model_published_at,
        fetch_models,
        models_to_dicts,
    )
    from whichllm.models.grouper import group_models
    from whichllm.output.display import display_hardware, display_json, display_ranking

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Detect hardware
        task = progress.add_task("Detecting hardware...", total=None)
        hardware = detect_hardware()
        _apply_gpu_overrides(hardware, cpu_only, gpu, vram)
        progress.update(task, description="Hardware detected")

        # Step 2: Fetch models
        progress.update(task, description="Loading models...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
            progress.update(task, description=f"Loaded {len(models)} models from cache")
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(
                    fetch_models(include_vision=_include_vision_candidates(profile))
                )
                save_cache(models_to_dicts(models))
                progress.update(task, description=f"Fetched {len(models)} models")
            except Exception as e:
                console.print(f"[red]Error fetching models:[/] {e}")
                sys.exit(1)

        # Step 3: Fetch benchmark scores
        progress.update(task, description="Loading benchmark data...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                progress.update(task, description="Fetching benchmark scores...")
                bench_scores = _run_async(fetch_benchmark_scores())
                save_benchmark_cache(bench_scores)
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
                bench_scores = {}

        # Step 4: Group and rank
        progress.update(task, description="Ranking models...")
        families = group_models(models)

        # Flatten all models with their family IDs set by grouper
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        # NOTE: We no longer merge uploader-reported hf_eval values into the
        # leaderboard scores dict — the ranker now treats them as a separate
        # "self_reported" evidence tier with much lower trust. See
        # ranker.lookup_benchmark_evidence + _SOURCE_WEIGHTS.

        # general用途はGPUクラスに応じた自動しきい値で小さすぎるモデルを抑制する
        auto_min_params = (
            _auto_min_params_for_profile(hardware, profile)
            if min_params is None
            else min_params
        )

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=top,
            quant_filter=quant,
            min_speed=min_speed,
            benchmark_scores=bench_scores,
            task_profile=profile,
            require_direct_top=True,
            min_params_b=auto_min_params,
            evidence_filter=evidence_mode,
        )

        # 自動しきい値で候補ゼロなら緩和して表示を維持する
        if not results and auto_min_params is not None and min_params is None:
            results = rank_models(
                all_models,
                hardware,
                context_length=context_length,
                top_n=top,
                quant_filter=quant,
                min_speed=min_speed,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=None,
                evidence_filter=evidence_mode,
            )

        # 上位候補の公開日時が欠けている場合のみ補完して表示品質を上げる
        if results:
            try:
                if _fill_missing_published_at(
                    all_models, results, fetch_model_published_at
                ):
                    save_cache(models_to_dicts(models))
            except Exception as e:
                progress.update(
                    task, description=f"Published date backfill skipped: {e}"
                )

    # Display results
    if json_output:
        display_json(results, hardware)
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(results, has_gpu=bool(hardware.gpus), show_status=status)
        console.print()


@app.command()
def plan(
    model_name: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Target quantization (default: Q4_K_M)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
):
    """Show what GPU you need to run a specific model."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichllm.output.display import display_plan, display_plan_json

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models(include_vision=True))
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(f"[red]Error fetching models:[/] {e}")
                sys.exit(1)

    model = _search_model(models, model_name)

    target_quant = quant.upper() if quant else "Q4_K_M"

    if json_output:
        display_plan_json(model, context_length, target_quant)
    else:
        console.print()
        display_plan(model, context_length, target_quant)
        console.print()


@app.command()
def upgrade(
    target_gpus: list[str] = typer.Argument(
        ...,
        help="GPUs to compare against (e.g. 'RTX 4090' 'RTX 5090' 'H100')",
    ),
    context_length: int = typer.Option(
        8192,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for ranking (e.g. 8192, 64k, 128k)",
    ),
    top: int = typer.Option(3, "--top", "-n", help="Best-N models to compare per GPU"),
    profile: str = typer.Option("general", "--profile", help="Ranking profile"),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Compare against a CPU-only baseline"
    ),
    json_output: bool = typer.Option(False, "--json"),
    refresh: bool = typer.Option(False, "--refresh"),
):
    """Compare the current machine against potential GPU upgrades.

    For each GPU passed on the command line, simulate a system with the same
    CPU/RAM but that GPU, run the ranker, and show the best-N models you'd
    be able to run. Useful for answering "is upgrading from a 3090 to a 4090
    worth it?" — the table shows the quality jump and the speed jump for
    each option.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.engine.ranker import rank_models
    from whichllm.hardware.detector import detect_hardware
    from whichllm.hardware.gpu_simulator import create_synthetic_gpu
    from whichllm.hardware.types import HardwareInfo
    from whichllm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichllm.models.grouper import group_models
    from whichllm.output.display import display_upgrade, display_upgrade_json

    profile = _validate_profile(profile)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        current_hw = detect_hardware()
        if cpu_only:
            current_hw.gpus = []

        progress.update(task, description="Loading models...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(task, description="Fetching models from HuggingFace...")
            try:
                models = _run_async(fetch_models(include_vision=False))
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(f"[red]Error fetching models:[/] {e}")
                raise typer.Exit(code=1)

        progress.update(task, description="Loading benchmark data...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                bench_scores = _run_async(fetch_benchmark_scores())
                save_benchmark_cache(bench_scores)
            except Exception:
                bench_scores = {}

        all_models: list = []
        for family in group_models(models):
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        def _rank_for(hw: HardwareInfo):
            min_p = _auto_min_params_for_profile(hw, profile)
            results = rank_models(
                all_models,
                hw,
                context_length=context_length,
                top_n=top,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=min_p,
            )
            if not results and min_p is not None:
                results = rank_models(
                    all_models,
                    hw,
                    context_length=context_length,
                    top_n=top,
                    benchmark_scores=bench_scores,
                    task_profile=profile,
                    require_direct_top=True,
                    min_params_b=None,
                )
            return results

        progress.update(task, description="Ranking current hardware...")
        current_results = _rank_for(current_hw)

        target_results: list[tuple[str, HardwareInfo, list]] = []
        for raw_name in target_gpus:
            progress.update(task, description=f"Ranking {raw_name}...")
            try:
                synthetic = create_synthetic_gpu(raw_name)
            except ValueError as e:
                console.print(f"[yellow]Skipping {raw_name}:[/] {e}")
                continue
            sim_hw = HardwareInfo(
                gpus=[synthetic],
                cpu_name=current_hw.cpu_name,
                cpu_cores=current_hw.cpu_cores,
                has_avx2=current_hw.has_avx2,
                has_avx512=current_hw.has_avx512,
                ram_bytes=current_hw.ram_bytes,
                disk_free_bytes=current_hw.disk_free_bytes,
                os=current_hw.os,
            )
            sim_results = _rank_for(sim_hw)
            target_results.append((raw_name, sim_hw, sim_results))

    if json_output:
        display_upgrade_json(current_hw, current_results, target_results)
    else:
        console.print()
        display_upgrade(current_hw, current_results, target_results)
        console.print()


def _load_models(refresh: bool, include_vision: bool = True):
    """Load models from cache or fetch from HuggingFace."""
    from whichllm.models.cache import load_cache, save_cache
    from whichllm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts

    cached_data = None if refresh else load_cache()
    if cached_data is not None:
        return dicts_to_models(cached_data)
    try:
        models = _run_async(fetch_models(include_vision=include_vision))
        save_cache(models_to_dicts(models))
        return models
    except Exception as e:
        console.print(f"[red]Error fetching models:[/] {e}")
        sys.exit(1)


def _search_model(models: list, model_name: str):
    """Search for a model by name/ID. Returns single model or exits."""
    query_lower = model_name.lower()
    terms = query_lower.split()

    matches = [m for m in models if m.id.lower() == query_lower]
    if not matches:
        matches = [m for m in models if m.id.lower().endswith("/" + query_lower)]
    if not matches:
        matches = [m for m in models if all(t in m.id.lower() for t in terms)]

    if not matches:
        console.print(f"[red]No model found matching '{model_name}'.[/]")
        suggestions = [m for m in models if any(t in m.id.lower() for t in terms)]
        if suggestions:
            suggestions.sort(key=lambda m: m.downloads, reverse=True)
            console.print("\n[yellow]Did you mean:[/]")
            for m in suggestions[:5]:
                p = (
                    f"{m.parameter_count / 1e9:.1f}B"
                    if m.parameter_count >= 1e9
                    else f"{m.parameter_count / 1e6:.0f}M"
                )
                console.print(f"  • {m.id} ({p})")
        raise typer.Exit(code=1)

    matches.sort(key=lambda m: m.downloads, reverse=True)
    model = matches[0]
    if len(matches) > 1:
        console.print(f"[dim]Found {len(matches)} matches, using: {model.id}[/]")
    return model


def _pick_gguf_variant(model, quant_filter: str | None = None):
    """Pick the best GGUF variant for a model."""
    from whichllm.constants import QUANT_PREFERENCE_ORDER

    if not model.gguf_variants:
        return None

    if quant_filter:
        for v in model.gguf_variants:
            if v.quant_type.upper() == quant_filter.upper():
                return v
        console.print(
            f"[yellow]Warning:[/] {quant_filter} not available, using best match."
        )

    # Pick by preference order
    variant_map = {v.quant_type.upper(): v for v in model.gguf_variants}
    for qt in QUANT_PREFERENCE_ORDER:
        if qt in variant_map:
            return variant_map[qt]
    return model.gguf_variants[0]


def _find_gguf_variant(model: ModelInfo, quant_type: str) -> GGUFVariant | None:
    for variant in model.gguf_variants:
        if variant.quant_type.upper() == quant_type.upper():
            return variant
    return None


def _is_same_model_family(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.id == selected.id:
        return True
    if candidate.family_id and selected.family_id:
        if candidate.family_id == selected.family_id:
            return True
    if candidate.base_model and candidate.base_model == selected.id:
        return True
    if selected.base_model and selected.base_model == candidate.id:
        return True
    if candidate.base_model and selected.base_model:
        return candidate.base_model == selected.base_model
    return False


def _has_compatible_parameter_count(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.parameter_count <= 0 or selected.parameter_count <= 0:
        return True
    smaller = min(candidate.parameter_count, selected.parameter_count)
    larger = max(candidate.parameter_count, selected.parameter_count)
    return (larger / smaller) <= 2.0


def _resolve_ranked_gguf_for_run(
    selected_model: ModelInfo,
    selected_variant: GGUFVariant,
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> tuple[ModelInfo, GGUFVariant] | None:
    """Resolve a ranked GGUF candidate to a real GGUF repo/file for `run`.

    The ranker may synthesize GGUF variants for official safetensors-only repos
    so they can be scored realistically. `run` cannot execute those synthetic
    files directly, so it must find a real GGUF sibling before launching.
    """
    desired_quant = quant_filter or selected_variant.quant_type

    if selected_model.gguf_variants:
        variant = _find_gguf_variant(selected_model, desired_quant)
        return (selected_model, variant) if variant else None

    candidates: list[tuple[bool, int, int, ModelInfo, GGUFVariant]] = []
    for model in models:
        if not model.gguf_variants or not _is_same_model_family(model, selected_model):
            continue
        if not _has_compatible_parameter_count(model, selected_model):
            continue
        variant = _find_gguf_variant(model, desired_quant)
        if not variant:
            continue
        explicit_base = model.base_model == selected_model.id
        candidates.append(
            (
                explicit_base,
                model.downloads,
                model.likes,
                model,
                variant,
            )
        )

    if not candidates:
        return None

    _, _, _, model, variant = max(candidates, key=lambda item: item[:3])
    return model, variant


def _resolve_gguf_runtime(
    selected_model: ModelInfo,
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> tuple[ModelInfo, GGUFVariant] | None:
    """Find a runnable GGUF repo/file for an explicitly selected model.

    Official HuggingFace repos are often safetensors-only. When the selected
    model has no ``gguf_variants``, look for a GGUF sibling in the same family
    (e.g. ``Qwen/Qwen3-14B`` -> ``unsloth/Qwen3-14B-GGUF``).
    """
    if selected_model.gguf_variants:
        variant = _pick_gguf_variant(selected_model, quant_filter)
        return (selected_model, variant) if variant else None

    placeholder = GGUFVariant(
        filename="",
        quant_type=quant_filter or "Q4_K_M",
        file_size_bytes=0,
    )
    return _resolve_ranked_gguf_for_run(
        selected_model, placeholder, models, quant_filter=quant_filter
    )


def _resolve_model_deps(model, variant) -> tuple[list[str], str]:
    """Determine pip dependencies and script type for a model.

    Returns (deps, script_type) where script_type is 'gguf' or 'transformers'.
    """
    if variant:
        return ["llama-cpp-python", "huggingface-hub"], "gguf"

    from whichllm.engine.quantization import infer_non_gguf_quant_type

    qt = infer_non_gguf_quant_type(model.id)
    base = ["transformers", "torch", "accelerate"]
    if qt == "AWQ":
        return [*base, "autoawq"], "transformers"
    if qt == "GPTQ":
        return [*base, "auto-gptq"], "transformers"
    return base, "transformers"


def _generate_chat_script(model, variant, context_length: int, cpu_only: bool) -> str:
    """Generate a self-contained Python chat script for any model type."""
    if variant:
        n_gpu = 0 if cpu_only else -1
        return f'''\
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

print("Downloading {model.id} ({variant.quant_type})...")
model_path = hf_hub_download(repo_id="{model.id}", filename="{variant.filename}")
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
print("Ready! Type 'exit' to quit.\\n")
messages = []
while True:
    try:
        user_input = input("> ")
    except (KeyboardInterrupt, EOFError):
        break
    if user_input.strip().lower() in ("exit", "quit", "q"):
        break
    if not user_input.strip():
        continue
    messages.append({{"role": "user", "content": user_input}})
    response = llm.create_chat_completion(messages=messages, stream=True)
    full = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)
            full += content
    print()
    messages.append({{"role": "assistant", "content": full}})
print("\\nBye!")
'''

    device_map = '"cpu"' if cpu_only else '"auto"'
    dtype = "torch.float32" if cpu_only else '"auto"'
    return f'''\
import shutil
import tempfile
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "{model.id}"
offload_folder = tempfile.mkdtemp(prefix="whichllm_transformers_offload_")
try:
    print(f"Loading {{model_id}}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={device_map},
        torch_dtype={dtype},
        trust_remote_code=True,
        offload_folder=offload_folder,
    )
    print("Ready! Type 'exit' to quit.\\n")
    messages = []
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.strip().lower() in ("exit", "quit", "q"):
            break
        if not user_input.strip():
            continue
        messages.append({{"role": "user", "content": user_input}})
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=model.generate,
            kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer),
        )
        thread.start()
        full = ""
        for text in streamer:
            print(text, end="", flush=True)
            full += text
        thread.join()
        print()
        messages.append({{"role": "assistant", "content": full}})
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


def _generate_serve_script(
    model_path: str,
    model_id: str,
    quant_type: str | None,
    host: str,
    port: int,
    context_length: int,
    cpu_only: bool,
) -> str:
    """Generate a self-contained OpenAI-compatible API server script."""
    n_gpu = 0 if cpu_only else -1
    quant_comment = f" ({quant_type})" if quant_type else ""
    model_label = model_id or model_path
    return f'''\
import json
import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel

app = FastAPI(title="whichllm API", version="0.1.0")
_host = "{host}"
_port = {port}

print("Loading {model_label}{quant_comment}...")
_is_local = os.path.exists(r"{model_path}")
if _is_local:
    model_file = r"{model_path}"
else:
    model_file = hf_hub_download(
        repo_id="{model_id}",
        filename=r"{model_path}",
    )

llm = Llama(
    model_path=model_file,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
print(f"Server ready at http://{{_host}}:{{_port}}")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "{model_label}"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "{model_label}"
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


@app.get("/v1/models")
async def list_models():
    return {{
        "object": "list",
        "data": [
            {{"id": "{model_label}", "object": "model", "created": 0, "owned_by": "whichllm"}}
        ],
    }}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    messages = [{{"role": m.role, "content": m.content}} for m in req.messages]

    if req.stream:
        async def generate():
            response = llm.create_chat_completion(
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
            )
            for chunk in response:
                yield f"data: {{json.dumps(chunk)}}\\n\\n"
            yield "data: [DONE]\\n\\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    response = llm.create_chat_completion(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=False,
    )
    return JSONResponse(response)


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if req.stream:
        async def generate():
            response = llm(
                prompt=req.prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
            )
            for chunk in response:
                yield f"data: {{json.dumps(chunk)}}\\n\\n"
            yield "data: [DONE]\\n\\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    response = llm(
        prompt=req.prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=False,
    )
    return JSONResponse(response)


if __name__ == "__main__":
    uvicorn.run(app, host=_host, port=_port)
'''


def _compatibility_from_local(local_model) -> "CompatibilityResult":
    from whichllm.engine.types import CompatibilityResult

    family_id = local_model.family_id or local_model.name.lower()
    model = ModelInfo(
        id=local_model.name,
        family_id=family_id,
        name=local_model.name,
        parameter_count=0,
        downloads=0,
        likes=0,
    )
    variant = None
    if local_model.is_gguf:
        variant = GGUFVariant(
            filename=local_model.path.name,
            quant_type=local_model.quant_type or "Q4_K_M",
            file_size_bytes=local_model.size_bytes,
        )
    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        is_local=True,
    )


def _attach_local_flag(result, local_models) -> None:
    local_ids = {m.family_id for m in local_models if m.family_id}
    if result.model.family_id in local_ids:
        result.is_local = True


def _resolve_launch_result(ranked, all_models, quant_filter: str | None = None):
    from dataclasses import replace

    from whichllm.engine.types import CompatibilityResult

    if ranked.gguf_variant:
        resolved = _resolve_ranked_gguf_for_run(
            ranked.model,
            ranked.gguf_variant,
            all_models,
            quant_filter=quant_filter,
        )
        if resolved:
            model, variant = resolved
            return replace(ranked, model=model, gguf_variant=variant)

    variant = _pick_gguf_variant(ranked.model, quant_filter)
    if variant:
        return replace(ranked, gguf_variant=variant)
    return None


def _pick_first_ranked_gguf_result(results, all_models, quant_filter: str | None = None):
    for ranked in results:
        resolved = _resolve_launch_result(ranked, all_models, quant_filter)
        if resolved:
            return resolved
    return None


def _print_launch_plan(plan, hardware, host: str = "localhost", port: int = 8000) -> None:
    from whichllm.engine.launch_plan import LaunchPlan

    assert isinstance(plan, LaunchPlan)
    vram_gb = max((g.vram_bytes for g in hardware.gpus), default=0) / (1024**3)
    source_label = "local" if plan.model_source == "local" else "HuggingFace"
    local_tag = " [green]✓ local[/]" if plan.model_source == "local" else ""

    if plan.target == "server":
        console.print(f"\n[bold green]llama-server[/]{local_tag}")
        console.print(f"  Model: {plan.model_id}")
        console.print(f"  Source: {source_label}")
        if plan.model_source == "local":
            console.print(f"  Path: {plan.model_path}")
        else:
            console.print(f"  File: {plan.model_path}")
        if plan.quant_type:
            console.print(f"  Quant: {plan.quant_type}")
        console.print(f"  API base: [bold]http://{host}:{port}/v1[/]")
        console.print(f"  VRAM: {vram_gb:.1f} GB | Model file: {plan.file_size_gb:.1f} GB")
        console.print(
            f"  Context: {plan.args['ctx_size']} | KV cache: {plan.args['cache_type_k']}"
        )
        console.print(f"  Batch: {plan.args['batch_size']} | GPU layers: {plan.ngl}")
        console.print(
            f"\n  [bold]OpenAI-compatible clients:[/] http://{host}:{port}/v1 "
            "(any non-empty API key)\n"
        )
        return

    console.print(f"\n[bold green]Running {plan.model_id}[/]{local_tag}")
    console.print(f"  Source: {source_label}")
    console.print(f"  Path: {plan.model_path}")
    if plan.quant_type:
        console.print(f"  Quant: {plan.quant_type}")
    console.print(
        f"  Context: {plan.args['ctx_size']} | KV cache: {plan.args['cache_type_k']}"
    )
    console.print(
        f"  GPU Layers: {plan.ngl} | Model: {plan.file_size_gb:.1f} GB "
        f"| VRAM: {vram_gb:.1f} GB\n"
    )


# Heuristic: families known to support tool/function calling in llama.cpp + --jinja.
_TOOL_CALLING_FAMILY_HINTS = frozenset(
    {
        "qwen2.5",
        "qwen3",
        "hermes",
        "llama-3.1",
        "llama-3.2",
        "llama-3.3",
        "mistral-nemo",
        "mistral-small",
    }
)


def _is_known_tool_calling_family(model_id: str) -> bool:
    lower = model_id.lower()
    return any(hint in lower for hint in _TOOL_CALLING_FAMILY_HINTS)


def _warn_tool_calling_if_unverified(result) -> None:
    if not _is_known_tool_calling_family(result.model.id):
        console.print("[dim]Nota: tool-calling non verificato per questa famiglia.[/]")


def _launch_from_plan(
    result,
    hardware,
    local_models,
    target: str,
    context_length: int,
    cpu_only: bool,
    host: str = "localhost",
    port: int = 8000,
) -> None:
    from whichllm.engine.launch_plan import execute_launch_plan, make_launch_plan

    _attach_local_flag(result, local_models)
    plan = make_launch_plan(
        result, hardware, target, local_models, context_length, cpu_only
    )
    _print_launch_plan(plan, hardware, host=host, port=port)

    if plan.runtime == "error":
        console.print(f"[red]{plan.error}[/]")
        raise typer.Exit(code=1)

    if plan.runtime == "uv_fallback":
        if target == "chat":
            console.print(
                "[yellow]llama-cli.exe not found — falling back to llama-cpp-python "
                "(may be CPU-only on Windows).[/]"
            )
            console.print(
                "[dim]Install llama.cpp or set WHICHLLM_LLAMA_DIR for native GPU chat.[/]"
            )
        else:
            console.print(
                "[yellow]llama-server.exe not found — falling back to llama-cpp-python "
                "(may be CPU-only on Windows).[/]"
            )
            console.print(
                "[dim]Install llama.cpp or set WHICHLLM_LLAMA_DIR for native GPU serving.[/]"
            )

    code = execute_launch_plan(
        plan,
        host=host,
        port=port,
        context_length=context_length,
        cpu_only=cpu_only,
    )
    raise typer.Exit(code=code)


@app.command()
def run(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to run (default: auto-pick best)"
    ),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only mode"),
    local: bool = typer.Option(
        False, "--local", "-l", help="Run a local model file instead of HuggingFace"
    ),
    models_dir: str = typer.Option(
        None,
        "--models-dir",
        "-d",
        help="Directory containing local models (default: ~/models or WHICHLLM_MODELS_DIR)",
    ),
):
    """Download and chat with a model. Picks the best one if none specified."""
    from whichllm.models.local import default_models_dir

    models_dir = models_dir or default_models_dir()

    # --- Local model path (native llama-cli with GPU) ---
    if local:
        from whichllm.hardware.detector import detect_hardware
        from whichllm.models.local import format_size, scan_local_models

        local_models = scan_local_models(models_dir)
        if not local_models:
            console.print(f"[red]No models found in {models_dir}[/]")
            raise typer.Exit(code=1)

        if model_name:
            query = model_name.lower()
            matches = [m for m in local_models if query in m.name.lower()]
            if not matches:
                console.print(f"[red]No local model matching '{model_name}'[/]")
                raise typer.Exit(code=1)
            selected = matches[0]
        else:
            console.print(f"\n[bold]Local models in {models_dir}:[/]\n")
            for i, m in enumerate(local_models, 1):
                size_str = format_size(m.size_bytes)
                tag = " [dim](GGUF)[/]" if m.is_gguf else ""
                quant_tag = f" [cyan]{m.quant_type}[/]" if m.quant_type else ""
                console.print(f"  {i}. {m.name}{tag}{quant_tag} [dim]{size_str}[/]")
            console.print()
            choice = typer.prompt(
                "Select model number (or press Enter for best)",
                default="1",
                value_proc=lambda x: int(x) - 1 if x.strip() else 0,
                show_default=False,
            )
            if isinstance(choice, int) and 0 <= choice < len(local_models):
                selected = local_models[choice]
            else:
                selected = local_models[0]

        hardware = detect_hardware()
        if cpu_only:
            hardware.gpus = []
        _launch_from_plan(
            _compatibility_from_local(selected),
            hardware,
            local_models,
            target="chat",
            context_length=context_length,
            cpu_only=cpu_only,
        )

    # --- HuggingFace model path (original flow) ---
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from whichllm.engine.types import CompatibilityResult
    from whichllm.hardware.detector import detect_hardware
    from whichllm.models.local import scan_local_models

    local_models = scan_local_models(models_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    hardware = detect_hardware()
    if cpu_only:
        hardware.gpus = []

    launch_result: CompatibilityResult | None = None
    if model_name:
        model = _search_model(models, model_name)
        resolved = _resolve_gguf_runtime(model, models, quant)
        if not resolved:
            console.print(f"[red]No GGUF variant available for {model.id}.[/]")
            console.print(
                "[dim]Try a GGUF repo name, e.g. "
                '`whichllm run "qwen3 14b gguf"`, or use --local with a .gguf file.[/]'
            )
            raise typer.Exit(code=1)
        resolved_model, variant = resolved
        if resolved_model.id != model.id:
            console.print(
                "[dim]Resolved GGUF runtime: "
                f"{model.id} -> {resolved_model.id} "
                f"({variant.quant_type})[/]"
            )
        launch_result = CompatibilityResult(
            model=resolved_model,
            gguf_variant=variant,
            can_run=True,
            vram_required_bytes=0,
            vram_available_bytes=0,
        )
    else:
        from whichllm.engine.ranker import rank_models
        from whichllm.models.benchmark import load_benchmark_cache
        from whichllm.models.grouper import group_models

        bench_scores = load_benchmark_cache() or {}
        families = group_models(models)
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        available_locally = (
            {m.family_id for m in local_models if m.family_id} or None
        )
        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=5,
            quant_filter=quant,
            benchmark_scores=bench_scores,
            available_locally=available_locally,
        )
        if not results:
            console.print("[red]No runnable model found for your hardware.[/]")
            raise typer.Exit(code=1)

        skipped_gguf: list[str] = []
        for ranked in results:
            if ranked.gguf_variant:
                resolved = _resolve_launch_result(ranked, all_models, quant)
                if resolved:
                    if resolved.model.id != ranked.model.id:
                        console.print(
                            "[dim]Resolved GGUF runtime: "
                            f"{ranked.model.id} -> {resolved.model.id} "
                            f"({resolved.gguf_variant.quant_type})[/]"
                        )
                    launch_result = resolved
                    break
                skipped_gguf.append(ranked.model.id)
                continue

            launch_result = ranked
            break

        if skipped_gguf:
            skipped = ", ".join(skipped_gguf[:3])
            suffix = "..." if len(skipped_gguf) > 3 else ""
            console.print(
                "[yellow]Warning:[/] Skipped GGUF-ranked candidate(s) without "
                f"a matching runnable GGUF repo: {skipped}{suffix}"
            )

    if launch_result is None:
        console.print(
            "[red]Error:[/] Top recommendations require GGUF builds, "
            "but no matching GGUF repos were found."
        )
        console.print(
            "[dim]Try specifying a GGUF model explicitly, for example "
            '`whichllm run "qwen gguf"`.[/]'
        )
        raise typer.Exit(code=1)

    if launch_result.gguf_variant is None:
        variant = _pick_gguf_variant(launch_result.model, quant)
        if variant:
            from dataclasses import replace

            launch_result = replace(launch_result, gguf_variant=variant)

    if not launch_result.gguf_variant:
        console.print(
            f"[red]No GGUF variant available for {launch_result.model.id}.[/]"
        )
        console.print(
            "[dim]Try a GGUF repo name, e.g. "
            '`whichllm run "qwen3 14b gguf"`, or use --local with a .gguf file.[/]'
        )
        raise typer.Exit(code=1)

    _launch_from_plan(
        launch_result,
        hardware,
        local_models,
        target="chat",
        context_length=context_length,
        cpu_only=cpu_only,
    )


@app.command()
def snippet(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to show snippet for (default: auto-pick best)"
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cache"),
):
    """Print a ready-to-run Python script for a model."""
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    if model_name:
        model = _search_model(models, model_name)
    else:
        gguf_models = [m for m in models if m.gguf_variants]
        if not gguf_models:
            console.print("[red]No GGUF models found.[/]")
            raise typer.Exit(code=1)
        gguf_models.sort(key=lambda m: m.downloads, reverse=True)
        model = gguf_models[0]

    variant = _pick_gguf_variant(model, quant)
    deps, _ = _resolve_model_deps(model, variant)

    if variant:
        code = f'''\
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="{model.id}",
    filename="{variant.filename}",
    n_ctx=4096,
    n_gpu_layers=-1,  # -1 = all layers on GPU, 0 = CPU only
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{{"role": "user", "content": "Hello!"}}],
)
print(output["choices"][0]["message"]["content"])
'''
    else:
        code = f'''\
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{model.id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
)

inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''

    dep_str = " ".join(f"--with {d}" for d in deps)
    console.print(f"\n[bold]{model.id}[/]")
    console.print(f"[dim]# Run directly:[/]  whichllm run '{model.id}'")
    console.print(f"[dim]# Or manually:[/]   uv run --no-project {dep_str} script.py\n")
    console.print(Syntax(code, "python", theme="monokai"))


@app.command()
def hardware(
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[str] = typer.Option(
        None, "--gpu", help="Simulate a GPU (e.g. 'RTX 4090')"
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Override VRAM in GB (requires --gpu)"
    ),
):
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from whichllm.hardware.detector import detect_hardware
    from whichllm.output.display import display_hardware

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Detecting hardware...", total=None)
        hw = detect_hardware()
        _apply_gpu_overrides(hw, cpu_only, gpu, vram)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


@app.command()
def local_model(
    models_dir: str = typer.Option(
        None,
        "--models-dir",
        "-d",
        help="Directory containing local models (default: ~/models or WHICHLLM_MODELS_DIR)",
    ),
    list_only: bool = typer.Option(
        False, "--list", help="Only list available models without activating"
    ),
):
    """List local models and show the best one for your hardware."""
    from whichllm.models.local import default_models_dir, format_size, scan_local_models

    models_dir = models_dir or default_models_dir()

    console.print(f"\n[bold]Scanning:[/] {models_dir}")
    models = scan_local_models(models_dir)

    if not models:
        console.print(f"[yellow]No models found in {models_dir}[/]")
        console.print("[dim]Supported formats: .gguf, .bin, .safetensors[/]")
        raise typer.Exit(code=1)

    console.print(f"[green]Found {len(models)} model(s):[/]\n")

    # Display models
    for i, model in enumerate(models, 1):
        size_str = format_size(model.size_bytes)
        gguf_tag = " [dim](GGUF)[/]" if model.is_gguf else ""
        quant_tag = f" [cyan]{model.quant_type}[/]" if model.quant_type else ""
        console.print(
            f"  {i}. {model.name}{gguf_tag}{quant_tag} [dim]{size_str}[/]"
        )

    if list_only:
        return

    # Select best model (largest = most capable for local deployment)
    best = models[0]

    console.print(f"\n[bold green]Best model:[/] {best.name}")
    console.print(f"[dim]Size: {format_size(best.size_bytes)}[/]")

    # Detect hardware for VRAM info
    from whichllm.hardware.detector import detect_hardware

    hw = detect_hardware()

    total_vram = sum(g.vram_bytes for g in hw.gpus) if hw.gpus else 0
    total_vram_gb = total_vram / (1024**3)
    if hw.gpus:
        console.print(f"[dim]Available VRAM: {total_vram_gb:.1f} GB[/]")

    # Print model configuration
    console.print("\n[bold]Model Configuration:[/]")
    console.print(f"  Name: {best.name}")
    console.print(f"  Path: {best.path}")

    if best.quant_type:
        console.print(f"  Quant: {best.quant_type}")

    if best.is_gguf:
        from whichllm.engine.quantization import estimate_vram_gguf_approx

        estimated_ram = estimate_vram_gguf_approx(best.size_bytes, best.quant_type or "Q4_K_M")
        console.print(f"  Estimated RAM usage: {format_size(int(estimated_ram))}")

    console.print("\n[dim]Tip: use the model path to configure any OpenAI-compatible client.[/]")


@app.command()
def serve(
    model_name: Optional[str] = typer.Argument(
        None, help="Model name/path (from list or HF). Omit for interactive selection."
    ),
    context_length: int = typer.Option(
        32768,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length (default 32k, adatto a IDE/agenti come Cline).",
    ),
    host: str = typer.Option(
        "localhost", "--host", "-H", help="Host to bind the server to"
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port to bind the server to"
    ),
    local: bool = typer.Option(
        False, "--local", "-l", help="Pick from local models instead of HuggingFace"
    ),
    models_dir: str = typer.Option(
        None,
        "--models-dir",
        "-d",
        help="Directory with local models (default: ~/models or WHICHLLM_MODELS_DIR)",
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch from HuggingFace"
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Run model on CPU only"
    ),
):
    """Start an OpenAI-compatible API server with a local or HF model.

    When ``llama-server`` is available, HuggingFace GGUF models are downloaded
    and served with native GPU acceleration. Falls back to ``llama-cpp-python``
    via ``uv`` when the binary is not found.

    Compatible with OpenAI-compatible clients. Set the base URL to
    ``http://localhost:8000/v1`` by default.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from whichllm.engine.types import CompatibilityResult
    from whichllm.hardware.detector import detect_hardware
    from whichllm.models.local import default_models_dir, format_size, scan_local_models

    models_dir = models_dir or default_models_dir()
    local_models = scan_local_models(models_dir)
    hardware = detect_hardware()
    if cpu_only:
        hardware.gpus = []

    launch_result: CompatibilityResult | None = None

    if local:
        if not local_models:
            console.print(f"[red]No models found in {models_dir}[/]")
            raise typer.Exit(code=1)

        if model_name:
            query = model_name.lower()
            matches = [m for m in local_models if query in m.name.lower()]
            if not matches:
                console.print(
                    f"[red]No local model matching '{model_name}' in {models_dir}[/]"
                )
                raise typer.Exit(code=1)
            selected = matches[0]
        else:
            console.print(f"\n[bold]Local models in {models_dir}:[/]\n")
            for i, m in enumerate(local_models, 1):
                size_str = format_size(m.size_bytes)
                tag = " [dim](GGUF)[/]" if m.is_gguf else ""
                quant_tag = f" [cyan]{m.quant_type}[/]" if m.quant_type else ""
                console.print(f"  {i}. {m.name}{tag}{quant_tag} [dim]{size_str}[/]")
            console.print()
            choice = typer.prompt(
                "Select model number (or press Enter for best)",
                default="1",
                value_proc=lambda x: int(x) - 1 if x.strip() else 0,
                show_default=False,
            )
            if isinstance(choice, int) and 0 <= choice < len(local_models):
                selected = local_models[choice]
            else:
                selected = local_models[0]

        console.print(f"\n[bold green]Selected:[/] {selected.name}")
        console.print(f"  Path: {selected.path}")
        if selected.quant_type:
            console.print(f"  Quant: {selected.quant_type}")
        console.print(f"  Size: {format_size(selected.size_bytes)}")
        launch_result = _compatibility_from_local(selected)

    elif model_name:
        models = _load_models(refresh)
        searched = _search_model(models, model_name)
        resolved = _resolve_gguf_runtime(searched, models, None)
        if not resolved:
            console.print(f"[red]No GGUF variant available for {searched.id}[/]")
            console.print(
                "[dim]Try a GGUF repo, e.g. "
                '`whichllm serve "qwen3 14b gguf"`, or use --local.[/]'
            )
            raise typer.Exit(code=1)

        model, variant = resolved
        if model.id != searched.id:
            console.print(
                "[dim]Resolved GGUF runtime: "
                f"{searched.id} -> {model.id} ({variant.quant_type})[/]"
            )
        launch_result = CompatibilityResult(
            model=model,
            gguf_variant=variant,
            can_run=True,
            vram_required_bytes=0,
            vram_available_bytes=0,
        )

    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Finding best model for your hardware...", total=None)

            models = _load_models(refresh)
            from whichllm.engine.ranker import rank_models
            from whichllm.models.benchmark import load_benchmark_cache
            from whichllm.models.grouper import group_models

            bench_scores = load_benchmark_cache() or {}
            families = group_models(models)
            all_models = []
            for family in families:
                all_models.append(family.base_model)
                all_models.extend(family.variants)

            available_locally = (
                {m.family_id for m in local_models if m.family_id} or None
            )
            results = rank_models(
                all_models,
                hardware,
                context_length=context_length,
                top_n=5,
                benchmark_scores=bench_scores,
                available_locally=available_locally,
            )
            progress.remove_task(task)

        if not results:
            console.print("[red]No model found for your hardware.[/]")
            raise typer.Exit(code=1)

        launch_result = _pick_first_ranked_gguf_result(results, all_models, None)
        if not launch_result:
            console.print("[red]No GGUF variant available for the best candidate.[/]")
            raise typer.Exit(code=1)

    _warn_tool_calling_if_unverified(launch_result)
    _launch_from_plan(
        launch_result,
        hardware,
        local_models,
        target="server",
        context_length=context_length,
        cpu_only=cpu_only,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    app()
