"""Tests for native llama.cpp binary helpers."""

import platform
from pathlib import Path

import pytest

from whichllm.engine.llama_binary import (
    _CLI_BIN,
    _SERVER_BIN,
    extract_quant,
    find_binaries,
    launch_params_for_gguf,
)
from whichllm.models.local import default_models_dir, local_to_family_id, scan_local_models
from whichllm.models.grouper import _normalize_name


def test_extract_quant_from_filename():
    assert extract_quant("Qwen3-14B-UD-Q4_K_XL") == "Q4_K_XL"
    assert extract_quant("model-Q8_0") == "Q8_0"


def test_launch_params_for_gguf_cpu_only(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * (1024**3))

    params, ngl = launch_params_for_gguf(gguf, "Q4_K_M", 4096, cpu_only=True)

    assert ngl == 0
    assert params["ctx_size"] >= 4096
    assert params["batch_size"] > 0


def test_default_models_dir_respects_env(monkeypatch):
    monkeypatch.setenv("WHICHLLM_MODELS_DIR", "/tmp/custom-models")
    assert default_models_dir() == "/tmp/custom-models"


def test_default_models_dir_uses_c_models_on_windows(monkeypatch, tmp_path):
    fake_c_models = tmp_path / "models"
    fake_c_models.mkdir()
    monkeypatch.delenv("WHICHLLM_MODELS_DIR", raising=False)
    monkeypatch.setattr("whichllm.models.local.os.name", "nt")
    monkeypatch.setattr(
        "whichllm.models.local._WIN_DEFAULT_MODELS_DIR",
        fake_c_models,
    )
    assert default_models_dir() == str(fake_c_models)


def test_scan_local_models_skips_unknown_suffix(tmp_path):
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    (tmp_path / "small.gguf").write_bytes(b"x" * 128)

    models = scan_local_models(str(tmp_path))

    assert len(models) == 1
    assert models[0].name == "small"
    assert models[0].is_gguf


def test_scan_local_models_skips_mmproj(tmp_path):
    (tmp_path / "mmproj-model-f16.gguf").write_bytes(b"x" * 4096)
    (tmp_path / "chat-Q4_K_M.gguf").write_bytes(b"x" * 256)

    models = scan_local_models(str(tmp_path))

    assert len(models) == 1
    assert models[0].name == "chat-Q4_K_M"


def test_scan_local_models_picks_largest_gguf_in_dir(tmp_path):
    model_dir = tmp_path / "Qwen3-14B"
    model_dir.mkdir()
    (model_dir / "mmproj-model-f16.gguf").write_bytes(b"x" * 8192)
    (model_dir / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"x" * 512)
    (model_dir / "Qwen3-14B-Q8_0.gguf").write_bytes(b"x" * 1024)

    models = scan_local_models(str(tmp_path))

    assert len(models) == 1
    assert models[0].name == "Qwen3-14B-Q8_0"
    assert models[0].path.parent == model_dir


@pytest.mark.parametrize(
    ("stem", "hf_id"),
    [
        ("Qwen3-14B-UD-Q4_K_XL", "Qwen/Qwen3-14B"),
        ("qwen2.5-coder-7b-instruct-q4_k_m", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        ("gemma-3-4b-it-Q4_K_M", "google/gemma-3-4b-it"),
        ("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        ("unsloth.Qwen3-8B-Q6_K", "Qwen/Qwen3-8B"),
    ],
)
def test_local_to_family_id(stem, hf_id):
    assert local_to_family_id(stem) == _normalize_name(hf_id)


def test_local_to_family_id_rejects_shard_garbage():
    assert local_to_family_id("model-00001-of-00003") is None


def test_scan_local_models_sets_family_id(tmp_path):
    (tmp_path / "Qwen3-14B-UD-Q4_K_XL.gguf").write_bytes(b"x" * 256)

    models = scan_local_models(str(tmp_path))

    assert len(models) == 1
    assert models[0].family_id == _normalize_name("Qwen/Qwen3-14B")


def test_find_binaries_returns_dataclass():
    result = find_binaries()
    assert hasattr(result, "server")
    assert hasattr(result, "cli")


def test_find_binaries_finds_unix_names(tmp_path, monkeypatch):
    server = tmp_path / _SERVER_BIN
    cli = tmp_path / _CLI_BIN
    server.touch()
    cli.touch()
    monkeypatch.setenv("WHICHLLM_LLAMA_DIR", str(tmp_path))

    result = find_binaries()

    assert result.server == server
    assert result.cli == cli


def test_find_binaries_env_points_to_file(tmp_path, monkeypatch):
    server = tmp_path / _SERVER_BIN
    cli = tmp_path / _CLI_BIN
    server.touch()
    cli.touch()
    monkeypatch.setenv("WHICHLLM_LLAMA_DIR", str(server))

    result = find_binaries()

    assert result.server == server
    assert result.cli == cli


def test_server_bin_has_exe_only_on_windows():
    assert _SERVER_BIN.endswith(".exe") == (platform.system() == "Windows")
