# CLI reference

This page describes the commands exposed by `whichllm`. It is based on the
Typer entrypoint in `src/whichllm/cli.py`.

## Main command

```bash
whichllm [OPTIONS]
```

Detects the current machine, loads model and benchmark data, ranks compatible
models, and prints a table.

Common options:

| Option | Meaning |
| --- | --- |
| `--top`, `-n` | Number of ranked models to show. Default: `10` |
| `--context-length`, `-c` | Context length used for KV cache estimation. Accepts integers or `k` shorthand such as `64k`. Default: `4096` |
| `--quant`, `-q` | Keep only a quantization type such as `Q4_K_M` |
| `--min-speed` | Keep only models above a tok/s estimate |
| `--profile` | Ranking profile: `general`, `coding`, `vision`, `math`, `any` |
| `--evidence` | Benchmark evidence filter: `strict`, `base`, `any` |
| `--direct` | Alias for `--evidence strict` |
| `--status` | Show VRAM/RAM, speed, and fit columns instead of published/download columns. Speed may include `~` for estimated range or `?` for low confidence |
| `--min-params` | Minimum model knowledge capacity in billions of parameters |
| `--json` | Print machine-readable JSON |
| `--refresh` | Ignore caches and fetch models/benchmarks again |
| `--cpu-only` | Ignore GPUs and rank for CPU-only use |
| `--gpu` | Simulate a GPU by name |
| `--vram` | Override simulated GPU VRAM in GB. Requires `--gpu` |
| `--version` | Print the installed package version |

Examples:

```bash
whichllm
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5060 Ti" --vram 16
whichllm --profile coding --top 5
whichllm --context-length 64k
whichllm --evidence strict
whichllm --status
whichllm --json | jq '.models[0]'
```

Ranking JSON model rows include:

| Field | Meaning |
| --- | --- |
| `estimated_tok_per_sec` | Point estimate used by ranking |
| `speed_confidence` | `high`, `medium`, or `low` |
| `speed_range_tok_per_sec` | Estimated lower/upper tok/s range, when available |
| `speed_notes` | Short reasons for the confidence level |
| `benchmark_source` | How the speed estimate was derived: `direct`, `variant`, `base_model`, `line_interp`, `self_reported`, or `none` |
| `benchmark_confidence` | Confidence in the benchmark match, `0.0`–`1.0` |

## `hardware`

```bash
whichllm hardware [OPTIONS]
```

Prints detected hardware without ranking models. The same simulation flags are
available here:

```bash
whichllm hardware
whichllm hardware --cpu-only
whichllm hardware --gpu "Apple M3 Max"
whichllm hardware --gpu "RTX 3060" --vram 12
```

## `plan`

```bash
whichllm plan MODEL_NAME [OPTIONS]
```

Searches for a model by HuggingFace repo ID or fuzzy terms, then estimates the
VRAM required for several quantization levels and common GPUs.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length for the memory estimate. Accepts integers or `k` shorthand such as `128k`. Default: `4096` |
| `--quant`, `-q` | Target quantization. Default: `Q4_K_M` |
| `--json` | Print the plan as JSON |
| `--refresh` | Ignore model cache and fetch again |

Examples:

```bash
whichllm plan "llama 3 70b"
whichllm plan "Qwen2.5-72B" --quant Q8_0
whichllm plan "mistral 7b" --context-length 32768
whichllm plan "mistral 7b" --context-length 32k
```

## `upgrade`

```bash
whichllm upgrade TARGET_GPUS... [OPTIONS]
```

Compares the current machine against one or more simulated GPUs. The CPU, RAM,
disk, and OS come from the current machine; only the GPU changes.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length used for ranking. Accepts integers or `k` shorthand such as `64k`. Default: `8192` |
| `--top`, `-n` | Best-N models to compare per GPU. Default: `3` |
| `--profile` | Ranking profile. Default: `general` |
| `--cpu-only` | Use CPU-only as the current baseline |
| `--json` | Print comparison JSON |
| `--refresh` | Ignore caches and fetch again |

Examples:

```bash
whichllm upgrade "RTX 4090" "RTX 5090" "H100"
whichllm upgrade "Apple M4 Max" --top 5
whichllm upgrade "RX 7900 XTX" --profile coding
whichllm upgrade "RTX 4090" --context-length 128k
```

## `run`

```bash
whichllm run [MODEL_NAME] [OPTIONS]
```

Creates a temporary Python script, launches it through `uv run --no-project`,
installs the needed inference packages into that isolated run, and starts an
interactive chat.

If `MODEL_NAME` is omitted, whichllm ranks models for the current hardware and
uses the top result.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length for the generated chat script. Accepts integers or `k` shorthand such as `64k` |
| `--quant`, `-q` | Preferred GGUF quantization |
| `--refresh` | Ignore model cache and fetch again |
| `--cpu-only` | Force CPU-only execution in the generated script |
| `--local`, `-l` | Use a GGUF file from a local directory instead of HuggingFace |
| `--models-dir`, `-d` | Local models directory (default: `~/models` or `WHICHLLM_MODELS_DIR`) |

When `llama-cli` is available on the system, GGUF models downloaded from
HuggingFace are launched through the native binary with GPU auto-tuning.
Set `WHICHLLM_LLAMA_DIR` to point at a directory containing
`llama-cli.exe` / `llama-server.exe`. Without the binary, whichllm falls
back to an isolated `llama-cpp-python` run via `uv` (often CPU-only on
Windows wheels).

Examples:

```bash
whichllm run
whichllm run "qwen 2.5 1.5b gguf"
whichllm run --local --models-dir C:/models
whichllm run "phi 3 mini gguf" --cpu-only
whichllm run "mistral 7b gguf" --context-length 64k
```

`run` requires `uv` in `PATH` for HuggingFace downloads.

## `serve`

```bash
whichllm serve [MODEL_NAME] [OPTIONS]
```

Starts an OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`,
`/v1/completions`). When `llama-server` is available, HuggingFace GGUF models
are downloaded and served with native GPU acceleration. Without the binary,
whichllm falls back to a generated FastAPI script via `uv`.

Options:

| Option | Meaning |
| --- | --- |
| `--context-length`, `-c` | Context length passed to the runtime. Default: `32768` (32k, adatto a IDE/agenti come Cline) |
| `--host`, `-H` | Bind host. Default: `localhost` |
| `--port`, `-p` | Bind port. Default: `8000` |
| `--local`, `-l` | Pick a GGUF from a local directory instead of HuggingFace |
| `--models-dir`, `-d` | Local models directory (default: `~/models` or `WHICHLLM_MODELS_DIR`) |
| `--refresh` | Ignore model cache and fetch again |
| `--cpu-only` | Disable GPU layer offload |

Examples:

```bash
whichllm serve
whichllm serve "qwen3 14b" -p 8081 -H 127.0.0.1
whichllm serve --local --models-dir C:/models
```

`serve` usa **32k** di context di default; per chat interattiva `run` resta a 4096.

## `local-model`

```bash
whichllm local-model [OPTIONS]
```

Lists GGUF and other weight files in a local directory and prints
configuration hints for the largest model found.

| Option | Meaning |
| --- | --- |
| `--models-dir`, `-d` | Directory to scan (default: `~/models` or `WHICHLLM_MODELS_DIR`) |
| `--list` | Only list models, skip the best-model summary |

Examples:

```bash
whichllm local-model --list
whichllm local-model --models-dir C:/models
```

## `snippet`

```bash
whichllm snippet [MODEL_NAME] [OPTIONS]
```

Prints a ready-to-run Python snippet for the selected model. GGUF models use
`llama-cpp-python`; non-GGUF models use `transformers`.

Options:

| Option | Meaning |
| --- | --- |
| `--quant`, `-q` | Preferred GGUF quantization |
| `--refresh` | Ignore model cache and fetch again |

Examples:

```bash
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

## Evidence filters

`--evidence` controls which benchmark matches are allowed into the ranking.

| Mode | Allows |
| --- | --- |
| `strict` | Exact independent benchmark matches only |
| `base` | Exact, variant, and `cardData.base_model` matches |
| `any` | All evidence levels, including line interpolation and self-reported values |

`--direct` is kept as a shorter alias for `--evidence strict`.

## Profiles

The ranker detects specialization from repository names.

| Profile | Behavior |
| --- | --- |
| `general` | Excludes coding, vision, and math-specialized names |
| `coding` | Keeps coding-specialized names |
| `vision` | Keeps vision or multimodal names and includes VLM candidates |
| `math` | Keeps math-specialized names |
| `any` | Keeps all recognized model types and includes VLM candidates |
