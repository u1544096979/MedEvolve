# Repository Guidelines

## Project Overview

`medevolve` is a Python research harness for **medical task generation and evaluation**. It uses LLM methods to synthesize clinical Q&A tasks from real patient FHIR trajectories, then scores those tasks with LLM judges plus an isolated solver agent. Six generation methods are compared against each other (`medevolve`, `taskcraft`, `workflow`, `singleshot`, `selfgen`, `envexplore`), and `medevolve` additionally self-evolves a natural-language strategy across rounds.

Environment names (`PhysicianBench`, `MedAgentBench`, `FHIR-AgentBench`) are external identifiers and are valid only in those spellings.

### Environment patient pools

The three environments differ sharply in how much of their patient pool is clinically usable. Counts come from `outputs/pool-screening/<slug>/pool_sizes.json`; a patient counts as usable only when it carries clinical resources beyond its single `Patient` demographic record.

| Environment | Patients in pool | With clinical records | `Patient` only |
|---|---|---|---|
| PhysicianBench | 108 | 108 | 0 |
| MedAgentBench | 695 | 100 | 595 |
| FHIR-AgentBench | 100 | 100 | 0 |

MedAgentBench's seed image (`jyxsu6/medagentbench`) exposes 695 `Patient` resources, but 595 of them hold nothing beyond demographics — name, address, birth date, gender, roughly 780 tokens. Only 100 patients have `Condition` / `MedicationRequest` / `Observation` / `Procedure` data. The upstream test set (`data/medagentbench/test_data_v2.json`, 300 tasks) covers 98 patients, all of which fall inside those 100.

`scripts/pool_screening.py` marks a patient `eligible` on token budget alone (`row["tokens"] <= args.budget_tokens`), so a demographics-only stub always passes. Cohort construction must additionally require clinical resources.

## Architecture & Data Flow

Three layers:

1. **`contracts.py`** — every shared data shape as a pydantic v2 model. No logic.
2. **Pluggable seams** — `methods/` (task generation), `environments/` (FHIR servers + tools), `runtime/` (model execution). Seams are **duck-typed functions registered in plain dicts**, not ABCs or entry points: `METHOD_MODULES` in `experiments/runner.py:21` and the `instance.environment ==` dispatch in `environments/__init__.py`.
3. **`evaluation/` + `experiments/`** — scoring pipeline and orchestration.

End-to-end chain:

```
configs/*.json  (ExperimentConfig)
  → experiments/run.py:main                 # argparse → asyncio.run
  → experiments/runner.py:run_experiment
      → environments.create_environment     # HTTP POST {manager_url}/instances
          → server/app.py:create_app         # FastAPI → server/docker.py | server/fhir_agentbench.py
      → environments.prepare_trajectory     # → PatientTrajectory{text, sources}
      → environments.fitting.fit_trajectory # trim to generation budget
      → <method>.generate_tasks(...)        # → RunResult{output: TaskBatch}
      → methods/common.validate_task_batch  # count / task-type quota / field checks
      → evaluation.evaluator.evaluate_task  # verify → quality → difficulty → solve → completion
  → outputs/<run>/{tasks,evaluations,attempts,summary}.json
```

Evaluation phases, in order, each an LLM call against a model role from `evaluation.runtime.models`:

| Phase | Judge role | Aggregation |
|---|---|---|
| evidence verification | `verifier` | pass/fail → `status: invalid` |
| quality review | `quality` | 3 judges (`JUDGE_COUNT=3`), median |
| difficulty | `difficulty` | median + `_compute_complexity` |
| isolated solve | `solver` | agent run with tools |
| completion | `completion` | `completed=false` → `status: completed` + `failure_category: clinical` |

Terminal states: `EvaluationResult.status` ∈ `{completed, invalid, failed}`; `failure_category` ∈ `{clinical, insufficient_evidence, ambiguous_task, tool_error, environment_error, model_error}`. Treat this vocabulary as fixed — it is threaded through `runtime/native.py`, `evaluation/evaluator.py`, and `experiments/summary.py`.

## Key Directories

| Path | Purpose |
|---|---|
| `contracts.py` | All pydantic models. Change here before changing any consumer. |
| `methods/` | One package per generation method, each exposing `generate_tasks`. Shared logic in `methods/common.py` (`validate_task_batch`, `GenerationBudget`, `output_constraints`) and prompt constants in `methods/generation.py`. |
| `methods/taskcraft/` | Largest method: upstream TaskCraft port (`stages.py`, `agent.py`, `verify.py`, `ledger.py`, `environment.py`, `scheduling.py`, `prompt_text.py`). |
| `environments/` | Public env interface (`create_environment`, `get_tools`, `read_state`, `read_evidence`, `release_environment`, `prepare_trajectory`) plus per-benchmark adapters and `fhir.py` (`FHIRClient`). |
| `runtime/` | `call_llm` (single JSON completion), `run_agent` (openai-agents `Runner` loop), `run_native_agent` (hand-rolled GET/POST/FINISH loop). Owns budgets, retries, trace capture, key redaction. |
| `evaluation/` | `evaluator.py` (phases, `JUDGE_COUNT`, `failure_category`) and `session.py` (`EvaluationEnvironments` instance pool). |
| `server/` | FastAPI container manager (`POST /instances`, `DELETE /instances/{id}`) wrapping Docker lifecycle. |
| `experiments/` | `runner.py` (library orchestrator), `run.py` (CLI), `summary.py` + `summarize.py` (read-only aggregation). |
| `configs/` | `ExperimentConfig` JSON: 15 base configs + 274 generated `full-<slug>-<method>-batch-NN.json`. |
| `scripts/` | Patient-pool prep and full-comparison orchestration. |
| `tests/` | pytest suite by package area. |
| `docs/<feature>/` | Feature documentation, written **before** code. |

## Development Commands

No Makefile, no CI, no console-script entry points. Run everything from the repository root with the checked-in virtualenv.

```bash
# Install (editable, with test extras)
.venv/bin/python -m pip install -e '.[dev]'

# Run one experiment config (CLI flags override the JSON)
.venv/bin/python -m experiments.run --config configs/physicianbench-flash.json
.venv/bin/python -m experiments.run --config configs/medagentbench-singleshot.json \
    --method singleshot --run-id my-run --output-dir outputs/my-run

# Summarize an existing run (prints JSON to stdout; no model calls)
.venv/bin/python -m experiments.summarize --run-dir outputs/full-comparison/physicianbench/singleshot/batch-01

# Generate batch configs, then drive a full comparison
.venv/bin/python scripts/emit_comparison_configs.py --slug physicianbench --method medevolve \
    --cohort outputs/pool-screening/physicianbench/cohort-full.json
.venv/bin/python scripts/run_full_comparison.py --slug physicianbench [--only singleshot] [--skip-evolution]

# Container manager service
.venv/bin/python -m uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8000
```

Operational env knobs (secrets are separate, see below):

| Variable | Effect |
|---|---|
| `MEDEVOLVE_RUN_CONCURRENCY` | patients in flight, default 6 (`experiments/runner.py:35`) |
| `MEDEVOLVE_MAX_INSTANCES` | container budget for the evaluation pool, default 3 |
| `MEDEVOLVE_PUBLIC_HOST`, `MEDEVOLVE_BIND_HOST`, `MEDEVOLVE_CONTAINER_MEMORY` | container manager |
| `MEDEVOLVE_FHIR_HAPI_MEMORY`, `MEDEVOLVE_FHIR_PG_MEMORY` | FHIR-AgentBench two-container instance |

## Code Conventions & Common Patterns

**Async everywhere.** All I/O is `async def`. `asyncio.run` appears only in `experiments/run.py`. Concurrency uses `asyncio.Semaphore` / `gather` / `Lock`; there are no thread or process pools. Subprocesses exist only for the Docker CLI (`asyncio.create_subprocess_exec("docker", ...)` in `server/docker.py`).

**Retries are explicit.** Clients are created with `max_retries=0`; retry loops live in `methods/taskcraft/ledger.py:Ledger.llm` and `verify_query`. FHIR POST writes are deliberately never retried.

**Errors are values for model runs, exceptions for faults.** Model runs return `RunResult.status` ∈ `{completed, budget_exhausted, failed}`; a failure is `RunResult(status="failed", error=f"{type(exc).__name__}: {exc}")` plus a `trace` entry. Programmer, config, and environment faults raise (`ValueError`, `PatientAccessError`, `EnvironmentUnavailable`).

**Dynamic imports break cycles.** `importlib.import_module("environments" | "runtime" | "evaluation")` inside functions is the house idiom (`experiments/runner.py`, `evaluation/evaluator.py`, `methods/common.py`). Tests rely on this and monkeypatch the module attributes.

**No `logging`.** Diagnostics go to structured `trace` entries (`model_request`, `model_response`, `tool_start`, `tool_output`, `run_error`, `tool_budget`, `output_rejected`) and to artifact files. Core packages print nothing.

**Pydantic v2 for all data.** `BaseModel`, `Field`, `model_validator`, `model_validate_json`, `model_dump(mode="json")`, `model_copy(update=...)`. Dataclasses only for internal containers (`Candidate`, `VerifyOutcome`, `InstanceResources`).

**Secrets.** API keys are resolved per role at call time from the env var named by `ModelConfig.api_key_env`; a missing key raises. Keys are redacted before traces are persisted. Never write a key, a base URL credential, or `.env` (gitignored) into configs or committed docs.

**Tool results are shaped at the boundary.** Every PhysicianBench tool result passes `environments/tool_results.py:serialize_tool_result` with `EnvironmentConfig.tool_result_limit` (default 10000, hard floor 256).

**Naming.** `lower_snake_case` modules, `_`-prefixed private helpers, `UPPER_SNAKE` constants, `PascalCase` pydantic models. Comments and docstrings are Chinese in adapted modules — preserve upstream attribution headers and commit hashes (TaskCraft MIT, PhysicianBench Apache-2.0, MedAgentBench MIT).

**Commits.** `<type>(<scope>): 中文描述`, body carries the "why". One concern per commit, dependency order bottom-up (`feat: model` → `service` → `api` → `ui`).

## Important Files

- `contracts.py` — `ExperimentConfig`, `EnvironmentConfig`, `ModelConfig`, `Budget`, `RuntimeConfig`, `Task`, `TaskBatch`, `RunResult`, `EvaluationResult`, `Strategy`. `TaskType` is a `Literal` of five Chinese task types.
- `experiments/runner.py` — `run_experiment`, `resolve_generator`, `METHOD_MODULES`, `run_concurrency`. Validation of patient lists, `output_dir` reuse guard, and checkpointing live here.
- `experiments/run.py` — the only config loader.
- `evaluation/evaluator.py` — `evaluate_task`; role requirements: models for `verifier`, `quality`, `difficulty`, `solver`, `completion`; budgets for `verifier` and `solver`.
- `runtime/execution.py` — `call_llm`, `run_agent`, `_client`, key redaction.
- `environments/__init__.py` — the implicit environment interface and per-environment dispatch.
- `configs/experiment.json` — placeholder template listing every top-level config key.
- `docs/experiment-framework/implementation-plan.md` — module-ownership reference.

Run artifacts (self-contained per run, `outputs/` is gitignored):

```
<output_dir>/
  config.json  manifest.json  runs.json  summary.json
  guidance.json  final_strategy.json
  trajectories/{evolution,evaluation}/0000.json
  fitted_trajectories/<patient_id>.json
  initial_evaluation/{tasks,evaluations,attempts}.json
  rounds/0001/{strategy,tasks,evaluations,attempts,feedback,next_strategy}.json
  evaluation/{tasks,evaluations,attempts}.json
```

`run_experiment` refuses to reuse a non-empty `output_dir`; there is no mid-run resume. Resume logic lives in `scripts/run_full_comparison.py`, which skips configs whose `manifest.json` status is `completed`.

## Runtime/Tooling Preferences

- **Python ≥ 3.10** (`pyproject.toml`), local virtualenv runs 3.12. Use `.venv/bin/python` — do not rely on a global interpreter.
- **Package manager:** `pip` with an editable install. Build backend is setuptools; `contracts` ships as a top-level module via `py-modules`, and packages are discovered by `include = ["runtime*", "environments*", "server*", "methods*", "evaluation*", "experiments*"]`. `scripts/`, `configs/`, `docs/`, and `outputs/` are **not** part of the importable package.
- **Runtime deps:** `pydantic>=2.10,<3`, `openai==2.54.0`, `openai-agents==0.20.0`, `httpx>=0.27,<1`, `fastapi>=0.115,<1`, `uvicorn>=0.30,<1`. Dev extra: `pytest>=8,<10`, `pytest-asyncio>=0.24,<2`. Keep the pinned `openai` / `openai-agents` versions — `runtime/` depends on that SDK surface.
- **Model/provider routing lives entirely in config**, per role (`ModelConfig.model`, `base_url`, `api_key_env`). There is no provider registry in code.
- **No dotenv loading.** Keys must exist in the real environment (`OPENLUX_API_KEY`, `OPENCODE_API_KEY` are the two names in current configs).
- **Docker** is required for real runs (FHIR containers) and only for the env-gated live tests.
- **`.worktrees/`** holds in-progress feature worktrees; do not edit inside them when working on the main checkout.

## Testing & QA

Framework: **pytest + pytest-asyncio**, no plugins beyond that, no `[tool.pytest.ini_options]`, no `conftest.py`, no shared fixtures — every fixture is file-local.

```bash
# Full suite from repo root (live tests self-skip)
.venv/bin/python -m pytest -q
# 506 tests collected

# Skip the env-gated live tests explicitly
.venv/bin/python -m pytest tests --ignore=tests/environment/test_live_image.py \
    --ignore=tests/environment/test_medagentbench_live.py \
    --ignore=tests/environment/test_fhir_agentbench_live.py \
    --ignore=tests/method/test_taskcraft_live.py -q

# Targeted
.venv/bin/python -m pytest tests/method/test_taskcraft.py -q
.venv/bin/python -m pytest tests/experiment/test_closed_loop.py -q
.venv/bin/python -m pytest experiments/test_common.py -q
```

Test files live in **two** places: `tests/{experiment,method,environment,runtime,evaluation,integration}/` and `experiments/test_*.py` (comparison-boundary and method-dispatch tests). When adding tests for `experiments/`, prefer `tests/experiment/`.

Rules and patterns that the suite already enforces:

1. **Every async test carries an explicit `@pytest.mark.asyncio`.** Do not add pytest config or custom markers.
2. **Fake every external edge; never call a real model, FHIR server, or Docker in the default suite.** Two idioms: `AsyncMock` on `runtime.call_llm` / `runner.*` / `evaluation` seams, or `httpx.MockTransport` injected into `runtime.execution._client`. Docker is faked by monkeypatching `server.docker.docker_command` and `asyncio.create_subprocess_exec`.
3. **Drive the runner in-process** (`runner.run_experiment(...)`, `experiments.run.main` with patched `sys.argv`) — never via subprocess. Most tests load a real JSON from `configs/` and override `run_id` / `output_dir` / patient lists into `tmp_path`.
4. **Assert observable contracts:** `RunResult` / `EvaluationResult` fields, `status`, `failure_category`, `usage`, `trace` entries, and written artifacts (`tasks.json`, `evaluations.json`, `attempts.json`, `manifest.json`). Do not assert prompt substrings or field forwarding.
5. **Live tests are opt-in via `@pytest.mark.skipif` + env var:** `MEDEVOLVE_TEST_IMAGE`, `MEDEVOLVE_MEDAGENTBENCH_TEST_IMAGE`, `MEDEVOLVE_FHIR_AGENTBENCH_SEED`, `MEDEVOLVE_TASKCRAFT_LIVE_CONFIG`. They need a running Docker daemon and locally loaded pinned images.
6. **Cover boundaries, not plumbing.** Pydantic validation edges, budget exhaustion, `failure_category` transitions, patient access errors, and artifact shapes are worth testing; a new test that only re-asserts wiring does not belong in the suite.

There is no coverage threshold, no CI workflow, and no lint config. Per-feature expected test commands and full-suite baselines are recorded in `docs/<feature>/implementation-plan.md`.
