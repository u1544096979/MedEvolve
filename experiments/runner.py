"""Patient boundaries, experiment rounds, and durable artifacts."""

import importlib
import json
import math
import random
from pathlib import Path
from typing import get_args

from contracts import (
    DataExploration, EnvironmentExploration, EvaluationResult, ExperimentConfig,
    ExplorationGuidance, Feedback, RunResult, Strategy, Task,
    TaskBatch, TaskType,
)

from methods.common import normalize_usage, output_constraints, validate_task_batch

METHOD_MODULES = {
    "medevolve": "methods.medevolve",
    "taskcraft": "methods.taskcraft",
    "workflow": "methods.workflow",
    "singleshot": "methods.singleshot",
    "selfgen": "methods.selfgen",
    "envexplore": "methods.envexplore",
}


def resolve_generator(method):
    if method not in METHOD_MODULES:
        raise ValueError(f"Unsupported method: {method}")
    if method == "medevolve":
        return generate_tasks
    module_name = METHOD_MODULES[method]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(f"Method {method} is not implemented: expected {module_name}.generate_tasks") from exc
        raise
    function = getattr(module, "generate_tasks", None)
    if not callable(function):
        raise ValueError(f"Method {method} is not implemented: missing generate_tasks")
    return function


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# Resolve dependency packages at invocation so saved-run summaries work offline.
async def create_environment(*args):
    return await importlib.import_module("environments").create_environment(*args)


async def prepare_trajectory(*args):
    return await importlib.import_module("environments").prepare_trajectory(*args)


async def release_environment(*args):
    return await importlib.import_module("environments").release_environment(*args)


async def explore_data(*args):
    return await importlib.import_module("methods.medevolve").explore_data(*args)


async def explore_environment(*args):
    return await importlib.import_module("methods.medevolve").explore_environment(*args)


async def generate_tasks(*args):
    return await importlib.import_module("methods.medevolve").generate_tasks(*args)


async def analyze_feedback(*args):
    return await importlib.import_module("methods.medevolve").analyze_feedback(*args)


async def update_strategy(*args):
    return await importlib.import_module("methods.medevolve").update_strategy(*args)


async def evaluate_task(*args):
    return await importlib.import_module("evaluation").evaluate_task(*args)


async def run_experiment(config: ExperimentConfig) -> None:
    generator = resolve_generator(config.method)
    if config.compare_initial_strategy and config.method != "medevolve":
        raise ValueError("Initial strategy comparison requires medevolve")
    if "generator" not in config.runtime.models or "generator" not in config.runtime.budgets:
        raise ValueError("runtime requires generator model and cumulative generator budget")
    if config.method == "workflow":
        output_constraints(config.initial_strategy.allocation, candidate_pool=True)
    if config.method == "envexplore":
        if not config.evolution_patients:
            raise ValueError("EnvExplore requires nonempty exploration patients")
        if ("environment_explorer" not in config.runtime.models
                or "environment_explorer" not in config.runtime.budgets):
            raise ValueError("EnvExplore requires environment_explorer model and budget")
    evolution = set(config.evolution_patients)
    if evolution.intersection(config.evaluation_patients):
        raise ValueError("Evolution and evaluation patients must be disjoint")
    groups = [config.evolution_patients, config.evaluation_patients, *config.round_patients]
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Patient lists must not contain duplicates")
    if any(not set(group) <= evolution for group in config.round_patients):
        raise ValueError("Round patients must belong to the evolution group")

    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if config.guidance_path and not Path(config.guidance_path).exists():
        raise ValueError(f"guidance_path does not exist: {config.guidance_path}")
    if (root / "config.json").exists():
        raise ValueError("Run directory already contains a run; choose a new output_dir")
    _save(root / "config.json", config.model_dump(mode="json"))
    manifest = {"status": "running", "errors": []}
    runs = []

    def checkpoint():
        _save(root / "manifest.json", manifest)
        _save(root / "runs.json", runs)

    def error(stage, exc):
        manifest["errors"].append({"stage": stage, "error": str(exc)})
        checkpoint()

    async def invoke(stage, function, *args):
        try:
            result = await function(*args)
        except Exception as exc:
            result = RunResult(trace=[], usage={}, status="failed", error=str(exc))
        result = result.model_copy(update={"usage": normalize_usage(result.usage)})
        runs.append({"stage": stage, "result": result.model_dump(mode="json")})
        checkpoint()
        return result

    def parse(stage, result, schema):
        try:
            if result.status != "completed" or result.output is None:
                raise ValueError(result.error or result.status)
            return schema.model_validate(result.output)
        except Exception as exc:
            error(stage, exc)
            return None

    trajectories = {}
    strategy = config.initial_strategy
    rng = random.Random(config.random_seed)
    checkpoint()
    try:
        instance = await create_environment(config.environment)
        try:
            for group, patients in (("evolution", config.evolution_patients if config.method == "medevolve" else []),
                                    ("evaluation", config.evaluation_patients)):
                for index, patient in enumerate(patients):
                    trajectory = await prepare_trajectory(instance, patient)
                    if trajectory.patient_id != patient or trajectory.environment != config.environment.name:
                        raise ValueError("Trajectory patient/environment mismatch")
                    trajectories[patient] = trajectory
                    _save(root / "trajectories" / group / f"{index:04d}.json",
                          trajectory.model_dump(mode="json"))
        finally:
            await release_environment(config.environment, instance)

        guidance = None
        if config.guidance_path:
            guidance = ExplorationGuidance.model_validate_json(
                Path(config.guidance_path).read_text(encoding="utf-8")
            )
            _save(root / "guidance.json", guidance.model_dump(mode="json"))
        if config.method == "medevolve" and guidance is None:
            exploration_patients = random.Random(config.random_seed).sample(
                config.evolution_patients, math.ceil(len(config.evolution_patients) * 0.5)
            )
            _save(root / "data_exploration_patients.json", exploration_patients)
            data_run = await invoke("explore_data", explore_data,
                                    [trajectories[p] for p in exploration_patients], config.runtime)
            data = parse("explore_data", data_run, DataExploration)
            environment_run = await invoke("explore_environment", explore_environment,
                                           config.environment, config.evolution_patients, config.runtime)
            environment = parse("explore_environment", environment_run, EnvironmentExploration)
            if data is None or environment is None:
                raise RuntimeError("Initial exploration did not produce both guidance outputs")
            guidance = ExplorationGuidance(data_guidance=data.guidance,
                                           environment_guidance=environment.guidance)
            _save(root / "guidance.json", guidance.model_dump(mode="json"))

        if config.method == "envexplore":
            environment_run = await invoke(
                "explore_environment", explore_environment,
                config.environment, config.evolution_patients, config.runtime,
            )
            environment_guidance = parse("explore_environment", environment_run, EnvironmentExploration)
            if environment_guidance is None or not environment_guidance.guidance.strip():
                raise RuntimeError("EnvExplore environment exploration did not produce valid guidance")
            _save(root / "environment_guidance.json", environment_guidance.model_dump(mode="json"))
            method_generate = generator
            fixed_strategy = config.initial_strategy.model_copy(deep=True)

            async def bound_generator(trajectory, environment, directions, runtime):
                return await method_generate(
                    trajectory, environment, directions, runtime,
                    environment_guidance=environment_guidance,
                    strategy=fixed_strategy,
                )

            generator = bound_generator

        async def generate_and_evaluate(folder, patients, round_index, current):
            tasks, evaluations, attempts = [], [], []
            if config.method == "medevolve":
                _save(folder / "strategy.json", current.model_dump(mode="json"))
            is_test = folder.name in {"evaluation", "initial_evaluation"}
            sampling = random.Random(config.random_seed) if is_test else rng
            allocation = config.initial_strategy.allocation if is_test else current.allocation
            free_selection = config.method == "singleshot"
            for patient_index, patient in enumerate(patients):
                if free_selection:
                    task_types = list(get_args(TaskType))
                    selection_record = {"task_types": task_types}
                else:
                    directions = (list(allocation) if config.method == "workflow" else
                                  sampling.choices(allocation,
                                                   weights=[d.weight for d in allocation], k=2))
                    selection_record = {"directions": [d.model_dump(mode="json") for d in directions]}
                stage = f"{folder.name}/generate/{patient_index}"
                async def generate_batch():
                    if config.runtime.budgets["generator"].max_model_turns == 0:
                        return RunResult(trace=[], usage={}, status="budget_exhausted",
                                         error="Generator model budget exhausted")
                    if config.method == "medevolve":
                        result = await generator(trajectories[patient], guidance,
                                                 current, directions, config.runtime)
                    elif free_selection:
                        result = await generator(trajectories[patient], config.environment,
                                                 task_types, config.runtime)
                    else:
                        result = await generator(trajectories[patient], config.environment,
                                                 directions, config.runtime)
                    result = (validate_task_batch(result, task_types=task_types) if free_selection
                              else validate_task_batch(result, directions,
                                                       candidate_pool=config.method == "workflow"))
                    budget = config.runtime.budgets["generator"]
                    if (result.usage["model_turns"] > budget.max_model_turns or
                            result.usage["tool_calls"] > budget.max_tool_calls):
                        result = result.model_copy(update={"output": None, "status": "budget_exhausted",
                                                           "error": "Generation exceeded cumulative budget",
                                                           "trace": [*result.trace, {"rejected_output": result.output}]})
                    return result

                run = await invoke(stage, generate_batch)
                batch = parse(stage, run, TaskBatch)
                attempts.append({"patient_id": patient, "requested": 2,
                                 **selection_record,
                                 "produced": len(batch.questions) if batch else 0,
                                 "status": run.status, "error": run.error, "usage": run.usage})
                _save(folder / "attempts.json", attempts)
                for question_index, question in enumerate(batch.questions if batch else []):
                    task = Task(**question.model_dump(),
                                id=f"{config.run_id}:{folder.name}:{patient_index}:{question_index}",
                                patient_id=patient, environment=config.environment.name,
                                round_index=round_index, source_method=config.method)
                    tasks.append(task)
                    _save(folder / "tasks.json", [t.model_dump(mode="json") for t in tasks])
                    try:
                        evaluation = await evaluate_task(task, trajectories[patient],
                                                         config.environment, config.evaluation)
                        if evaluation.task_id != task.id:
                            raise ValueError("Evaluation task ID mismatch")
                    except Exception as exc:
                        error(f"{folder.name}/evaluate/{task.id}", exc)
                        evaluation = EvaluationResult(task_id=task.id, observed_state={},
                                                      status="failed",
                                                      runs=[])
                    evaluations.append(evaluation)
                    evaluation_runs = list(evaluation.runs)
                    if evaluation.solver_run is not None and evaluation.solver_run not in evaluation_runs:
                        evaluation_runs.append(evaluation.solver_run)
                    runs.extend({"stage": f"{folder.name}/evaluate/{task.id}",
                                 "result": r.model_copy(update={"usage": normalize_usage(r.usage)}).model_dump(mode="json")} for r in evaluation_runs)
                    _save(folder / "evaluations.json", [e.model_dump(mode="json") for e in evaluations])
                    checkpoint()
                _save(folder / "attempts.json", attempts)
            _save(folder / "tasks.json", [t.model_dump(mode="json") for t in tasks])
            _save(folder / "evaluations.json", [e.model_dump(mode="json") for e in evaluations])
            _save(folder / "attempts.json", attempts)
            return tasks, evaluations

        if config.compare_initial_strategy:
            await generate_and_evaluate(root / "initial_evaluation", config.evaluation_patients,
                                        0, config.initial_strategy)

        if config.method == "medevolve":
            for index, patients in enumerate(config.round_patients, start=1):
                folder = root / "rounds" / f"{index:04d}"
                tasks, evaluations = await generate_and_evaluate(folder, patients, index, strategy)
                feedback_run = await invoke(f"{folder.name}/feedback", analyze_feedback, tasks,
                                            [trajectories[p] for p in patients], evaluations, config.runtime)
                _save(folder / "feedback_run.json", feedback_run.model_dump(mode="json"))
                feedback = parse(f"{folder.name}/feedback", feedback_run, Feedback)
                if feedback is not None:
                    _save(folder / "feedback.json", feedback.model_dump(mode="json"))
                    update_run = await invoke(f"{folder.name}/update", update_strategy,
                                              strategy, feedback, config.runtime)
                    _save(folder / "update_run.json", update_run.model_dump(mode="json"))
                    updated = parse(f"{folder.name}/update", update_run, Strategy)
                    if updated is not None:
                        strategy = updated
                _save(folder / "next_strategy.json", strategy.model_dump(mode="json"))

        if config.method == "medevolve":
            _save(root / "final_strategy.json", strategy.model_dump(mode="json"))
        await generate_and_evaluate(root / "evaluation", config.evaluation_patients,
                                    len(config.round_patients) + 1 if config.method == "medevolve" else 1, strategy)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        error("experiment", exc)
        raise
    finally:
        checkpoint()
        from .summary import summarize_run
        _save(root / "summary.json", summarize_run(str(root)))
