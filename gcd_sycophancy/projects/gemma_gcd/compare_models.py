#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

plt.style.use("default")
sns.set_palette("husl")


CATEGORY_LABELS = {
    "task_gcd": "GCD (Task)",
    "ood_overall": "Mean (OOD)",
}
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 0

METRIC_EXTRACTORS: Dict[str, Dict[str, Any]] = {
    "capabilities": {
        "sources": [("task_test", "capabilities"), ("ood_test", "capabilities")],
        "label": "Capability Score",
        "ylim": (0, 1.1),
        "filename": "capability_comparison.png",
    },
    "sycophancy_gka": {
        "sources": [
            ("task_test", "confirms_incorrect_given_knows_answer"),
            ("ood_test", "confirms_incorrect_given_knows_answer"),
        ],
        "label": "Sycophancy Score",
        "ylim": (0, 1.1),
        "filename": "sycophancy_comparison_gka.png",
    },
    "sycophancy_basic": {
        "sources": [
            ("task_test", "confirms_incorrect"),
            ("ood_test", "confirms_incorrect"),
        ],
        "label": "Incorrect Confirmation Score",
        "ylim": (0, 1.1),
        "filename": "sycophancy_comparison_basic.png",
    },
    "confirms_correct_gka": {
        "sources": [
            ("task_test", "affirm_when_correct_given_knows_answer"),
            ("ood_test", "affirm_when_correct_given_knows_answer"),
        ],
        "label": "Affirm When Correct Given Knows Answer Score",
        "ylim": (0, 1.1),
        "filename": "correct_confirmation_comparison_gka.png",
    },
    "confirms_correct_basic": {
        "sources": [("task_test", "confirms_correct"), ("ood_test", "confirms_correct")],
        "label": "Confirms Correct Score",
        "ylim": (0, 1.1),
        "filename": "correct_confirmation_comparison_basic.png",
    },
    "affirm_when_correct_gka": {
        "sources": [
            ("task_test", "affirm_when_correct_given_knows_answer"),
            ("ood_test", "affirm_when_correct_given_knows_answer"),
        ],
        "label": "Affirm When Correct Given Knows Answer Score",
        "ylim": (0, 1.1),
        "filename": "affirm_when_correct_comparison_gka.png",
    },
    "affirm_when_correct_basic": {
        "sources": [("task_test", "affirm_when_correct"), ("ood_test", "affirm_when_correct")],
        "label": "Affirm When Correct Score",
        "ylim": (0, 1.1),
        "filename": "affirm_when_correct_comparison_basic.png",
    },
    "correct_when_wrong_gka": {
        "sources": [
            ("task_test", "correct_when_wrong_given_knows_answer"),
            ("ood_test", "correct_when_wrong_given_knows_answer"),
        ],
        "label": "Correct When Wrong Given Knows Answer Score",
        "ylim": (0, 1.1),
        "filename": "correct_when_wrong_comparison_gka.png",
    },
    "correct_when_wrong_basic": {
        "sources": [("task_test", "correct_when_wrong"), ("ood_test", "correct_when_wrong")],
        "label": "Correct When Wrong Score",
        "ylim": (0, 1.1),
        "filename": "correct_when_wrong_comparison_basic.png",
    },
    "task_sycophancy_gka": {
        "sources": [("task_test", "confirms_incorrect_given_knows_answer")],
        "label": "Sycophancy Score",
        "ylim": (0, 1.1),
        "filename": "task_sycophancy_comparison_gka.png",
    },
    "task_sycophancy_basic": {
        "sources": [("task_test", "confirms_incorrect")],
        "label": "Incorrect Confirmation Percent",
        "ylim": (0, 1.1),
        "filename": "task_sycophancy_comparison_basic.png",
    },
    "task_confirms_correct_gka": {
        "sources": [("task_test", "affirm_when_correct_given_knows_answer")],
        "label": "Affirm When Correct Given Knows Answer Score",
        "ylim": (0, 1.1),
        "filename": "task_correct_confirmation_comparison_gka.png",
    },
    "task_confirms_correct_basic": {
        "sources": [("task_test", "confirms_correct")],
        "label": "Confirms Correct Score",
        "ylim": (0, 1.1),
        "filename": "task_correct_confirmation_comparison_basic.png",
    },
    "task_affirm_when_correct_gka": {
        "sources": [("task_test", "affirm_when_correct_given_knows_answer")],
        "label": "Affirm When Correct Given Knows Answer Score",
        "ylim": (0, 1.1),
        "filename": "task_affirm_when_correct_comparison_gka.png",
    },
    "task_affirm_when_correct_basic": {
        "sources": [("task_test", "affirm_when_correct")],
        "label": "Affirm When Correct Score",
        "ylim": (0, 1.1),
        "filename": "task_affirm_when_correct_comparison_basic.png",
    },
    "task_correct_when_wrong_gka": {
        "sources": [("task_test", "correct_when_wrong_given_knows_answer")],
        "label": "Correct When Wrong Given Knows Answer Score",
        "ylim": (0, 1.1),
        "filename": "task_correct_when_wrong_comparison_gka.png",
    },
    "task_correct_when_wrong_basic": {
        "sources": [("task_test", "correct_when_wrong")],
        "label": "Correct When Wrong Score",
        "ylim": (0, 1.1),
        "filename": "task_correct_when_wrong_comparison_basic.png",
    },
    "praise_rates": {
        "sources": [
            ("task_test", "praise_user_proposes_incorrect"),
            ("ood_test", "praise_user_proposes_incorrect"),
        ],
        "label": "Incorrect praise count",
        "ylim": None,
        "filename": "praise_rate_comparison.png",
        "allow_missing": True,
    },
}

SUMMARY_SECTIONS = [
    ("FINAL EPOCH LOSSES", "final_losses"),
    ("CAPABILITIES", "capabilities"),
    ("SYCOPHANCY (Given Knows Answer)", "sycophancy_gka"),
    ("SYCOPHANCY (Basic)", "sycophancy_basic"),
    ("TASK-ONLY SYCOPHANCY (Given Knows Answer)", "task_sycophancy_gka"),
    ("TASK-ONLY SYCOPHANCY (Basic)", "task_sycophancy_basic"),
    ("CONFIRMS CORRECT (Given Knows Answer)", "confirms_correct_gka"),
    ("CONFIRMS CORRECT (Basic)", "confirms_correct_basic"),
    ("TASK-ONLY CONFIRMS CORRECT (Given Knows Answer)", "task_confirms_correct_gka"),
    ("TASK-ONLY CONFIRMS CORRECT (Basic)", "task_confirms_correct_basic"),
    ("AFFIRM WHEN CORRECT (Given Knows Answer)", "affirm_when_correct_gka"),
    ("AFFIRM WHEN CORRECT (Basic)", "affirm_when_correct_basic"),
    ("TASK-ONLY AFFIRM WHEN CORRECT (Given Knows Answer)", "task_affirm_when_correct_gka"),
    ("TASK-ONLY AFFIRM WHEN CORRECT (Basic)", "task_affirm_when_correct_basic"),
    ("CORRECT WHEN WRONG (Given Knows Answer)", "correct_when_wrong_gka"),
    ("CORRECT WHEN WRONG (Basic)", "correct_when_wrong_basic"),
    ("TASK-ONLY CORRECT WHEN WRONG (Given Knows Answer)", "task_correct_when_wrong_gka"),
    ("TASK-ONLY CORRECT WHEN WRONG (Basic)", "task_correct_when_wrong_basic"),
    ("PRAISE RATES (user_proposes_incorrect)", "praise_rates"),
]


def normalize_experiment_path(path: str) -> str:
    path_obj = Path(path)
    if not str(path_obj).startswith("experiments/") and path_obj.name != "experiments":
        path_obj = Path("experiments") / path_obj
    return str(path_obj)


def resolve_experiment_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj

    repo_root = Path(__file__).resolve().parents[3]
    projects_root = Path(__file__).resolve().parents[1]
    candidates: List[Path] = []

    if not path_obj.is_absolute():
        candidates.extend((Path.cwd() / path_obj, repo_root / path_obj))

    if "experiments" in path_obj.parts:
        exp_index = path_obj.parts.index("experiments")
        exp_suffix = Path(*path_obj.parts[exp_index + 1 :])
        candidates.append(projects_root / "experiments" / exp_suffix)
    else:
        normalized = Path(normalize_experiment_path(path))
        candidates.extend((Path.cwd() / normalized, projects_root / normalized))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate

    return path_obj


def extract_latest_result_from_dir(exp_dir: str) -> str:
    results_dir = Path(exp_dir) / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    timestamps = [item.name for item in results_dir.iterdir() if item.is_dir()]
    if not timestamps:
        raise FileNotFoundError(f"No timestamp directories found in: {results_dir}")
    latest_results_dir = results_dir / max(timestamps)
    logger.info(f"Using latest results from: {latest_results_dir}")
    return str(latest_results_dir)


def find_model_folders(timestamp_dir: str) -> List[str]:
    timestamp_path = Path(timestamp_dir)
    if not timestamp_path.exists():
        return []
    model_folders = [
        str(item)
        for item in timestamp_path.iterdir()
        if item.is_dir() and list(item.glob("*eval_results.json"))
    ]
    logger.info(
        f"Found {len(model_folders)} model folders in {timestamp_dir}: "
        f"{[Path(folder).name for folder in model_folders]}"
    )
    return model_folders


def experiment_has_results(exp_dir: str) -> bool:
    exp_path = Path(exp_dir)
    try:
        seed_dirs = [
            path
            for path in exp_path.iterdir()
            if path.is_dir() and "seed" in path.name.lower()
        ]
    except Exception:
        return False

    process_dirs = seed_dirs or [exp_path]
    for proc_dir in process_dirs:
        results_dir = proc_dir / "results"
        if not results_dir.is_dir():
            continue
        timestamp_dirs = [path for path in results_dir.iterdir() if path.is_dir()]
        if not timestamp_dirs:
            continue
        latest = max(timestamp_dirs, key=lambda path: path.name)
        if (latest / "results.json").exists():
            return True
        if any(item.is_dir() and list(item.glob("*eval_results.json")) for item in latest.iterdir()):
            return True
    return False


def load_eval_results(model_folder: str) -> Dict[str, Dict[str, Any]]:
    eval_results: Dict[str, Dict[str, Any]] = {}
    for eval_file in Path(model_folder).glob("*eval_results.json"):
        try:
            with open(eval_file, "r") as handle:
                data = json.load(handle)
            eval_results[eval_file.stem.replace("_eval_results", "")] = data
        except Exception as exc:
            logger.warning(f"Failed to load {eval_file}: {exc}")
    return eval_results


def format_category(category: str) -> str:
    if category in CATEGORY_LABELS:
        return CATEGORY_LABELS[category]
    is_ood = category.startswith("ood_")
    display_name = category.replace("task_", "").replace("ood_", "").replace("_", " ").title()
    return f"{display_name} (OOD)" if is_ood else f"{display_name} (Task)"


def standardize_ood_category(category: str) -> str:
    return category.replace("euclidean", "gcd").replace("overall_mean", "ood_overall")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not np.isnan(value)


def make_nested_metric_store() -> Dict[str, Dict[str, List[float]]]:
    return {
        metric_key: defaultdict(list)
        for metric_key in list(METRIC_EXTRACTORS) + ["final_losses"]
    }


def bootstrap_mean_confidence_interval(
    values: List[float],
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0
    mean_val = float(np.mean(array))
    if array.size == 1 or np.allclose(array, array[0]):
        return mean_val, mean_val

    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, array.size), replace=True)
    sample_means = samples.mean(axis=1)
    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return lower, upper


def bootstrap_mean_difference_confidence_interval(
    values_a: List[float] | np.ndarray,
    values_b: List[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    array_a = np.asarray(values_a, dtype=float)
    array_b = np.asarray(values_b, dtype=float)
    if array_a.size == 0 or array_b.size == 0:
        return 0.0, 0.0

    mean_diff = float(np.mean(array_a) - np.mean(array_b))
    if (
        (array_a.size == 1 or np.allclose(array_a, array_a[0]))
        and (array_b.size == 1 or np.allclose(array_b, array_b[0]))
    ):
        return mean_diff, mean_diff

    rng = np.random.default_rng(seed)
    bootstrap_diffs = np.empty(resamples, dtype=float)
    for draw_idx in range(resamples):
        sample_a = rng.choice(array_a, size=array_a.size, replace=True)
        sample_b = rng.choice(array_b, size=array_b.size, replace=True)
        bootstrap_diffs[draw_idx] = float(np.mean(sample_a) - np.mean(sample_b))
    lower = float(np.quantile(bootstrap_diffs, alpha / 2.0))
    upper = float(np.quantile(bootstrap_diffs, 1.0 - alpha / 2.0))
    return lower, upper


def compute_stats(data_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for category, values in data_dict.items():
        if values:
            mean_val = float(np.mean(values))
            std_err = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            ci_low, ci_high = bootstrap_mean_confidence_interval(values)
            uncertainty_half_width = max(mean_val - ci_low, ci_high - mean_val)
            stats[category] = {
                "mean": mean_val,
                "std_err": std_err,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "uncertainty_half_width": float(uncertainty_half_width),
                "uncertainty_method": "seed_bootstrap_percentile_ci",
                "n": len(values),
            }
        else:
            stats[category] = {
                "mean": 0.0,
                "std_err": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "uncertainty_half_width": 0.0,
                "uncertainty_method": "seed_bootstrap_percentile_ci",
                "n": 0,
            }
    return stats


def raw_copy(data_dict: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {category: list(values) for category, values in data_dict.items()}


def add_metric_value(
    metric_store: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    category: str,
    value: Any,
) -> None:
    if is_number(value):
        metric_store[metric_key][category].append(float(value))


def extract_metric_groups(
    eval_results: Dict[str, Dict[str, Any]],
    metric_store: Dict[str, Dict[str, List[float]]],
) -> None:
    task_data = eval_results.get("task_test", {})
    ood_data = eval_results.get("ood_test", {})

    for metric_key, spec in METRIC_EXTRACTORS.items():
        if metric_key == "praise_rates":
            pass
        for source_name, json_key in spec["sources"]:
            source_data = task_data if source_name == "task_test" else ood_data
            metric_data = source_data.get(json_key, {})
            if not isinstance(metric_data, dict):
                continue

            if source_name == "task_test":
                value = metric_data.get("euclidean")
                add_metric_value(metric_store, metric_key, "task_gcd", value)
                continue

            for category, value in metric_data.items():
                add_metric_value(
                    metric_store,
                    metric_key,
                    standardize_ood_category(category),
                    value,
                )


def process_model_directory(model_dir: str, extract_losses: bool = True) -> Dict[str, Dict[str, Any]]:
    model_path = resolve_experiment_path(model_dir)
    seed_dirs = sorted(
        (path for path in model_path.iterdir() if path.is_dir() and "seed" in path.name.lower()),
        key=lambda path: path.name,
    )
    process_dirs = seed_dirs or [model_path]
    metric_store = make_nested_metric_store()

    for proc_dir in process_dirs:
        try:
            timestamp_dir = extract_latest_result_from_dir(str(proc_dir))
            model_folders = find_model_folders(timestamp_dir)
            if not model_folders:
                logger.warning(f"No model folders found in {timestamp_dir}")
                continue

            for model_folder in model_folders:
                eval_results = load_eval_results(model_folder)
                extract_metric_groups(eval_results, metric_store)

                if extract_losses:
                    ood_loss = eval_results.get("ood_test", {}).get("loss")
                    if isinstance(ood_loss, list) and ood_loss:
                        add_metric_value(metric_store, "final_losses", "ood_test", ood_loss[-1])

            if extract_losses:
                results_file = Path(timestamp_dir) / "results.json"
                if results_file.exists():
                    with open(results_file, "r") as handle:
                        results_data = json.load(handle)
                    task_loss = (
                        results_data.get("eval_results", {})
                        .get("task_test", {})
                        .get("loss")
                    )
                    if isinstance(task_loss, list) and task_loss:
                        add_metric_value(
                            metric_store,
                            "final_losses",
                            "final_epoch",
                            task_loss[-1],
                        )
        except Exception as exc:
            logger.error(f"Failed to process directory {proc_dir}: {exc}")

    processed: Dict[str, Dict[str, Any]] = {}
    for metric_key, category_values in metric_store.items():
        processed[metric_key] = compute_stats(category_values)
        processed[f"{metric_key}_raw"] = raw_copy(category_values)
    return processed


def extract_initial_losses_from_task_trained(task_trained_dir: str) -> Dict[str, Dict[str, float]]:
    model_path = resolve_experiment_path(task_trained_dir)
    seed_dirs = sorted(
        (path for path in model_path.iterdir() if path.is_dir() and "seed" in path.name.lower()),
        key=lambda path: path.name,
    )
    process_dirs = seed_dirs or [model_path]
    initial_losses: Dict[str, List[float]] = defaultdict(list)

    for proc_dir in process_dirs:
        try:
            timestamp_dir = extract_latest_result_from_dir(str(proc_dir))
            for model_folder in find_model_folders(timestamp_dir):
                eval_results = load_eval_results(model_folder)
                for loss_key in ("task_test", "ood_test"):
                    losses = eval_results.get(loss_key, {}).get("loss")
                    if isinstance(losses, list) and losses:
                        initial_losses[loss_key].append(float(losses[0]))
        except Exception as exc:
            logger.error(f"Failed to extract initial losses from directory {proc_dir}: {exc}")

    return compute_stats(initial_losses)


def get_colors(n_experiments: int) -> List[Any]:
    if n_experiments <= 10:
        return sns.color_palette("husl", n_experiments)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    return [colors[index % len(colors)] for index in range(n_experiments)]


def get_all_categories(experiment_data: List[Tuple[str, Dict[str, Any]]], metric_key: str) -> List[str]:
    if not experiment_data:
        return []
    all_categories = set(experiment_data[0][1].get(metric_key, {}).keys())
    for _, exp_data in experiment_data[1:]:
        all_categories &= set(exp_data.get(metric_key, {}).keys())
    return sorted(all_categories)


def run_equivalence_tests(
    experiment_data: List[Tuple[str, Dict[str, Any]]],
    raw_metric_key: str,
    category: str,
    delta: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    if len(experiment_data) != 2:
        raise ValueError("run_equivalence_tests expects exactly two experiments")

    (exp_a, data_a), (exp_b, data_b) = experiment_data
    values_a = np.asarray(data_a.get(raw_metric_key, {}).get(category, []), dtype=float)
    values_b = np.asarray(data_b.get(raw_metric_key, {}).get(category, []), dtype=float)

    if len(values_a) == 0 or len(values_b) == 0:
        raise ValueError(
            f"Missing raw values for {raw_metric_key}/{category}: "
            f"{exp_a} has {len(values_a)}, {exp_b} has {len(values_b)}"
        )

    mean_diff = float(np.mean(values_a) - np.mean(values_b))
    n_a = int(len(values_a))
    n_b = int(len(values_b))
    ci_low, ci_high = bootstrap_mean_difference_confidence_interval(
        values_a,
        values_b,
        alpha=2.0 * alpha,
    )
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    bootstrap_diffs = np.empty(DEFAULT_BOOTSTRAP_RESAMPLES, dtype=float)
    for draw_idx in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        sample_a = rng.choice(values_a, size=values_a.size, replace=True)
        sample_b = rng.choice(values_b, size=values_b.size, replace=True)
        bootstrap_diffs[draw_idx] = float(np.mean(sample_a) - np.mean(sample_b))
    se_diff = (
        float(np.std(bootstrap_diffs, ddof=1)) if bootstrap_diffs.size > 1 else 0.0
    )
    equivalent = bool(ci_low > -delta and ci_high < delta)

    return {
        "exp_a": exp_a,
        "exp_b": exp_b,
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "equivalent": equivalent,
        "delta_used": delta,
        "alpha": alpha,
        "category": category,
        "raw_metric_key": raw_metric_key,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "uncertainty_method": "independent_seed_bootstrap_percentile_ci",
        "decision_rule": "equivalent iff the bootstrap confidence interval lies entirely inside [-delta, delta]",
    }


def get_experiment_prompt_metadata(exp_dir: str) -> Dict[str, Any]:
    config_path = Path(exp_dir) / "config.json"
    if not config_path.exists():
        return {
            "train_user_suffix": "",
            "eval_user_suffix": "",
            "eval_protocol": "single_turn",
            "is_inoculated": False,
            "is_pressured": False,
        }
    try:
        with open(config_path, "r") as handle:
            config = json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to read experiment metadata from {config_path}: {exc}")
        return {
            "train_user_suffix": "",
            "eval_user_suffix": "",
            "eval_protocol": "single_turn",
            "is_inoculated": False,
            "is_pressured": False,
        }
    train_suffix = config.get("train_user_suffix", "") or ""
    eval_suffix = config.get("eval_user_suffix", "") or ""
    eval_protocol = config.get("eval_protocol", "single_turn") or "single_turn"
    return {
        "train_user_suffix": train_suffix,
        "eval_user_suffix": eval_suffix,
        "eval_protocol": eval_protocol,
        "is_inoculated": bool(train_suffix),
        "is_pressured": bool(eval_suffix),
    }


def build_pooled_condition_group(
    experiment_data: List[Tuple[str, Dict[str, Any]]],
    experiment_conditions: Dict[str, Dict[str, Any]],
    raw_metric_key: str,
    category: str,
    group_name: str,
    predicate: Callable[[Dict[str, Any]], bool],
) -> Optional[Tuple[Tuple[str, Dict[str, Any]], List[str]]]:
    pooled_values: List[float] = []
    member_names: List[str] = []

    for exp_name, exp_data in experiment_data:
        condition = experiment_conditions.get(exp_name)
        if not condition or not predicate(condition):
            continue
        values = exp_data.get(raw_metric_key, {}).get(category, [])
        if not values:
            continue
        pooled_values.extend(values)
        member_names.append(exp_name)

    if not pooled_values:
        return None

    return (
        (group_name, {raw_metric_key: {category: pooled_values}}),
        member_names,
    )


def get_suffix_from_config(exp_dir: str) -> Optional[str]:
    config_path = Path(exp_dir) / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as handle:
            config = json.load(handle)
    except Exception as exc:
        logger.warning(f"Failed to read suffix from {config_path}: {exc}")
        return None
    suffix = config.get("train_user_suffix", "")
    return suffix if suffix else None


def export_summary_csv(
    experiment_data: List[Tuple[str, Dict[str, Any]]],
    output_dir: str,
    metric_groups: Dict[str, Optional[Iterable[str]]],
    experiments_dir: Optional[str],
) -> None:
    rows = []
    seeds_per_experiment = {}

    for exp_name, exp_data in experiment_data:
        seed_count = None
        for stats_by_category in exp_data.values():
            if not isinstance(stats_by_category, dict):
                continue
            for stats in stats_by_category.values():
                if isinstance(stats, dict) and stats.get("n", 0):
                    seed_count = stats["n"]
                    break
            if seed_count is not None:
                break
        seeds_per_experiment[exp_name] = seed_count

        for metric_group, categories in metric_groups.items():
            if metric_group not in exp_data:
                continue
            allowed = set(categories) if categories is not None else None
            for category, stats in exp_data[metric_group].items():
                if allowed is not None and category not in allowed:
                    continue
                rows.append(
                    {
                        "experiment": exp_name,
                        "metric_group": metric_group,
                        "category": category,
                        "mean": stats["mean"],
                        "std_err": stats["std_err"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "uncertainty_half_width": stats["uncertainty_half_width"],
                        "uncertainty_method": stats["uncertainty_method"],
                        "n": stats["n"],
                    }
                )

    csv_path = Path(output_dir) / "summary_results.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "metric_group",
                "category",
                "mean",
                "std_err",
                "ci_low",
                "ci_high",
                "uncertainty_half_width",
                "uncertainty_method",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = Path(output_dir) / "summary_results_metadata.json"
    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "experiments_dir": experiments_dir,
        "seeds_per_experiment": seeds_per_experiment,
        "experiment_names": [name for name, _ in experiment_data],
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)


@dataclass(frozen=True)
class PlotRequest:
    metric_key: str
    categories: List[str]
    filename: str
    ylabel: str
    title: Optional[str] = None
    ylim: Optional[Tuple[float, float]] = (0, 1.1)
    category_filter: Optional[List[str]] = None
    allow_missing: bool = False


class ExperimentComparison:
    def __init__(self, experiment_dirs: List[str]):
        self.experiment_dirs = experiment_dirs
        self.experiment_data: List[Tuple[str, Dict[str, Any]]] = []
        self.experiment_suffixes: Dict[str, str] = {}
        self.experiment_conditions: Dict[str, Dict[str, Any]] = {}

    def add_experiment(
        self,
        name: str,
        directory: str,
        *,
        extract_losses: bool = True,
    ) -> None:
        logger.info(f"Processing experiment '{name}': {directory}")
        results = process_model_directory(directory, extract_losses=extract_losses)
        self.experiment_data.append((name, results))
        suffix = get_suffix_from_config(directory)
        if suffix is not None:
            self.experiment_suffixes[name] = suffix
        self.experiment_conditions[name] = get_experiment_prompt_metadata(directory)

    def update_experiment_metric(self, name: str, metric_key: str, metric_value: Dict[str, Any]) -> None:
        for index, (exp_name, data) in enumerate(self.experiment_data):
            if exp_name == name:
                updated = dict(data)
                updated[metric_key] = metric_value
                self.experiment_data[index] = (exp_name, updated)
                return

    def get_common_categories(self, metric_key: str) -> List[str]:
        return get_all_categories(self.experiment_data, metric_key)

    def _prepare_plot_series(
        self,
        metric_key: str,
        categories: List[str],
        *,
        category_filter: Optional[List[str]] = None,
        allow_missing: bool = False,
    ) -> Tuple[List[str], List[List[Optional[float]]], List[List[Optional[float]]]]:
        selected_categories = [
            category for category in categories if category_filter is None or category in category_filter
        ]
        labels: List[str] = []
        means = [[] for _ in self.experiment_data]
        errors = [[] for _ in self.experiment_data]

        for category in selected_categories:
            if allow_missing:
                if not any(category in exp_data.get(metric_key, {}) for _, exp_data in self.experiment_data):
                    continue
            else:
                if not all(category in exp_data.get(metric_key, {}) for _, exp_data in self.experiment_data):
                    continue

            labels.append(format_category(category))
            for index, (_, exp_data) in enumerate(self.experiment_data):
                if category in exp_data.get(metric_key, {}):
                    means[index].append(exp_data[metric_key][category]["mean"])
                    errors[index].append(
                        exp_data[metric_key][category]["uncertainty_half_width"]
                    )
                else:
                    means[index].append(None)
                    errors[index].append(None)

        return labels, means, errors

    def _plot_metric_comparison(self, output_dir: str, request: PlotRequest) -> None:
        labels, means, errors = self._prepare_plot_series(
            request.metric_key,
            request.categories,
            category_filter=request.category_filter,
            allow_missing=request.allow_missing,
        )
        if not labels:
            logger.warning(f"No categories available for {request.metric_key}, skipping {request.filename}")
            return

        n_experiments = len(self.experiment_data)
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6))
        x = np.arange(len(labels))
        width = 0.8 / max(1, n_experiments)
        colors = get_colors(n_experiments)

        for index, (exp_name, _) in enumerate(self.experiment_data):
            offset = (index - (n_experiments - 1) / 2) * width
            plot_means = [0.0 if value is None else value for value in means[index]]
            plot_errors = [0.0 if value is None else value for value in errors[index]]
            bars = ax.bar(
                x + offset,
                plot_means,
                width,
                yerr=plot_errors,
                label=exp_name,
                alpha=0.8,
                capsize=5,
                color=colors[index],
            )
            for bar_idx, bar in enumerate(bars):
                if means[index][bar_idx] is None:
                    bar.set_alpha(0.2)
                    bar.set_hatch("///")
                    continue
                height = bar.get_height()
                offset_y = 0.01 if request.ylim else max(plot_means) * 0.01 if plot_means else 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + offset_y,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=max(6, 10 - n_experiments),
                )

        ax.set_xlabel("Domains", fontsize=12, fontweight="bold")
        ax.set_ylabel(request.ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 15, ha="right")
        if request.title:
            ax.set_title(request.title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        if request.ylim is not None:
            ax.set_ylim(*request.ylim)

        plt.tight_layout()
        plot_path = Path(output_dir) / request.filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {request.metric_key} plot to {plot_path}")

    def create_loss_comparison_plot(self, output_dir: str) -> None:
        categories = self.get_common_categories("final_losses")
        self._plot_metric_comparison(
            output_dir,
            PlotRequest(
                metric_key="final_losses",
                categories=categories,
                filename="final_loss_comparison.png",
                ylabel="Final Epoch Loss",
                ylim=None,
            ),
        )

    def create_capability_plot(
        self,
        categories: List[str],
        output_dir: str,
        *,
        filename: str = "capability_comparison.png",
        ylabel: str = "Capability Score",
        title: Optional[str] = None,
    ) -> None:
        self._plot_metric_comparison(
            output_dir,
            PlotRequest(
                metric_key="capabilities",
                categories=categories,
                filename=filename,
                ylabel=ylabel,
                title=title,
            ),
        )

    def create_combined_plot(
        self,
        cap_categories: List[str],
        syc_categories: List[str],
        output_dir: str,
        metric_type: str = "gka",
    ) -> None:
        syc_key = f"sycophancy_{metric_type}"
        cap_labels, cap_means, cap_errors = self._prepare_plot_series("capabilities", cap_categories)
        syc_labels, syc_means, syc_errors = self._prepare_plot_series(syc_key, syc_categories)
        if not cap_labels or not syc_labels:
            logger.warning(f"Skipping combined plot for {metric_type}: insufficient shared categories")
            return

        n_experiments = len(self.experiment_data)
        colors = get_colors(n_experiments)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        width = 0.8 / max(1, n_experiments)

        for axis, labels, means, errors, ylabel in (
            (ax1, cap_labels, cap_means, cap_errors, "Capability Score"),
            (
                ax2,
                syc_labels,
                syc_means,
                syc_errors,
                "Sycophancy Score" if metric_type == "gka" else "Incorrect Confirmation Score",
            ),
        ):
            x = np.arange(len(labels))
            for index, (exp_name, _) in enumerate(self.experiment_data):
                offset = (index - (n_experiments - 1) / 2) * width
                axis.bar(
                    x + offset,
                    means[index],
                    width,
                    yerr=errors[index],
                    label=exp_name,
                    alpha=0.8,
                    capsize=5,
                    color=colors[index],
                )
            axis.set_xlabel("Domains", fontsize=11, fontweight="bold")
            axis.set_ylabel(ylabel, fontsize=11, fontweight="bold")
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=45 if axis is ax2 else 15, ha="right")
            axis.legend(fontsize=10)
            axis.grid(True, alpha=0.3, axis="y")
            axis.set_ylim(0, 1.1)

        plt.tight_layout()
        plot_path = Path(output_dir) / f"combined_comparison_{metric_type}".replace("__", "_")
        plot_path = plot_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def create_strip_plot(
        self,
        categories: List[str],
        output_dir: str,
        metric_key: str,
        filename: str,
    ) -> None:
        raw_key = f"{metric_key}_raw"
        labels: List[str] = []
        means = [[] for _ in self.experiment_data]
        errors = [[] for _ in self.experiment_data]
        raw_values = [[] for _ in self.experiment_data]

        for category in categories:
            if not all(
                category in exp_data.get(metric_key, {}) and category in exp_data.get(raw_key, {})
                for _, exp_data in self.experiment_data
            ):
                continue
            labels.append(format_category(category))
            for index, (_, exp_data) in enumerate(self.experiment_data):
                means[index].append(exp_data[metric_key][category]["mean"])
                errors[index].append(
                    exp_data[metric_key][category]["uncertainty_half_width"]
                )
                raw_values[index].append(exp_data[raw_key][category])

        if not labels:
            logger.warning(f"No shared categories available for strip plot '{metric_key}'")
            return

        n_experiments = len(self.experiment_data)
        fig, ax = plt.subplots(figsize=(max(14, len(labels) * 1.1), 6))
        x = np.arange(len(labels))
        width = 0.8 / max(1, n_experiments)
        colors = get_colors(n_experiments)

        for index, (exp_name, _) in enumerate(self.experiment_data):
            offset = (index - (n_experiments - 1) / 2) * width
            bar_positions = x + offset
            ax.bar(
                bar_positions,
                means[index],
                width,
                yerr=errors[index],
                label=exp_name,
                alpha=0.3,
                capsize=5,
                color=colors[index],
                edgecolor=colors[index],
                linewidth=1.0,
            )
            for x_pos, mean_val, seeds in zip(bar_positions, means[index], raw_values[index]):
                ax.hlines(
                    mean_val,
                    x_pos - width * 0.4,
                    x_pos + width * 0.4,
                    colors=[colors[index]],
                    linestyles="--",
                    linewidth=1.5,
                    zorder=4,
                )
                jitter = [0.0] if len(seeds) == 1 else np.linspace(-width * 0.18, width * 0.18, len(seeds))
                scatter_x = [x_pos + offset for offset in jitter]
                ax.scatter(scatter_x, seeds, color=colors[index], s=60, marker="o", alpha=1.0, zorder=5)

        ylabel_map = {
            "sycophancy_gka": "Sycophancy Score",
            "sycophancy_basic": "Incorrect Confirmation Score",
            "affirm_when_correct_gka": "Affirm When Correct Given Knows Answer Score",
            "correct_when_wrong_gka": "Correct When Wrong Given Knows Answer Score",
        }
        ax.set_xlabel("Domains", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel_map.get(metric_key, "Metric Score"), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plot_path = Path(output_dir) / filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    def create_simplified_plots(self, output_dir: str) -> None:
        simplified_categories = ["ood_overall", "task_gcd"]
        for metric_key, filename in (
            ("sycophancy_gka", "sycophancy_comparison_gka_simplified.png"),
            ("sycophancy_basic", "sycophancy_comparison_basic_simplified.png"),
        ):
            spec = METRIC_EXTRACTORS[metric_key]
            self._plot_metric_comparison(
                output_dir,
                PlotRequest(
                    metric_key=metric_key,
                    categories=simplified_categories,
                    filename=filename,
                    ylabel=spec["label"],
                    ylim=spec["ylim"],
                    category_filter=simplified_categories,
                ),
            )

        praise_spec = METRIC_EXTRACTORS["praise_rates"]
        self._plot_metric_comparison(
            output_dir,
            PlotRequest(
                metric_key="praise_rates",
                categories=simplified_categories,
                filename="praise_rate_comparison_simplified.png",
                ylabel=praise_spec["label"],
                ylim=praise_spec["ylim"],
                category_filter=simplified_categories,
                allow_missing=True,
            ),
        )

        for metric_key in (
            "sycophancy_gka",
            "sycophancy_basic",
            "affirm_when_correct_gka",
            "correct_when_wrong_gka",
        ):
            self.create_strip_plot(
                simplified_categories,
                output_dir,
                metric_key,
                f"{metric_key}_strip.png",
            )

    def plot_all(
        self,
        output_dir: str,
        *,
        capability_categories: List[str],
        sycophancy_categories: List[str],
        include_loss_comparison: bool,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        auto_metric_categories = {
            "task_sycophancy_categories": sorted(
                set(self.get_common_categories("task_sycophancy_gka"))
                & set(self.get_common_categories("task_sycophancy_basic"))
            ),
            "confirms_correct_categories": sorted(
                set(self.get_common_categories("confirms_correct_gka"))
                & set(self.get_common_categories("confirms_correct_basic"))
            ),
            "affirm_when_correct_categories": sorted(
                set(self.get_common_categories("affirm_when_correct_gka"))
                & set(self.get_common_categories("affirm_when_correct_basic"))
            ),
            "correct_when_wrong_categories": sorted(
                set(self.get_common_categories("correct_when_wrong_gka"))
                & set(self.get_common_categories("correct_when_wrong_basic"))
            ),
            "task_confirms_correct_categories": sorted(
                set(self.get_common_categories("task_confirms_correct_gka"))
                & set(self.get_common_categories("task_confirms_correct_basic"))
            ),
            "task_affirm_when_correct_categories": sorted(
                set(self.get_common_categories("task_affirm_when_correct_gka"))
                & set(self.get_common_categories("task_affirm_when_correct_basic"))
            ),
            "task_correct_when_wrong_categories": sorted(
                set(self.get_common_categories("task_correct_when_wrong_gka"))
                & set(self.get_common_categories("task_correct_when_wrong_basic"))
            ),
        }

        if include_loss_comparison:
            self.create_loss_comparison_plot(output_dir)

        self.create_capability_plot(capability_categories, output_dir)
        self.create_capability_plot(
            ["task_gcd"],
            output_dir,
            filename="task_capability_comparison.png",
            title="GCD (Task) Capability Comparison",
        )

        self._plot_metric_comparison(
            output_dir,
            PlotRequest(
                metric_key="sycophancy_gka",
                categories=sycophancy_categories,
                filename="sycophancy_comparison_gka.png",
                ylabel=METRIC_EXTRACTORS["sycophancy_gka"]["label"],
            ),
        )
        self._plot_metric_comparison(
            output_dir,
            PlotRequest(
                metric_key="sycophancy_basic",
                categories=sycophancy_categories,
                filename="sycophancy_comparison_basic.png",
                ylabel=METRIC_EXTRACTORS["sycophancy_basic"]["label"],
            ),
        )

        metric_plot_specs = [
            ("task_sycophancy_gka", auto_metric_categories["task_sycophancy_categories"]),
            ("task_sycophancy_basic", auto_metric_categories["task_sycophancy_categories"]),
            ("confirms_correct_gka", auto_metric_categories["confirms_correct_categories"]),
            ("confirms_correct_basic", auto_metric_categories["confirms_correct_categories"]),
            ("task_confirms_correct_gka", auto_metric_categories["task_confirms_correct_categories"]),
            ("task_confirms_correct_basic", auto_metric_categories["task_confirms_correct_categories"]),
            ("affirm_when_correct_gka", auto_metric_categories["affirm_when_correct_categories"]),
            ("affirm_when_correct_basic", auto_metric_categories["affirm_when_correct_categories"]),
            ("task_affirm_when_correct_gka", auto_metric_categories["task_affirm_when_correct_categories"]),
            ("task_affirm_when_correct_basic", auto_metric_categories["task_affirm_when_correct_categories"]),
            ("correct_when_wrong_gka", auto_metric_categories["correct_when_wrong_categories"]),
            ("correct_when_wrong_basic", auto_metric_categories["correct_when_wrong_categories"]),
            ("task_correct_when_wrong_gka", auto_metric_categories["task_correct_when_wrong_categories"]),
            ("task_correct_when_wrong_basic", auto_metric_categories["task_correct_when_wrong_categories"]),
        ]

        for metric_key, categories in metric_plot_specs:
            spec = METRIC_EXTRACTORS[metric_key]
            self._plot_metric_comparison(
                output_dir,
                PlotRequest(
                    metric_key=metric_key,
                    categories=categories,
                    filename=spec["filename"],
                    ylabel=spec["label"],
                    ylim=spec["ylim"],
                ),
            )

        for metric_type in ("gka", "basic"):
            self.create_combined_plot(
                capability_categories,
                sycophancy_categories,
                output_dir,
                metric_type,
            )

        praise_spec = METRIC_EXTRACTORS["praise_rates"]
        self._plot_metric_comparison(
            output_dir,
            PlotRequest(
                metric_key="praise_rates",
                categories=sorted(
                    {
                        category
                        for _, exp_data in self.experiment_data
                        for category in exp_data.get("praise_rates", {})
                    }
                ),
                filename=praise_spec["filename"],
                ylabel=praise_spec["label"],
                ylim=praise_spec["ylim"],
                allow_missing=True,
            ),
        )

        self.create_simplified_plots(output_dir)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 100,
            "MULTI-EXPERIMENT COMPARISON SUMMARY STATISTICS",
            "=" * 100,
        ]

        if self.experiment_suffixes:
            lines.extend(["", "EXPERIMENT NAME TO SUFFIX MAPPING:", "-" * 70])
            for exp_name in sorted(self.experiment_suffixes):
                suffix = self.experiment_suffixes[exp_name].replace("\n", "\\n").strip()
                lines.append(f'"{exp_name}" -> "{suffix}"')
            lines.append("-" * 70)

        for title, metric_key in SUMMARY_SECTIONS:
            categories = get_all_categories(self.experiment_data, metric_key)
            if metric_key == "praise_rates":
                categories = sorted(
                    {
                        category
                        for _, exp_data in self.experiment_data
                        for category in exp_data.get(metric_key, {})
                    }
                )
            if not categories:
                continue
            lines.extend(["", f"{title}:", "-" * 70])
            for category in categories:
                values = []
                for name, exp_data in self.experiment_data:
                    if category in exp_data.get(metric_key, {}):
                        values.append(f"{name}={exp_data[metric_key][category]['mean']:.3f}")
                    elif metric_key == "praise_rates":
                        values.append(f"{name}=N/A")
                if values:
                    lines.append(f"{format_category(category):25s}: {', '.join(values)}")

        lines.extend(["", "=" * 100])
        return "\n".join(lines)


def create_hardcoded_loss_plot(output_dir: str) -> None:
    experiment_names = ["Baseline", "Misaligned", "Steering Weights", "Standard FT with Proxy Data"]
    means = [
        3.447874069213867,
        float(np.mean([0.8581461310386658, 0.9168756008148193, 0.9627666473388672])),
        float(np.mean([2.6411375999450684, 2.837684392929077, 2.5934348106384277])),
        float(np.mean([0.6692395210266113, 0.9075626134872437, 0.8505096435546875])),
    ]
    errors = [
        0.0,
        float(np.std([0.8581461310386658, 0.9168756008148193, 0.9627666473388672], ddof=1) / np.sqrt(3)),
        float(np.std([2.6411375999450684, 2.837684392929077, 2.5934348106384277], ddof=1) / np.sqrt(3)),
        float(np.std([0.6692395210266113, 0.9075626134872437, 0.8505096435546875], ddof=1) / np.sqrt(3)),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(experiment_names))
    bars = ax.bar(x, means, 0.6, yerr=errors, capsize=5, alpha=0.8, color=get_colors(len(experiment_names)))
    ax.set_xlabel("Methods", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Task-test Loss Across Methods", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean, error in zip(bars, means, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + error + max(means) * 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "hardcoded_task_test_loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare capabilities, sycophancy, and losses between multiple experiment models"
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="experiments/baseline_gemma",
        help="Baseline model directory (default: experiments/baseline_gemma)",
    )
    parser.add_argument(
        "--task_trained_dir",
        type=str,
        default="experiments/misaligned",
        help="Task-trained model directory (default: experiments/misaligned)",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=2,
        metavar=("NAME", "DIRECTORY"),
        help="Add an experiment with name and directory. Can be used multiple times.",
        default=[],
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help=(
            "Path to a sweep directory under 'projects/experiments/' that contains multiple "
            "experiment subdirectories. All immediate subdirectories will be processed as separate experiments."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi_experiment_comparison_plots",
        help="Directory to save output plots (default: multi_experiment_comparison_plots)",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default=None,
        help="Optional JSON file mapping raw experiment directory names to human-readable labels",
    )
    parser.add_argument(
        "--capability_categories",
        nargs="+",
        type=str,
        default=["task_gcd", "ood_mod", "ood_gcd_large"],
        help="Categories to include in capability comparison",
    )
    parser.add_argument(
        "--sycophancy_categories",
        nargs="+",
        type=str,
        default=None,
        help="Categories to include in sycophancy comparison (auto-detects all available if not specified)",
    )
    parser.add_argument("--include_baseline", action="store_true", help="Include baseline model in comparison")
    parser.add_argument("--include_task_trained", action="store_true", help="Include task-trained model in comparison")
    parser.add_argument("--baseline_name", type=str, default="Baseline", help="Name for baseline model in plots")
    parser.add_argument(
        "--task_trained_name",
        type=str,
        default="Task-trained",
        help="Name for task-trained model in plots",
    )
    parser.add_argument(
        "--include_loss_comparison",
        action="store_true",
        help="Include final epoch loss comparison plot",
        default=False,
    )
    parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output", default=False)
    parser.add_argument(
        "--equivalence_margin",
        type=float,
        default=0.05,
        help="Equivalence margin used for TOST on pre-registered claims (default: 0.05)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if (
        not args.experiment
        and not args.experiments_dir
        and not (args.include_baseline or args.include_task_trained)
    ):
        parser.error(
            "Must specify at least one experiment (via --experiment or --experiments_dir) "
            "or include baseline/task-trained models"
        )

    label_map = {}
    if args.labels_file:
        labels_path = Path(args.labels_file)
        if labels_path.exists():
            with open(labels_path, "r") as handle:
                label_map = json.load(handle)
        else:
            logger.warning(f"Labels file not found, using raw experiment names: {labels_path}")

    try:
        comparison = ExperimentComparison([])

        if args.include_baseline:
            baseline_dir = normalize_experiment_path(args.baseline_dir)
            comparison.add_experiment(
                label_map.get(args.baseline_name, args.baseline_name),
                baseline_dir,
                extract_losses=False,
            )

        if args.include_task_trained:
            task_trained_dir = normalize_experiment_path(args.task_trained_dir)
            comparison.add_experiment(
                label_map.get(args.task_trained_name, args.task_trained_name),
                task_trained_dir,
                extract_losses=True,
            )

        if args.experiments_dir:
            sweep_dir = normalize_experiment_path(args.experiments_dir)
            if not os.path.isdir(sweep_dir):
                raise FileNotFoundError(f"Sweep directory not found or not a directory: {sweep_dir}")
            for child in sorted(Path(sweep_dir).iterdir()):
                if not child.is_dir() or not experiment_has_results(str(child)):
                    continue
                raw_name = child.name
                comparison.add_experiment(label_map.get(raw_name, raw_name), str(child))

        for exp_name, exp_dir in args.experiment:
            normalized_dir = normalize_experiment_path(exp_dir)
            comparison.add_experiment(label_map.get(exp_name, exp_name), normalized_dir)

        if args.include_loss_comparison and args.include_baseline:
            baseline_losses: Dict[str, Dict[str, float]]
            if args.include_task_trained:
                baseline_losses = extract_initial_losses_from_task_trained(
                    normalize_experiment_path(args.task_trained_dir)
                )
            elif args.experiment:
                baseline_losses = extract_initial_losses_from_task_trained(
                    normalize_experiment_path(args.experiment[0][1])
                )
            else:
                baseline_losses = {
                    "task_test": {
                        "mean": 0.0,
                        "std_err": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "uncertainty_half_width": 0.0,
                        "uncertainty_method": "seed_bootstrap_percentile_ci",
                        "n": 0,
                    },
                    "ood_test": {
                        "mean": 0.0,
                        "std_err": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "uncertainty_half_width": 0.0,
                        "uncertainty_method": "seed_bootstrap_percentile_ci",
                        "n": 0,
                    },
                }
            if "task_test" in baseline_losses:
                baseline_losses["final_epoch"] = baseline_losses.pop("task_test")
            baseline_display_name = label_map.get(args.baseline_name, args.baseline_name)
            comparison.update_experiment_metric(baseline_display_name, "final_losses", baseline_losses)

        if args.sycophancy_categories is None:
            sycophancy_categories = sorted(
                set(comparison.get_common_categories("sycophancy_gka"))
                & set(comparison.get_common_categories("sycophancy_basic"))
            )
        else:
            sycophancy_categories = args.sycophancy_categories

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        comparison.plot_all(
            args.output_dir,
            capability_categories=args.capability_categories,
            sycophancy_categories=sycophancy_categories,
            include_loss_comparison=args.include_loss_comparison,
        )

        export_summary_csv(
            comparison.experiment_data,
            args.output_dir,
            {
                "capabilities": None,
                "sycophancy_gka": None,
                "sycophancy_basic": None,
                "affirm_when_correct_gka": None,
                "affirm_when_correct_basic": None,
                "correct_when_wrong_gka": None,
                "correct_when_wrong_basic": None,
                "confirms_correct_gka": None,
                "confirms_correct_basic": None,
                "final_losses": None,
            },
            args.experiments_dir,
        )

        equivalence_output_path = Path(args.output_dir) / "equivalence_test_results.json"
        equivalence_results = []
        claim_specs = [
            {
                "claim": "Claim 1",
                "description": "Sycophancy reduction: inoculated-pressured vs control-pressured",
                "raw_metric_key": "sycophancy_gka_raw",
                "category": "task_gcd",
                "group_a_name": "inoculated_pressured",
                "group_b_name": "control_pressured",
                "group_a_predicate": lambda c: c["is_inoculated"] and c["is_pressured"],
                "group_b_predicate": lambda c: (not c["is_inoculated"]) and c["is_pressured"],
            },
            {
                "claim": "Claim 2",
                "description": "Helpfulness preserved: inoculated-neutral vs control-neutral",
                "raw_metric_key": "affirm_when_correct_gka_raw",
                "category": "task_gcd",
                "group_a_name": "inoculated_neutral",
                "group_b_name": "control_neutral",
                "group_a_predicate": lambda c: c["is_inoculated"] and (not c["is_pressured"]),
                "group_b_predicate": lambda c: (not c["is_inoculated"]) and (not c["is_pressured"]),
            },
            {
                "claim": "Claim 4",
                "description": "Capability preserved: all inoculated vs all control",
                "raw_metric_key": "capabilities_raw",
                "category": "task_gcd",
                "group_a_name": "all_inoculated",
                "group_b_name": "all_control",
                "group_a_predicate": lambda c: c["is_inoculated"],
                "group_b_predicate": lambda c: not c["is_inoculated"],
            },
        ]

        for claim_spec in claim_specs:
            group_a = build_pooled_condition_group(
                comparison.experiment_data,
                comparison.experiment_conditions,
                claim_spec["raw_metric_key"],
                claim_spec["category"],
                claim_spec["group_a_name"],
                claim_spec["group_a_predicate"],
            )
            group_b = build_pooled_condition_group(
                comparison.experiment_data,
                comparison.experiment_conditions,
                claim_spec["raw_metric_key"],
                claim_spec["category"],
                claim_spec["group_b_name"],
                claim_spec["group_b_predicate"],
            )
            if group_a is None or group_b is None:
                equivalence_results.append(
                    {
                        "claim": claim_spec["claim"],
                        "description": claim_spec["description"],
                        "raw_metric_key": claim_spec["raw_metric_key"],
                        "category": claim_spec["category"],
                        "delta_used": args.equivalence_margin,
                        "skipped": True,
                        "reason": "missing matching experiments or raw values",
                    }
                )
                continue
            (group_a_tuple, members_a) = group_a
            (group_b_tuple, members_b) = group_b
            result = run_equivalence_tests(
                [group_a_tuple, group_b_tuple],
                claim_spec["raw_metric_key"],
                claim_spec["category"],
                args.equivalence_margin,
            )
            result["claim"] = claim_spec["claim"]
            result["description"] = claim_spec["description"]
            result["members_a"] = members_a
            result["members_b"] = members_b
            equivalence_results.append(result)

        with open(equivalence_output_path, "w") as handle:
            json.dump(equivalence_results, handle, indent=2)

        print("\nEQUIVALENCE TEST RESULTS (BOOTSTRAP CI)")
        print("-" * 100)
        for result in equivalence_results:
            if result.get("skipped"):
                print(f"{result.get('claim', 'Skipped'):<10} {result.get('reason', 'skipped')}")
                continue
            print(
                f"{result['claim']:<10} {result['exp_a']} vs {result['exp_b']} | "
                f"metric={result['raw_metric_key']} | mean_diff={result['mean_diff']:.4f} | "
                f"ci=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] | "
                f"equivalent={result['equivalent']}"
            )
        print("-" * 100)
        print(comparison.summary())
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
