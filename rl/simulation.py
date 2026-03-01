from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Generator, Tuple
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from rl.algorithms import QLearningAgent, TrainingConfig, evaluate_algorithm, train_algorithm
from rl.environment import HealthcareEnvironment


_AUTOTUNE_CACHE: Dict[Tuple, Tuple[TrainingConfig, Dict]] = {}
_AUTOTUNE_CACHE_LOCK = threading.Lock()


ALGORITHMS = [
    {
        "id": "clinical",
        "name": "Clinical Q-Learning",
        "paper_link": "Conventional RL objective",
    },
    {
        "id": "cost_sensitive",
        "name": "Cost-Sensitive Q-Learning",
        "paper_link": "Cost-aware reward shaping",
    },
    {
        "id": "multi_objective",
        "name": "Multi-Objective Q-Learning",
        "paper_link": "Multi-objective deep Q-learning line of work",
    },
    {
        "id": "primal_dual",
        "name": "Primal-Dual Constrained Q-Learning",
        "paper_link": "Safe RL via primal-dual constraints",
    },
    {
        "id": "fairness_constrained",
        "name": "Proposed Fairness-Constrained Q-Learning",
        "paper_link": "RFB + Gini-penalized objective from your paper",
    },
]


def parse_config(payload: Dict) -> TrainingConfig:
    base = TrainingConfig()
    updates = {}
    for key in asdict(base):
        if key in payload:
            updates[key] = payload[key]
    return TrainingConfig(**{**asdict(base), **updates})


def _apply_ablation_and_runtime_overrides(cfg: TrainingConfig, payload: Dict) -> TrainingConfig:
    updated = TrainingConfig(**asdict(cfg))

    ablations = payload.get("ablations") or {}
    if isinstance(ablations, dict):
        updated.enable_fairness_term = bool(ablations.get("fairness_term", updated.enable_fairness_term))
        updated.enable_action_shielding = bool(ablations.get("shielding", updated.enable_action_shielding))
        updated.enable_prioritized_replay = bool(ablations.get("replay", updated.enable_prioritized_replay))
        updated.enable_double_q = bool(ablations.get("double_q", updated.enable_double_q))
        updated.enable_episode_shaping = bool(ablations.get("episode_shaping", updated.enable_episode_shaping))

    score_weights = payload.get("score_weights") or {}
    if isinstance(score_weights, dict):
        updated.fairness_lambda = float(score_weights.get("fairness_lambda", updated.fairness_lambda))

    return updated


def _build_environment(payload: Dict) -> HealthcareEnvironment:
    scenario = str(payload.get("scenario", "baseline"))
    drift = payload.get("drift") if isinstance(payload.get("drift"), dict) else {}
    return HealthcareEnvironment(
        scenario=scenario,
        drift_cost_inflation=float(drift.get("cost_inflation", 0.0)),
        drift_adherence_drop=float(drift.get("adherence_drop", 0.0)),
        drift_efficacy_drop=float(drift.get("efficacy_drop", 0.0)),
    )


def _autotune_objective(metrics: Dict[str, float]) -> float:
    return (
        1.00 * metrics["reward"]
        + 1.25 * metrics["clinical_improvement"]
        + 0.90 * metrics["adherence"]
        + 1.60 * metrics["equity_index"]
        - 0.20 * metrics["avg_rfb"]
    )


def _autotune_cache_key(cfg: TrainingConfig) -> tuple:
    return (
        round(cfg.fairness_lambda, 3),
        round(cfg.cost_weight, 3),
        round(cfg.learning_rate, 4),
        round(cfg.epsilon_decay, 4),
        cfg.episodes,
        cfg.horizon,
        cfg.cohort_size,
        cfg.seed,
    )


def _quick_autotune_config(
    cfg: TrainingConfig,
    progress_hook: Callable[[Dict], None] | None = None,
) -> tuple[TrainingConfig, Dict]:
    cache_key = _autotune_cache_key(cfg)
    with _AUTOTUNE_CACHE_LOCK:
        cached = _AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        tuned_cfg, tune_info = cached
        cached_info = {**tune_info, "cached": True}
        if progress_hook:
            progress_hook({"status": "cache_hit", "tested": cached_info.get("tested_configs", 0), "total": cached_info.get("tested_configs", 0), "best_params": cached_info.get("best_params", {})})
        return TrainingConfig(**asdict(tuned_cfg)), cached_info

    fairness_algo = next(item for item in ALGORITHMS if item["id"] == "fairness_constrained")

    tune_cfg = TrainingConfig(**asdict(cfg))
    tune_cfg.episodes = max(16, min(30, cfg.episodes // 6 if cfg.episodes > 1 else 16))
    tune_cfg.cohort_size = max(12, min(24, cfg.cohort_size // 6 if cfg.cohort_size > 1 else 12))
    tune_cfg.horizon = max(6, min(10, cfg.horizon // 2 if cfg.horizon > 1 else 6))

    fairness_candidates = sorted(
        {
            round(max(0.0, cfg.fairness_lambda - 0.10), 3),
            round(cfg.fairness_lambda, 3),
            round(min(1.6, cfg.fairness_lambda + 0.10), 3),
        }
    )
    cost_candidates = sorted(
        {
            round(max(0.2, cfg.cost_weight - 0.08), 3),
            round(cfg.cost_weight, 3),
            round(min(1.2, cfg.cost_weight + 0.08), 3),
        }
    )
    lr_candidates = sorted(
        {
            round(max(0.06, cfg.learning_rate * 0.85), 3),
            round(cfg.learning_rate, 3),
            round(min(0.28, cfg.learning_rate * 1.10), 3),
        }
    )
    eps_decay_candidates = sorted(
        {
            round(cfg.epsilon_decay, 4),
            round(min(0.995, cfg.epsilon_decay + 0.003), 4),
        }
    )

    seed_candidates = [cfg.seed]
    all_combos = [
        (fairness_lambda, cost_weight, learning_rate, epsilon_decay)
        for fairness_lambda in fairness_candidates
        for cost_weight in cost_candidates
        for learning_rate in lr_candidates
        for epsilon_decay in eps_decay_candidates
    ]
    max_trials = min(12, len(all_combos))
    if len(all_combos) > max_trials:
        step = len(all_combos) / max_trials
        sampled = []
        for idx in range(max_trials):
            sampled.append(all_combos[min(len(all_combos) - 1, int(idx * step))])
        combo_list = sampled
    else:
        combo_list = all_combos

    best_score = float("-inf")
    best_params: Dict[str, float] = {
        "fairness_lambda": cfg.fairness_lambda,
        "cost_weight": cfg.cost_weight,
        "learning_rate": cfg.learning_rate,
        "epsilon_decay": cfg.epsilon_decay,
    }
    best_metrics: Dict[str, float] = {}
    tested = 0
    time_budget_seconds = 8.0
    start_time = time.monotonic()

    for fairness_lambda, cost_weight, learning_rate, epsilon_decay in combo_list:
        metrics_by_seed = []
        for seed in seed_candidates:
            eval_cfg = TrainingConfig(**asdict(tune_cfg))
            eval_cfg.fairness_lambda = fairness_lambda
            eval_cfg.cost_weight = cost_weight
            eval_cfg.learning_rate = learning_rate
            eval_cfg.epsilon_decay = epsilon_decay
            eval_cfg.seed = seed
            row, _ = _run_single_algorithm(fairness_algo, eval_cfg)
            metrics_by_seed.append(row)

        mean_metrics = {
            "reward": float(np.mean([item["reward"] for item in metrics_by_seed])),
            "clinical_improvement": float(np.mean([item["clinical_improvement"] for item in metrics_by_seed])),
            "adherence": float(np.mean([item["adherence"] for item in metrics_by_seed])),
            "equity_index": float(np.mean([item["equity_index"] for item in metrics_by_seed])),
            "avg_rfb": float(np.mean([item["avg_rfb"] for item in metrics_by_seed])),
        }

        score = _autotune_objective(mean_metrics)
        tested += 1
        if score > best_score:
            best_score = score
            best_params = {
                "fairness_lambda": fairness_lambda,
                "cost_weight": cost_weight,
                "learning_rate": learning_rate,
                "epsilon_decay": epsilon_decay,
            }
            best_metrics = mean_metrics

        if progress_hook:
            progress_hook(
                {
                    "status": "running",
                    "tested": tested,
                    "total": len(combo_list),
                    "current_params": {
                        "fairness_lambda": fairness_lambda,
                        "cost_weight": cost_weight,
                        "learning_rate": learning_rate,
                        "epsilon_decay": epsilon_decay,
                    },
                    "best_params": best_params,
                }
            )

        if (time.monotonic() - start_time) >= time_budget_seconds:
            break

    tuned_cfg = TrainingConfig(**asdict(cfg))
    tuned_cfg.fairness_lambda = float(best_params["fairness_lambda"])
    tuned_cfg.cost_weight = float(best_params["cost_weight"])
    tuned_cfg.learning_rate = float(best_params["learning_rate"])
    tuned_cfg.epsilon_decay = float(best_params["epsilon_decay"])

    tune_info = {
        "enabled": True,
        "cached": False,
        "tested_configs": tested,
        "search_space": {
            "fairness_lambda": fairness_candidates,
            "cost_weight": cost_candidates,
            "learning_rate": lr_candidates,
            "epsilon_decay": eps_decay_candidates,
            "max_trials": max_trials,
        },
        "best_params": best_params,
        "best_objective": best_score,
        "best_metrics": best_metrics,
        "tune_budget": {
            "episodes": tune_cfg.episodes,
            "cohort_size": tune_cfg.cohort_size,
            "horizon": tune_cfg.horizon,
            "seeds": seed_candidates,
            "time_budget_seconds": time_budget_seconds,
        },
    }

    with _AUTOTUNE_CACHE_LOCK:
        _AUTOTUNE_CACHE[cache_key] = (TrainingConfig(**asdict(tuned_cfg)), tune_info)

    return tuned_cfg, tune_info


def prepare_config(payload: Dict) -> tuple[TrainingConfig, Dict]:
    cfg = parse_config(payload)
    auto_tune = payload.get("auto_tune", True)
    if not auto_tune:
        return cfg, {"enabled": False}
    tuned_cfg, tune_info = _quick_autotune_config(cfg)
    return tuned_cfg, tune_info


def _algo_seed_offset(algo_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(algo_id))


def _run_single_algorithm(algo: Dict, cfg: TrainingConfig) -> tuple[dict, dict]:
    env = HealthcareEnvironment()
    seed_offset = _algo_seed_offset(algo["id"])
    rng_train = np.random.default_rng(cfg.seed + seed_offset)
    rng_eval = np.random.default_rng(cfg.seed + 999 + seed_offset)
    agent = QLearningAgent(env.action_count, cfg)

    training_history = train_algorithm(agent, env, algo["id"], cfg, rng_train)
    evaluation = evaluate_algorithm(agent, env, algo["id"], cfg, rng_eval)

    row = {
        "algorithm_id": algo["id"],
        "algorithm": algo["name"],
        "reference": algo["paper_link"],
        **evaluation,
    }
    series = {
        "algorithm_id": algo["id"],
        "algorithm": algo["name"],
        "episode_rewards": training_history["episode_rewards"],
        "episode_gini": training_history["episode_gini"],
        "episode_adherence": training_history["episode_adherence"],
        "episode_clinical": training_history["episode_clinical"],
    }
    return row, series


def _run_single_algorithm_with_payload(algo: Dict, cfg: TrainingConfig, payload: Dict) -> tuple[dict, dict]:
    env = _build_environment(payload)
    seed_offset = _algo_seed_offset(algo["id"])
    rng_train = np.random.default_rng(cfg.seed + seed_offset)
    rng_eval = np.random.default_rng(cfg.seed + 999 + seed_offset)
    agent = QLearningAgent(env.action_count, cfg)

    training_history = train_algorithm(agent, env, algo["id"], cfg, rng_train)
    evaluation = evaluate_algorithm(agent, env, algo["id"], cfg, rng_eval)

    row = {
        "algorithm_id": algo["id"],
        "algorithm": algo["name"],
        "reference": algo["paper_link"],
        **evaluation,
    }
    series = {
        "algorithm_id": algo["id"],
        "algorithm": algo["name"],
        "episode_rewards": training_history["episode_rewards"],
        "episode_gini": training_history["episode_gini"],
        "episode_adherence": training_history["episode_adherence"],
        "episode_clinical": training_history["episode_clinical"],
    }
    return row, series


def _attach_uncertainty(row: Dict, algo: Dict, cfg: TrainingConfig, payload: Dict) -> Dict:
    seeds = payload.get("uncertainty_seeds", 3)
    try:
        seeds = int(seeds)
    except (TypeError, ValueError):
        seeds = 3
    seeds = max(2, min(6, seeds))

    metrics = {"reward": [], "equity_index": [], "clinical_improvement": []}
    for offset in range(seeds):
        test_cfg = TrainingConfig(**asdict(cfg))
        test_cfg.episodes = max(24, min(70, cfg.episodes // 3 if cfg.episodes > 1 else 24))
        test_cfg.horizon = max(8, min(14, cfg.horizon // 2 if cfg.horizon > 1 else 8))
        test_cfg.cohort_size = max(24, min(60, cfg.cohort_size // 2 if cfg.cohort_size > 1 else 24))
        test_cfg.seed = cfg.seed + 17 * offset
        eval_row, _ = _run_single_algorithm_with_payload(algo, test_cfg, payload)
        metrics["reward"].append(eval_row["reward"])
        metrics["equity_index"].append(eval_row["equity_index"])
        metrics["clinical_improvement"].append(eval_row["clinical_improvement"])

    uncertainty = {}
    for key, values in metrics.items():
        arr = np.array(values, dtype=float)
        std = float(np.std(arr))
        ci95 = float(1.96 * std / max(np.sqrt(len(arr)), 1.0))
        uncertainty[key] = {
            "mean": float(np.mean(arr)),
            "std": std,
            "ci95": ci95,
            "lower": float(np.mean(arr) - ci95),
            "upper": float(np.mean(arr) + ci95),
        }

    return {**row, "uncertainty": uncertainty, "uncertainty_seed_count": seeds}


def run_comparison(payload: Dict) -> Dict:
    cfg, autotune = prepare_config(payload)
    cfg = _apply_ablation_and_runtime_overrides(cfg, payload)
    selected = payload.get("selected_algorithms")
    selected_ids = set(selected) if isinstance(selected, list) and selected else None

    output = {
        "config": asdict(cfg),
        "autotune": autotune,
        "results": [],
        "series": [],
    }

    active_algorithms = [algo for algo in ALGORITHMS if not selected_ids or algo["id"] in selected_ids]
    max_workers = max(1, min(8, len(active_algorithms)))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_single_algorithm_with_payload, algo, cfg, payload): algo
            for algo in active_algorithms
        }
        for future in as_completed(futures):
            row, series = future.result()
            row = _attach_uncertainty(row, futures[future], cfg, payload)
            output["results"].append(row)
            output["series"].append(series)

    output["results"].sort(
        key=lambda item: (
            item.get("ranking_score", item.get("overall_score", item["reward"])),
            item.get("overall_score", item["reward"]),
            item["reward"],
        ),
        reverse=True,
    )
    return output


def run_comparison_stream(payload: Dict) -> Generator[dict, None, None]:
    auto_tune = payload.get("auto_tune", True)
    if auto_tune:
        yield {"type": "autotune_start", "message": "Auto-tuning fairness model parameters"}

    cfg_base = parse_config(payload)
    cfg = cfg_base
    autotune = {"enabled": False}

    if auto_tune:
        tune_queue: queue.SimpleQueue = queue.SimpleQueue()
        tune_result_box: dict = {}

        def tune_hook(progress_data: Dict) -> None:
            tune_queue.put({"type": "autotune_progress", **progress_data})

        def tune_worker() -> None:
            try:
                tuned_cfg, tune_info = _quick_autotune_config(cfg_base, progress_hook=tune_hook)
                tune_result_box["cfg"] = tuned_cfg
                tune_result_box["autotune"] = tune_info
            except Exception as exc:
                tune_result_box["cfg"] = cfg_base
                tune_result_box["autotune"] = {
                    "enabled": False,
                    "error": str(exc),
                }
                tune_queue.put({"type": "autotune_error", "message": str(exc)})
            finally:
                tune_queue.put({"type": "autotune_done"})

        tune_thread = threading.Thread(target=tune_worker, daemon=True)
        tune_thread.start()

        last_keepalive = time.monotonic()
        while True:
            try:
                tune_event = tune_queue.get(timeout=0.20)
                if tune_event["type"] == "autotune_progress":
                    yield tune_event
                elif tune_event["type"] == "autotune_error":
                    yield tune_event
                elif tune_event["type"] == "autotune_done":
                    break
            except queue.Empty:
                now = time.monotonic()
                if now - last_keepalive >= 1.0:
                    last_keepalive = now
                    yield {"type": "autotune_keepalive", "message": "Auto-tuning in progress"}

        tune_thread.join()
        cfg = tune_result_box.get("cfg", cfg_base)
        cfg = _apply_ablation_and_runtime_overrides(cfg, payload)
        autotune = tune_result_box.get("autotune", {"enabled": False})
        yield {"type": "autotune_result", "autotune": autotune, "config": asdict(cfg)}
    else:
        cfg, autotune = prepare_config(payload)
        cfg = _apply_ablation_and_runtime_overrides(cfg, payload)

    selected = payload.get("selected_algorithms")
    selected_ids = set(selected) if isinstance(selected, list) and selected else None

    yield {
        "type": "start",
        "config": asdict(cfg),
        "autotune": autotune,
    }

    all_results = []
    all_series = []
    stream_errors = []
    active_algorithms = [algo for algo in ALGORITHMS if not selected_ids or algo["id"] in selected_ids]
    total_algorithms = len(active_algorithms)

    if total_algorithms == 0:
        yield {
            "type": "complete",
            "config": asdict(cfg),
            "autotune": autotune,
            "results": [],
            "series": [],
            "errors": [],
        }
        return

    progress_queue: queue.SimpleQueue = queue.SimpleQueue()
    done_counter = {"count": 0}
    done_lock = threading.Lock()

    def worker_runner(algo_index: int, algo: Dict) -> None:
        try:
            env = _build_environment(payload)
            seed_offset = _algo_seed_offset(algo["id"])
            rng_train = np.random.default_rng(cfg.seed + seed_offset)
            rng_eval = np.random.default_rng(cfg.seed + 999 + seed_offset)
            agent = QLearningAgent(env.action_count, cfg)

            series_bundle = {
                "algorithm_id": algo["id"],
                "algorithm": algo["name"],
                "episode_rewards": [],
                "episode_gini": [],
                "episode_adherence": [],
                "episode_clinical": [],
            }

            def on_progress(progress: dict) -> None:
                series_bundle["episode_rewards"].append(progress["reward"])
                series_bundle["episode_gini"].append(progress["gini"])
                series_bundle["episode_adherence"].append(progress["adherence"])
                series_bundle["episode_clinical"].append(progress["clinical"])
                progress_queue.put(
                    {
                        "type": "progress",
                        "algorithm_id": algo["id"],
                        "algorithm": algo["name"],
                        "algorithm_index": algo_index,
                        "algorithm_total": total_algorithms,
                        "episode": progress["episode"],
                        "episodes": progress["episodes"],
                        "reward": progress["reward"],
                        "gini": progress["gini"],
                        "adherence": progress["adherence"],
                        "clinical": progress["clinical"],
                        "epsilon": progress.get("epsilon"),
                        "dual_mu": progress.get("dual_mu"),
                    }
                )

            train_algorithm(agent, env, algo["id"], cfg, rng_train, progress_hook=on_progress)
            evaluation = evaluate_algorithm(agent, env, algo["id"], cfg, rng_eval)

            row = {
                "algorithm_id": algo["id"],
                "algorithm": algo["name"],
                "reference": algo["paper_link"],
                **evaluation,
            }
            row = _attach_uncertainty(row, algo, cfg, payload)
            progress_queue.put(
                {
                    "type": "algorithm_done",
                    "algorithm_id": algo["id"],
                    "result": row,
                    "series": series_bundle,
                }
            )
        except Exception as exc:
            progress_queue.put(
                {
                    "type": "algorithm_error",
                    "algorithm_id": algo["id"],
                    "algorithm": algo["name"],
                    "message": str(exc),
                }
            )
        finally:
            with done_lock:
                done_counter["count"] += 1

    workers: list[threading.Thread] = []
    for algo_index, algo in enumerate(active_algorithms, start=1):
        thread = threading.Thread(target=worker_runner, args=(algo_index, algo), daemon=True)
        workers.append(thread)
        thread.start()

    last_keepalive = time.monotonic()
    while True:
        try:
            event = progress_queue.get(timeout=0.05)
            if event["type"] == "algorithm_done":
                all_results.append(event["result"])
                all_series.append(event["series"])
            elif event["type"] == "algorithm_error":
                stream_errors.append(
                    {
                        "algorithm_id": event["algorithm_id"],
                        "algorithm": event["algorithm"],
                        "message": event["message"],
                    }
                )
            yield event
        except queue.Empty:
            with done_lock:
                if done_counter["count"] >= total_algorithms:
                    break
            now = time.monotonic()
            if now - last_keepalive >= 1.0:
                last_keepalive = now
                yield {
                    "type": "stream_keepalive",
                    "active_algorithms": total_algorithms - done_counter["count"],
                }
            time.sleep(0.01)

    for thread in workers:
        thread.join()

    all_results.sort(
        key=lambda item: (
            item.get("ranking_score", item.get("overall_score", item["reward"])),
            item.get("overall_score", item["reward"]),
            item["reward"],
        ),
        reverse=True,
    )
    yield {
        "type": "complete",
        "config": asdict(cfg),
        "autotune": autotune,
        "results": all_results,
        "series": all_series,
        "errors": stream_errors,
    }


def run_ablation_studio(payload: Dict) -> Dict:
    cfg, autotune = prepare_config(payload)
    cfg = _apply_ablation_and_runtime_overrides(cfg, payload)
    fairness_algo = next(item for item in ALGORITHMS if item["id"] == "fairness_constrained")

    components = [
        ("fairness_term", "enable_fairness_term"),
        ("shielding", "enable_action_shielding"),
        ("replay", "enable_prioritized_replay"),
        ("double_q", "enable_double_q"),
        ("episode_shaping", "enable_episode_shaping"),
    ]

    baseline_cfg = TrainingConfig(**asdict(cfg))
    for _, attr in components:
        setattr(baseline_cfg, attr, True)

    baseline_row, _ = _run_single_algorithm_with_payload(fairness_algo, baseline_cfg, payload)
    experiments = []
    for component_name, attr in components:
        ablated_cfg = TrainingConfig(**asdict(baseline_cfg))
        setattr(ablated_cfg, attr, False)
        row, _ = _run_single_algorithm_with_payload(fairness_algo, ablated_cfg, payload)
        experiments.append(
            {
                "component": component_name,
                "with_component": baseline_row,
                "without_component": row,
                "impact": {
                    "reward": baseline_row["reward"] - row["reward"],
                    "equity_index": baseline_row["equity_index"] - row["equity_index"],
                    "clinical_improvement": baseline_row["clinical_improvement"] - row["clinical_improvement"],
                    "overall_score": baseline_row.get("overall_score", baseline_row["reward"])
                    - row.get("overall_score", row["reward"]),
                },
            }
        )

    return {
        "config": asdict(cfg),
        "autotune": autotune,
        "baseline": baseline_row,
        "experiments": experiments,
    }


def run_scenario_lab(payload: Dict) -> Dict:
    left = payload.get("scenario_left", "baseline")
    right = payload.get("scenario_right", "low_income_skew")

    left_payload = {**payload, "scenario": left, "auto_tune": False}
    right_payload = {**payload, "scenario": right, "auto_tune": False}

    left_result = run_comparison(left_payload)
    right_result = run_comparison(right_payload)

    proposed_left = next((item for item in left_result["results"] if item["algorithm_id"] == "fairness_constrained"), None)
    proposed_right = next((item for item in right_result["results"] if item["algorithm_id"] == "fairness_constrained"), None)

    delta = {}
    if proposed_left and proposed_right:
        for key in ["reward", "equity_index", "clinical_improvement", "avg_treatment_cost", "avg_rfb"]:
            delta[key] = proposed_right[key] - proposed_left[key]

    return {
        "left": {"scenario": left, **left_result},
        "right": {"scenario": right, **right_result},
        "proposed_delta": delta,
    }


def explain_counterfactual(payload: Dict) -> Dict:
    env = _build_environment(payload)
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}

    severity = float(state.get("severity", 0.62))
    cumulative_cost = float(state.get("cumulative_cost", 200.0))
    last_adherence = float(state.get("last_adherence", 1.0))
    economic_capacity = float(profile.get("economic_capacity", 1800.0))
    insurance_support = float(profile.get("insurance_support", 0.28))
    vulnerability = float(profile.get("vulnerability", 0.52))

    action_insights = []
    for action_id, action in enumerate(env.actions):
        effective_cost = action["cost"] * (1.0 - insurance_support) * (1.0 + env.drift_cost_inflation)
        projected_rfb = (cumulative_cost + effective_cost) / max(economic_capacity, 1.0)
        adherence_prob = (
            0.92
            - 0.58 * projected_rfb
            - 0.11 * severity
            - 0.12 * vulnerability
            + 0.08 * insurance_support
            - env.drift_adherence_drop
        )
        adherence_prob = float(min(0.98, max(0.04, adherence_prob)))
        efficacy = action["efficacy"] * (1.0 - env.drift_efficacy_drop)
        expected_clinical_delta = adherence_prob * efficacy * (1.0 - 0.25 * vulnerability)
        proxy_reward = (
            8.0 * expected_clinical_delta
            - 0.55 * max(0.0, severity - expected_clinical_delta)
            - 2.2 * max(0.0, projected_rfb - 0.38)
        )

        action_insights.append(
            {
                "action_id": action_id,
                "action": action["name"],
                "projected_rfb": projected_rfb,
                "adherence_prob": adherence_prob,
                "expected_clinical_delta": expected_clinical_delta,
                "proxy_reward": proxy_reward,
            }
        )

    sorted_actions = sorted(action_insights, key=lambda item: item["proxy_reward"], reverse=True)
    best = sorted_actions[0]
    second = sorted_actions[1] if len(sorted_actions) > 1 else sorted_actions[0]

    baseline_choices = {
        "clinical": max(action_insights, key=lambda item: item["expected_clinical_delta"])["action"],
        "cost_sensitive": min(action_insights, key=lambda item: item["projected_rfb"])["action"],
        "multi_objective": sorted_actions[0]["action"],
        "primal_dual": sorted(action_insights, key=lambda item: (item["projected_rfb"], -item["expected_clinical_delta"]))[0]["action"],
    }

    return {
        "recommended_action": best["action"],
        "recommended_action_id": best["action_id"],
        "why": f"Selected {best['action']} because it maximizes expected reward proxy with controlled projected burden.",
        "predicted_outcome_delta_vs_next_best": {
            "reward": best["proxy_reward"] - second["proxy_reward"],
            "clinical": best["expected_clinical_delta"] - second["expected_clinical_delta"],
            "rfb": best["projected_rfb"] - second["projected_rfb"],
        },
        "baseline_actions": baseline_choices,
        "action_table": sorted_actions,
    }
