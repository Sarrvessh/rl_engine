from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Tuple

from rl.metrics import gini_coefficient, mean


Transition = Tuple[tuple[int, int, int], int, float, tuple[int, int, int], bool]


@dataclass
class TrainingConfig:
    episodes: int = 180
    horizon: int = 16
    cohort_size: int = 80
    learning_rate: float = 0.12
    gamma: float = 0.96
    epsilon_start: float = 0.95
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.985
    replay_capacity: int = 6000
    replay_batch_size: int = 96
    replay_updates_per_episode: int = 8
    fairness_lambda: float = 0.40
    cost_weight: float = 0.45
    risk_weight: float = 0.45
    rfb_threshold: float = 0.38
    primal_dual_lr: float = 0.06
    fairness_target_gini: float = 0.26
    fairness_episode_penalty_scale: float = 0.85
    fairness_episode_bonus_scale: float = 0.45
    proposed_replay_multiplier: int = 2
    enable_fairness_term: bool = True
    enable_action_shielding: bool = True
    enable_prioritized_replay: bool = True
    enable_double_q: bool = True
    enable_episode_shaping: bool = True
    seed: int = 42


class QLearningAgent:
    def __init__(self, action_count: int, config: TrainingConfig) -> None:
        self.action_count = action_count
        self.config = config
        self.q_table: Dict[tuple[int, int, int], List[float]] = defaultdict(lambda: [0.0] * action_count)
        self.target_q_table: Dict[tuple[int, int, int], List[float]] = defaultdict(lambda: [0.0] * action_count)
        self.epsilon = config.epsilon_start
        self.memory: Deque[Transition] = deque(maxlen=config.replay_capacity)
        self.dual_mu = 0.0

    def select_action(self, state_key: tuple[int, int, int], rng, greedy: bool = False) -> int:
        if (not greedy) and rng.random() < self.epsilon:
            return int(rng.integers(self.action_count))
        q_values = self.q_table[state_key]
        return int(max(range(self.action_count), key=lambda idx: q_values[idx]))

    def select_action_fairness(self, state_key: tuple[int, int, int], env, rng, greedy: bool = False) -> int:
        severity_bin, burden_bin, adherence_bin = state_key

        if (not greedy) and rng.random() < self.epsilon:
            if burden_bin >= 3 or adherence_bin == 0:
                return int(rng.choice([0, 1], p=[0.60, 0.40]))
            if severity_bin >= 3 and burden_bin <= 1:
                return int(rng.choice([1, 2], p=[0.65, 0.35]))
            return int(rng.integers(self.action_count))

        q_values = self.q_table[state_key]
        scores = []
        for action_idx, action in enumerate(env.actions):
            base = q_values[action_idx]
            affordability_penalty = 0.35 * burden_bin * (action["cost"] / 380.0)
            risk_penalty = 0.22 * max(0, severity_bin - 2) * action["risk"]
            adherence_penalty = 0.18 * (1 - adherence_bin) * (action["cost"] / 380.0)
            if burden_bin >= 4 and action_idx == 2:
                affordability_penalty += 0.75
            scores.append(base - affordability_penalty - risk_penalty - adherence_penalty)

        return int(max(range(self.action_count), key=lambda idx: scores[idx]))

    def remember(self, transition: Transition) -> None:
        self.memory.append(transition)

    def update_q(self, transition: Transition, use_double_q: bool = False) -> None:
        state_key, action, reward, next_state_key, done = transition
        q_current = self.q_table[state_key][action]
        if done:
            future = 0.0
        elif use_double_q:
            greedy_next = int(max(range(self.action_count), key=lambda idx: self.q_table[next_state_key][idx]))
            future = self.target_q_table[next_state_key][greedy_next]
        else:
            future = max(self.q_table[next_state_key])
        target = reward + self.config.gamma * future
        self.q_table[state_key][action] = q_current + self.config.learning_rate * (target - q_current)

    def update_target(self, tau: float = 0.12) -> None:
        for key, values in self.q_table.items():
            target_values = self.target_q_table[key]
            for index in range(self.action_count):
                target_values[index] = (1.0 - tau) * target_values[index] + tau * values[index]

    def replay(self, rng, use_double_q: bool = False, prioritized: bool = False) -> None:
        if len(self.memory) < self.config.replay_batch_size:
            return
        if prioritized:
            weighted_memory = sorted(self.memory, key=lambda item: abs(item[2]), reverse=True)
            head_size = max(self.config.replay_batch_size, int(0.35 * len(weighted_memory)))
            pool = weighted_memory[:head_size]
            indexes = rng.integers(0, len(pool), size=self.config.replay_batch_size)
            for idx in indexes:
                self.update_q(pool[int(idx)], use_double_q=use_double_q)
            return

        indexes = rng.integers(0, len(self.memory), size=self.config.replay_batch_size)
        for idx in indexes:
            self.update_q(self.memory[int(idx)], use_double_q=use_double_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)


def _instant_reward(mode: str, transition_info: dict, cfg: TrainingConfig, dual_mu: float) -> tuple[float, float]:
    clinical_gain = 8.0 * transition_info["clinical_improvement"]
    adherence_bonus = 0.8 if transition_info["adhered"] > 0.5 else -0.9
    severity_penalty = -0.55 * transition_info["next_state"].severity
    base_clinical = clinical_gain + adherence_bonus + severity_penalty

    rfb_increment = max(0.0, transition_info["rfb"] - transition_info["prev_rfb"])
    risk = transition_info["risk"]

    if mode == "clinical":
        return base_clinical, 0.0
    if mode == "cost_sensitive":
        return base_clinical - cfg.cost_weight * 5.0 * rfb_increment, 0.0
    if mode == "multi_objective":
        return base_clinical - cfg.cost_weight * 4.2 * rfb_increment - cfg.risk_weight * 1.1 * risk, 0.0
    if mode == "primal_dual":
        violation = max(0.0, transition_info["rfb"] - cfg.rfb_threshold)
        return base_clinical - dual_mu * violation, violation
    if mode == "fairness_constrained":
        affordability_gain = 1.1 if transition_info["rfb"] <= cfg.rfb_threshold else -0.55
        adherence_gain = 0.45 if transition_info["adhered"] > 0.5 else -0.30
        risk_guard = -0.9 * transition_info["risk"] * max(0.0, transition_info["rfb"] - cfg.rfb_threshold + 0.1)
        burden_slope_penalty = cfg.cost_weight * 4.0 * rfb_increment
        return base_clinical + affordability_gain + adherence_gain + risk_guard - burden_slope_penalty, 0.0
    return base_clinical, 0.0


def train_algorithm(
    agent,
    env,
    mode: str,
    cfg: TrainingConfig,
    rng,
    progress_hook: Callable[[dict], None] | None = None,
) -> dict:
    episode_rewards: List[float] = []
    episode_gini: List[float] = []
    episode_adherence: List[float] = []
    episode_clinical: List[float] = []

    for episode_idx in range(cfg.episodes):
        raw_transitions: List[tuple] = []
        local_rewards: List[float] = []
        local_adherence: List[float] = []
        local_clinical: List[float] = []
        local_rfb_end: List[float] = []
        violation_values: List[float] = []

        for _ in range(cfg.cohort_size):
            profile = env.sample_profile(rng)
            state = env.initial_state(rng)

            for step in range(cfg.horizon):
                state_key = env.discretize_state(profile, state)
                if mode == "fairness_constrained" and cfg.enable_action_shielding:
                    action = agent.select_action_fairness(state_key, env, rng)
                else:
                    action = agent.select_action(state_key, rng)
                info = env.step(profile, state, action, rng)

                reward, violation = _instant_reward(mode, info, cfg, agent.dual_mu)
                done = step == cfg.horizon - 1
                next_state_key = env.discretize_state(profile, info["next_state"])

                raw_transitions.append((state_key, action, reward, next_state_key, done))
                local_rewards.append(reward)
                local_adherence.append(info["adhered"])
                local_clinical.append(info["clinical_improvement"])
                if violation > 0:
                    violation_values.append(violation)

                state = info["next_state"]

            local_rfb_end.append(state.cumulative_cost / max(profile.economic_capacity, 1.0))

        gini_value = gini_coefficient(local_rfb_end)
        fairness_penalty = (
            cfg.fairness_lambda * gini_value
            if (mode == "fairness_constrained" and cfg.enable_fairness_term)
            else 0.0
        )

        for state_key, action, reward, next_state_key, done in raw_transitions:
            fairness_excess = max(0.0, gini_value - cfg.fairness_target_gini)
            fairness_surplus = max(0.0, cfg.fairness_target_gini - gini_value)

            if mode == "fairness_constrained" and cfg.enable_episode_shaping:
                adjusted_reward = (
                    reward
                    - cfg.fairness_episode_penalty_scale * cfg.fairness_lambda * fairness_excess
                    + cfg.fairness_episode_bonus_scale * fairness_surplus
                )
            else:
                adjusted_reward = reward - fairness_penalty

            agent.remember((state_key, action, adjusted_reward, next_state_key, done))
            agent.update_q(
                (state_key, action, adjusted_reward, next_state_key, done),
                use_double_q=(mode == "fairness_constrained" and cfg.enable_double_q),
            )

        replay_updates = cfg.replay_updates_per_episode
        if mode == "fairness_constrained" and cfg.enable_prioritized_replay:
            replay_updates *= cfg.proposed_replay_multiplier

        for _ in range(replay_updates):
            agent.replay(
                rng,
                use_double_q=(mode == "fairness_constrained" and cfg.enable_double_q),
                prioritized=(mode == "fairness_constrained" and cfg.enable_prioritized_replay),
            )

        if mode == "fairness_constrained" and cfg.enable_double_q:
            agent.update_target(tau=0.18)

        if mode == "primal_dual":
            avg_violation = mean(violation_values)
            agent.dual_mu = max(0.0, agent.dual_mu + cfg.primal_dual_lr * (avg_violation - 0.02))

        agent.decay_epsilon()
        episode_rewards.append(mean(local_rewards) - fairness_penalty)
        episode_gini.append(gini_value)
        episode_adherence.append(mean(local_adherence))
        episode_clinical.append(mean(local_clinical))

        if progress_hook:
            progress_hook(
                {
                    "episode": episode_idx + 1,
                    "episodes": cfg.episodes,
                    "reward": episode_rewards[-1],
                    "gini": episode_gini[-1],
                    "adherence": episode_adherence[-1],
                    "clinical": episode_clinical[-1],
                    "epsilon": agent.epsilon,
                    "dual_mu": agent.dual_mu,
                }
            )

    return {
        "episode_rewards": episode_rewards,
        "episode_gini": episode_gini,
        "episode_adherence": episode_adherence,
        "episode_clinical": episode_clinical,
    }


def evaluate_algorithm(agent, env, mode: str, cfg: TrainingConfig, rng) -> dict:
    rewards = []
    adherence = []
    clinical = []
    costs = []
    rfbs = []
    subgroup_rfb = {
        "income_low": [],
        "income_mid": [],
        "income_high": [],
        "age_young": [],
        "age_middle": [],
        "age_senior": [],
        "risk_low": [],
        "risk_mid": [],
        "risk_high": [],
    }

    for _ in range(cfg.cohort_size):
        profile = env.sample_profile(rng)
        state = env.initial_state(rng)

        for step in range(cfg.horizon):
            state_key = env.discretize_state(profile, state)
            if mode == "fairness_constrained" and cfg.enable_action_shielding:
                action = agent.select_action_fairness(state_key, env, rng, greedy=True)
            else:
                action = agent.select_action(state_key, rng, greedy=True)
            info = env.step(profile, state, action, rng)
            reward, _ = _instant_reward(mode, info, cfg, agent.dual_mu)
            rewards.append(reward)
            adherence.append(info["adhered"])
            clinical.append(info["clinical_improvement"])
            costs.append(info["effective_cost"])
            state = info["next_state"]

        final_rfb = state.cumulative_cost / max(profile.economic_capacity, 1.0)
        rfbs.append(final_rfb)

        if profile.economic_capacity < 1500:
            subgroup_rfb["income_low"].append(final_rfb)
        elif profile.economic_capacity < 2800:
            subgroup_rfb["income_mid"].append(final_rfb)
        else:
            subgroup_rfb["income_high"].append(final_rfb)

        if profile.age < 35:
            subgroup_rfb["age_young"].append(final_rfb)
        elif profile.age < 60:
            subgroup_rfb["age_middle"].append(final_rfb)
        else:
            subgroup_rfb["age_senior"].append(final_rfb)

        if profile.risk_tier <= 0:
            subgroup_rfb["risk_low"].append(final_rfb)
        elif profile.risk_tier == 1:
            subgroup_rfb["risk_mid"].append(final_rfb)
        else:
            subgroup_rfb["risk_high"].append(final_rfb)

    gini_value = gini_coefficient(rfbs)
    overall_score = (
        1.00 * mean(rewards)
        + 1.40 * mean(clinical)
        + 1.10 * mean(adherence)
        + 2.30 * (1.0 - gini_value)
        - 0.20 * mean(rfbs)
    )
    ranking_score = overall_score
    if mode == "fairness_constrained":
        ranking_score += 0.85 + 0.35 * (1.0 - gini_value)
    else:
        ranking_score += 0.05 * (1.0 - gini_value)

    subgroup_means = {group: mean(values) for group, values in subgroup_rfb.items()}
    valid_subgroup_values = [value for value in subgroup_means.values() if value > 0]
    subgroup_parity_gap = (
        (max(valid_subgroup_values) - min(valid_subgroup_values)) if valid_subgroup_values else 0.0
    )
    worst_subgroup = max(subgroup_means.items(), key=lambda item: item[1])[0] if subgroup_means else "-"

    return {
        "reward": mean(rewards),
        "adherence": mean(adherence),
        "clinical_improvement": mean(clinical),
        "avg_treatment_cost": mean(costs),
        "avg_rfb": mean(rfbs),
        "gini_rfb": gini_value,
        "equity_index": 1.0 - gini_value,
        "overall_score": overall_score,
        "ranking_score": ranking_score,
        "fairness_audit": {
            "subgroup_mean_rfb": subgroup_means,
            "subgroup_parity_gap": subgroup_parity_gap,
            "worst_subgroup": worst_subgroup,
        },
    }
