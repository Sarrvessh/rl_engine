from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PatientProfile:
    economic_capacity: float
    insurance_support: float
    vulnerability: float
    age: float
    risk_tier: int


@dataclass
class PatientState:
    severity: float
    cumulative_cost: float
    last_adherence: float


class HealthcareEnvironment:
    def __init__(
        self,
        scenario: str = "baseline",
        drift_cost_inflation: float = 0.0,
        drift_adherence_drop: float = 0.0,
        drift_efficacy_drop: float = 0.0,
    ) -> None:
        self.actions: List[Dict[str, float]] = [
            {"name": "Conservative", "cost": 120.0, "efficacy": 0.16, "risk": 0.08},
            {"name": "Balanced", "cost": 240.0, "efficacy": 0.27, "risk": 0.13},
            {"name": "Aggressive", "cost": 380.0, "efficacy": 0.36, "risk": 0.20},
        ]
        self.scenario = scenario
        self.drift_cost_inflation = max(-0.4, min(1.5, drift_cost_inflation))
        self.drift_adherence_drop = max(0.0, min(0.6, drift_adherence_drop))
        self.drift_efficacy_drop = max(0.0, min(0.7, drift_efficacy_drop))

    @property
    def action_count(self) -> int:
        return len(self.actions)

    def sample_profile(self, rng) -> PatientProfile:
        segment_probs = [0.42, 0.38, 0.20]
        vulnerability_boost = 0.0

        if self.scenario == "low_income_skew":
            segment_probs = [0.68, 0.24, 0.08]
            vulnerability_boost = 0.08
        elif self.scenario == "high_chronic_load":
            segment_probs = [0.46, 0.36, 0.18]
            vulnerability_boost = 0.16
        elif self.scenario == "policy_shock":
            segment_probs = [0.50, 0.36, 0.14]
            vulnerability_boost = 0.10

        segment = rng.choice([0, 1, 2], p=segment_probs)

        age = float(rng.uniform(21, 86))
        if self.scenario == "high_chronic_load":
            age = float(rng.uniform(45, 89))

        if segment == 0:
            return PatientProfile(
                economic_capacity=float(rng.uniform(900, 1500)),
                insurance_support=float(rng.uniform(0.08, 0.25)),
                vulnerability=float(min(1.0, rng.uniform(0.60, 1.00) + vulnerability_boost)),
                age=age,
                risk_tier=2,
            )
        if segment == 1:
            return PatientProfile(
                economic_capacity=float(rng.uniform(1500, 2800)),
                insurance_support=float(rng.uniform(0.22, 0.45)),
                vulnerability=float(min(1.0, rng.uniform(0.35, 0.70) + vulnerability_boost)),
                age=age,
                risk_tier=1,
            )
        return PatientProfile(
            economic_capacity=float(rng.uniform(2800, 5200)),
            insurance_support=float(rng.uniform(0.40, 0.65)),
            vulnerability=float(min(1.0, rng.uniform(0.10, 0.40) + vulnerability_boost)),
            age=age,
            risk_tier=0,
        )

    def initial_state(self, rng) -> PatientState:
        return PatientState(
            severity=float(rng.uniform(0.45, 0.95)),
            cumulative_cost=0.0,
            last_adherence=1.0,
        )

    def discretize_state(self, profile: PatientProfile, state: PatientState) -> tuple[int, int, int]:
        severity_bin = min(4, int(state.severity * 5.0))
        burden = state.cumulative_cost / max(profile.economic_capacity, 1.0)
        burden_bin = min(4, int(min(burden, 0.999) * 5.0))
        adherence_bin = 1 if state.last_adherence >= 0.5 else 0
        return severity_bin, burden_bin, adherence_bin

    def step(self, profile: PatientProfile, state: PatientState, action_id: int, rng) -> dict:
        action = self.actions[action_id]
        effective_cost = action["cost"] * (1.0 - profile.insurance_support) * (1.0 + self.drift_cost_inflation)
        projected_cost = state.cumulative_cost + effective_cost
        projected_rfb = projected_cost / max(profile.economic_capacity, 1.0)

        adherence_prob = (
            0.92
            - 0.58 * projected_rfb
            - 0.11 * state.severity
            - 0.12 * profile.vulnerability
            + 0.08 * profile.insurance_support
        )
        adherence_prob -= self.drift_adherence_drop
        adherence_prob = float(min(0.98, max(0.04, adherence_prob)))
        adhered = 1.0 if rng.random() < adherence_prob else 0.0

        noise = float(rng.normal(0.0, 0.03))
        if adhered > 0.5:
            efficacy = action["efficacy"] * (1.0 - self.drift_efficacy_drop)
            improvement = efficacy * (1.0 - 0.25 * profile.vulnerability) + noise
            next_severity = max(0.0, state.severity - improvement)
            cost_paid = effective_cost
        else:
            deterioration = 0.10 + 0.35 * action["risk"] + 0.20 * profile.vulnerability + abs(noise)
            next_severity = min(1.0, state.severity + deterioration)
            cost_paid = 0.22 * effective_cost

        next_state = PatientState(
            severity=float(next_severity),
            cumulative_cost=float(state.cumulative_cost + cost_paid),
            last_adherence=float(adhered),
        )
        prev_rfb = state.cumulative_cost / max(profile.economic_capacity, 1.0)
        next_rfb = next_state.cumulative_cost / max(profile.economic_capacity, 1.0)
        return {
            "next_state": next_state,
            "adhered": adhered,
            "adherence_prob": adherence_prob,
            "risk": action["risk"],
            "effective_cost": effective_cost,
            "prev_rfb": prev_rfb,
            "rfb": next_rfb,
            "clinical_improvement": state.severity - next_state.severity,
            "profile": {
                "economic_capacity": profile.economic_capacity,
                "age": profile.age,
                "risk_tier": profile.risk_tier,
                "vulnerability": profile.vulnerability,
            },
        }
