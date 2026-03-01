from __future__ import annotations

from typing import Iterable, List


def gini_coefficient(values: Iterable[float]) -> float:
    data = [max(0.0, float(v)) for v in values]
    n = len(data)
    if n == 0:
        return 0.0
    total = sum(data)
    if total <= 1e-12:
        return 0.0
    ordered = sorted(data)
    weighted_sum = 0.0
    for index, value in enumerate(ordered, start=1):
        weighted_sum += index * value
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
