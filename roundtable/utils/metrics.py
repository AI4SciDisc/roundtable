"""Evaluation metrics from the paper: Wilcoxon signed-rank test, Kendall's Tau."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def wilcoxon_signed_rank(
    rp_scores: list[float],
    baseline_scores: list[float],
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """One-sided Wilcoxon signed-rank test (RP > baseline).

    Tests whether Roundtable Policy significantly improves over a baseline
    player, as described in Appendix A.2 of the paper.

    Args:
        rp_scores: Per-round scores from Roundtable Policy.
        baseline_scores: Per-round scores from the baseline player.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with keys:
            w_plus: The W+ test statistic.
            z_score: Continuity-corrected z-score (if n >= 20).
            p_value: One-sided p-value.
            significant: Whether p < alpha.
            n: Number of non-zero paired differences.
    """
    assert len(rp_scores) == len(baseline_scores), "Score lists must have equal length"

    # Compute relative gains: g_r = (s_rp - s_base) / |s_base| * 100
    gains: list[float] = []
    for rp, base in zip(rp_scores, baseline_scores):
        if abs(base) < 1e-10:
            continue
        gains.append((rp - base) / abs(base) * 100.0)

    # Discard zero differences
    nonzero = [(abs(g), g > 0) for g in gains if abs(g) > 1e-10]
    n = len(nonzero)

    if n == 0:
        return {
            "w_plus": 0.0, "z_score": 0.0, "p_value": 1.0,
            "significant": False, "n": 0,
        }

    # Rank absolute values
    nonzero.sort(key=lambda x: x[0])
    ranks: list[tuple[float, bool]] = []
    i = 0
    while i < n:
        j = i
        while j < n and abs(nonzero[j][0] - nonzero[i][0]) < 1e-10:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, nonzero[k][1]))
        i = j

    # W+ = sum of ranks where g > 0
    w_plus = sum(rank for rank, positive in ranks if positive)

    # Normal approximation with continuity correction (Eq. 6 in paper)
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

    if std_w < 1e-10:
        z_score = 0.0
    else:
        z_score = (w_plus - mean_w - 0.5) / std_w

    # One-sided p-value (upper tail)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2))

    return {
        "w_plus": w_plus,
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < alpha,
        "n": n,
    }


def kendalls_tau(ranking_a: list[int], ranking_b: list[int]) -> float:
    """Compute Kendall's Tau rank correlation coefficient.

    Measures the ordinal association between two rankings, used to assess
    inter-grader consistency (Appendix A.3 in the paper).

    Args:
        ranking_a: Ranking from grader a (list of ranks for each item).
        ranking_b: Ranking from grader b.

    Returns:
        Kendall's Tau in [-1, 1]. 1 = perfect agreement, -1 = perfect
        disagreement, 0 = statistical independence.
    """
    n = len(ranking_a)
    assert n == len(ranking_b), "Rankings must have equal length"
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0

    for (i, j) in combinations(range(n), 2):
        sign_a = _sign(ranking_a[i] - ranking_a[j])
        sign_b = _sign(ranking_b[i] - ranking_b[j])
        if sign_a == sign_b:
            concordant += 1
        elif sign_a != 0 and sign_b != 0:
            discordant += 1

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


def _sign(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def compute_percentage_gain(
    rp_scores: list[float],
    baseline_scores: list[float],
) -> list[float]:
    """Compute per-round percentage gain of RP over a baseline.

    Implements Eq. 7 from the paper:
        g_r = (s_rp - s_base) / |s_base| * 100

    Args:
        rp_scores: Per-round RP scores.
        baseline_scores: Per-round baseline scores.

    Returns:
        List of percentage gains per round.
    """
    gains: list[float] = []
    for rp, base in zip(rp_scores, baseline_scores):
        if abs(base) < 1e-10:
            gains.append(0.0)
        else:
            gains.append((rp - base) / abs(base) * 100.0)
    return gains


def average_grader_agreement(
    grader_rankings: dict[str, list[int]],
) -> np.ndarray:
    """Compute pairwise Kendall's Tau between all graders.

    Args:
        grader_rankings: Dict mapping grader_id to their ranking of players.

    Returns:
        Symmetric matrix of shape (num_graders, num_graders) with
        Kendall's Tau values.
    """
    grader_ids = sorted(grader_rankings.keys())
    n = len(grader_ids)
    tau_matrix = np.ones((n, n), dtype=np.float64)

    for i, j in combinations(range(n), 2):
        tau = kendalls_tau(
            grader_rankings[grader_ids[i]],
            grader_rankings[grader_ids[j]],
        )
        tau_matrix[i, j] = tau
        tau_matrix[j, i] = tau

    return tau_matrix
