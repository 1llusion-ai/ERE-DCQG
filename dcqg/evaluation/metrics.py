"""Fair metrics computation (from baselines.py).

compute_fair_metrics: macro-average, end-to-end, and monotonicity metrics.
Only uses stdlib (collections.Counter is not even needed here).
"""
from collections import Counter


def compute_fair_metrics(results, scored, n_total=None):
    """Compute macro-average, end-to-end, and monotonicity metrics.
    Returns dict with all fair metrics."""
    if n_total is None:
        n_total = len(results)

    n_scored = len(scored)
    if n_scored == 0:
        return {}

    # Per-difficulty solver_correct
    by_diff = {}
    for level in ["Easy", "Medium", "Hard"]:
        items_l = [s for s in scored if s["difficulty"] == level]
        if items_l:
            by_diff[level] = {
                "n": len(items_l),
                "solver_correct": sum(s["judge_solver_correct"] for s in items_l) / len(items_l),
                "answerable": sum(s["judge_answerable"] for s in items_l) / len(items_l),
                "composite": sum(s["composite"] for s in items_l) / len(items_l),
            }
        else:
            by_diff[level] = {"n": 0, "solver_correct": 0, "answerable": 0, "composite": 0}

    easy_sol = by_diff["Easy"]["solver_correct"]
    med_sol = by_diff["Medium"]["solver_correct"]
    hard_sol = by_diff["Hard"]["solver_correct"]

    # Macro-average: (Easy mean + Medium mean + Hard mean) / 3
    macro_sol = (easy_sol + med_sol + hard_sol) / 3
    macro_ans = (by_diff["Easy"]["answerable"] + by_diff["Medium"]["answerable"] + by_diff["Hard"]["answerable"]) / 3
    macro_comp = (by_diff["Easy"]["composite"] + by_diff["Medium"]["composite"] + by_diff["Hard"]["composite"]) / 3

    # End-to-end: total score / n_total (fail=0)
    e2e_sol = sum(s["judge_solver_correct"] for s in scored) / n_total
    e2e_ans = sum(s["judge_answerable"] for s in scored) / n_total
    e2e_comp = sum(s["composite"] for s in scored) / n_total

    # Monotonicity metrics
    e_h_gap = easy_sol - hard_sol
    dc_score = max(0, easy_sol - med_sol) + max(0, med_sol - hard_sol)
    violation_penalty = max(0, med_sol - easy_sol) + max(0, hard_sol - med_sol)
    violations = 0
    if easy_sol < med_sol:
        violations += 1
    if med_sol < hard_sol:
        violations += 1

    return {
        "by_diff": by_diff,
        "easy_sol": easy_sol, "med_sol": med_sol, "hard_sol": hard_sol,
        "e_h_gap": e_h_gap,
        "dc_score": dc_score,
        "violation_penalty": violation_penalty,
        "violations": violations,
        "macro_sol": macro_sol, "macro_ans": macro_ans, "macro_comp": macro_comp,
        "e2e_sol": e2e_sol, "e2e_ans": e2e_ans, "e2e_comp": e2e_comp,
        "n_total": n_total, "n_scored": n_scored,
    }
