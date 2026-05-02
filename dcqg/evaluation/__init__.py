"""Evaluation module: solver, judge, metrics, and reporting."""

from dcqg.evaluation.solver import Solver, Judge, evaluate_all
from dcqg.evaluation.judge import (
    solve,
    target_event_hit,
    llm_judge_v2,
    quality_judge,
    evaluate_item,
    evaluate_file,
)
from dcqg.evaluation.metrics import compute_fair_metrics
from dcqg.evaluation.report import print_comparison_table, print_fair_metrics_table

__all__ = [
    "Solver",
    "Judge",
    "evaluate_all",
    "solve",
    "target_event_hit",
    "llm_judge_v2",
    "quality_judge",
    "evaluate_item",
    "evaluate_file",
    "compute_fair_metrics",
    "print_comparison_table",
    "print_fair_metrics_table",
]
