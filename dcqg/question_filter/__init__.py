"""Question filter package for DCQG.

Exports the main filter functions and pipeline for generated question quality control.
"""
from dcqg.question_filter.grammar import (
    grammar_filter,
    enhanced_grammar_filter,
    check_weak_trigger,
)
from dcqg.question_filter.consistency import (
    answer_event_consistency_judge,
    extract_gold_answer_phrase,
)
from dcqg.question_filter.path_coverage import (
    check_path_coverage_lexical,
    path_coverage_judge,
)
from dcqg.question_filter.shortcut import (
    hard_degraded_check,
    check_banned_phrases,
)
from dcqg.question_filter.pipeline import (
    quality_filter_pipeline,
    apply_final_filter,
)

__all__ = [
    "grammar_filter",
    "enhanced_grammar_filter",
    "check_weak_trigger",
    "answer_event_consistency_judge",
    "extract_gold_answer_phrase",
    "check_path_coverage_lexical",
    "path_coverage_judge",
    "hard_degraded_check",
    "check_banned_phrases",
    "quality_filter_pipeline",
    "apply_final_filter",
]
