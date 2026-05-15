"""Difficulty assessment: definitions, data, classifier, reranker."""

from .definitions import (
    DIFFICULTY_DEFINITIONS,
    difficulty_definition,
    difficulty_definitions_block,
)

__all__ = [
    "DIFFICULTY_DEFINITIONS",
    "difficulty_definition",
    "difficulty_definitions_block",
]


def get_classifier():
    """Deferred import for MultiTaskDifficultyClassifier (requires torch)."""
    from .classifier import MultiTaskDifficultyClassifier
    return MultiTaskDifficultyClassifier


def get_reranker():
    """Deferred import for DifficultyReranker (requires torch)."""
    from .reranker import DifficultyReranker
    return DifficultyReranker
