"""Difficulty assessment: definitions, data, classifier, reranker."""

from .definitions import DIFFICULTY_INSTRUCTIONS, DIFFICULTY_FRAMEWORK, difficulty_instruction

__all__ = ["DIFFICULTY_INSTRUCTIONS", "DIFFICULTY_FRAMEWORK", "difficulty_instruction"]


def get_classifier():
    """Deferred import for MultiTaskDifficultyClassifier (requires torch)."""
    from .classifier import MultiTaskDifficultyClassifier
    return MultiTaskDifficultyClassifier


def get_reranker():
    """Deferred import for DifficultyReranker (requires torch)."""
    from .reranker import DifficultyReranker
    return DifficultyReranker
