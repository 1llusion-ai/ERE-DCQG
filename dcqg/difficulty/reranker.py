"""Classifier-based reranking for difficulty-controlled question generation.

Provides DifficultyReranker which selects from K candidate questions the one
most likely to match the target difficulty according to a trained
MultiTaskDifficultyClassifier, subject to a quality gate.

Typical usage::

    reranker = DifficultyReranker("outputs/models/multitask_deberta_v1/fold_0/model/")
    best, scored = reranker.rerank(candidates, story, "Hard", target_answer)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

DIFF_IDX: dict[str, int] = {"Easy": 0, "Medium": 1, "Hard": 2}
IDX_DIFF: dict[int, str] = {0: "Easy", 1: "Medium", 2: "Hard"}


class DifficultyReranker:
    """Rerank generated questions by target difficulty probability.

    Pipeline:
      1. Generate K candidates (caller provides or use generate_and_rerank)
      2. Quality gate: check answerable, asks_expected_answer, fluent
      3. Among quality-passing: argmax P(target_difficulty)
    """

    def __init__(self, classifier_path: str, device: str = "cuda") -> None:
        """Load the trained MultiTaskDifficultyClassifier.

        Parameters
        ----------
        classifier_path : str
            Path to the saved model directory (containing config.json,
            model.pt, and tokenizer/).
        device : str
            ``"cuda"`` or ``"cpu"``.
        """
        from dcqg.difficulty.classifier import MultiTaskDifficultyClassifier

        self.classifier = MultiTaskDifficultyClassifier.load(classifier_path, device)
        logger.info("DifficultyReranker loaded classifier from %s", classifier_path)

    # ------------------------------------------------------------------
    # Core reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        story_section: str,
        target_difficulty: str,
        target_answer: str,
        *,
        run_quality_judge: bool = True,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Rerank K candidates by difficulty probability.

        Parameters
        ----------
        candidates : list[dict]
            Each dict must have at least ``"generated_question"`` (str).
        story_section : str
            Story context.
        target_difficulty : str
            ``"Easy"`` | ``"Medium"`` | ``"Hard"``.
        target_answer : str
            Target answer string.
        run_quality_judge : bool
            If True (default), apply quality gate via LLM judge. If False,
            treat every non-empty candidate as quality-passing.

        Returns
        -------
        (best_candidate_or_None, all_scored_candidates)
            Each scored candidate is augmented with:
            ``quality_pass``, ``difficulty_probs``, ``target_prob``,
            ``classifier_predicted_difficulty``, ``rerank_selected``.
        """
        if target_difficulty not in DIFF_IDX:
            raise ValueError(
                f"target_difficulty must be Easy/Medium/Hard, got {target_difficulty!r}"
            )
        target_idx = DIFF_IDX[target_difficulty]

        best: dict[str, Any] | None = None
        best_prob = -1.0

        for c in candidates:
            q = c.get("generated_question", "")
            c["rerank_selected"] = False

            if not q or not q.strip():
                c["quality_pass"] = False
                c["difficulty_probs"] = [0.0, 0.0, 0.0]
                c["target_prob"] = 0.0
                c["classifier_predicted_difficulty"] = "N/A"
                continue

            # Quality gate
            if run_quality_judge:
                qj = self._quality_gate(q, story_section, target_answer, target_difficulty)
                c["quality_pass"] = qj.get("quality_pass", False)
                c["quality_judge_detail"] = qj
            else:
                c["quality_pass"] = True

            if not c["quality_pass"]:
                c["difficulty_probs"] = [0.0, 0.0, 0.0]
                c["target_prob"] = 0.0
                c["classifier_predicted_difficulty"] = "N/A"
                continue

            # Get difficulty probabilities from classifier
            probs = self._classify(q, story_section)
            c["difficulty_probs"] = probs
            c["target_prob"] = probs[target_idx]
            c["classifier_predicted_difficulty"] = IDX_DIFF[int(_argmax(probs))]

            if probs[target_idx] > best_prob:
                best_prob = probs[target_idx]
                best = c

        if best is not None:
            best["rerank_selected"] = True

        n_quality = sum(1 for c in candidates if c.get("quality_pass"))
        logger.info(
            "Rerank: %d candidates, %d quality-pass, best target_prob=%.3f",
            len(candidates), n_quality, best_prob if best else 0.0,
        )

        return (best, candidates)

    # ------------------------------------------------------------------
    # Generate-then-rerank convenience
    # ------------------------------------------------------------------

    def generate_and_rerank(
        self,
        generate_fn: Callable,
        story_section: str,
        target_answer: str,
        target_difficulty: str,
        K: int = 5,
        temperature: float = 0.7,
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Generate K candidates using *generate_fn* then rerank.

        Parameters
        ----------
        generate_fn : callable
            Signature: ``(story_section, target_answer, difficulty, **kw)
            -> (result_dict, attempt_count)``.
        story_section : str
        target_answer : str
        target_difficulty : str
        K : int
            Number of candidates to generate (default 5).
        temperature : float
            Passed through to ``generate_fn`` if it accepts it.
        **generate_kwargs
            Extra keyword arguments forwarded to ``generate_fn``.

        Returns
        -------
        (best_candidate_or_None, all_scored_candidates)
        """
        candidates: list[dict[str, Any]] = []
        for k in range(K):
            try:
                result, attempts = generate_fn(
                    story_section, target_answer, target_difficulty, **generate_kwargs
                )
            except Exception as e:
                logger.warning("generate_fn raised %s on candidate %d", e, k)
                result = {
                    "generated_question": "",
                    "method": "unknown",
                    "parse_ok": False,
                    "generation_error": str(e),
                }
                attempts = 1

            result["candidate_index"] = k
            result["candidate_attempts"] = attempts
            candidates.append(result)

        return self.rerank(
            candidates, story_section, target_difficulty, target_answer
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quality_gate(
        self, question: str, story_section: str, target_answer: str, difficulty: str
    ) -> dict[str, Any]:
        """Run quality_judge from fairytale_qg and return result dict."""
        from dcqg.generation.fairytale_qg import quality_judge

        return quality_judge(question, story_section, target_answer, difficulty)

    def _classify(self, question: str, story_section: str) -> list[float]:
        """Return [P(Easy), P(Medium), P(Hard)] from the classifier."""
        from dcqg.difficulty.data import add_sentence_markers

        marked_text, _ = add_sentence_markers(story_section)
        full_text = f"{question} [SEP] {marked_text}"
        return self.classifier.predict_difficulty_probs(full_text)


def _argmax(values: list[float]) -> int:
    """Return index of maximum value."""
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


# ------------------------------------------------------------------
# LLM-based reranking (alternative to classifier)
# ------------------------------------------------------------------

def llm_rerank(
    candidates: list[dict[str, Any]],
    story_section: str,
    target_difficulty: str,
    target_answer: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Rerank candidates using the LLM difficulty_evidence_judge.

    For each quality-passing candidate, calls difficulty_evidence_judge
    and selects the one whose predicted_difficulty matches target_difficulty.
    Among matches, prefers the one needing the most evidence sentences for
    Hard, fewest for Easy.

    Parameters
    ----------
    candidates : list[dict]
        Each with ``"generated_question"`` and ``"quality_pass"`` already set.
    story_section, target_difficulty, target_answer : str

    Returns
    -------
    (best_or_None, all_scored_candidates)
    """
    from dcqg.generation.fairytale_qg import difficulty_evidence_judge

    best: dict[str, Any] | None = None
    best_score = -1.0

    for c in candidates:
        c["llm_rerank_selected"] = False
        q = c.get("generated_question", "")

        if not q or not c.get("quality_pass", False):
            c["llm_predicted_difficulty"] = "N/A"
            c["llm_num_sentences_needed"] = 0
            continue

        dj = difficulty_evidence_judge(q, story_section, target_answer, target_difficulty)
        pred_diff = dj.get("predicted_difficulty", "judge_error")
        num_needed = dj.get("num_sentences_needed", 0)
        if not isinstance(num_needed, (int, float)):
            num_needed = 0

        c["llm_predicted_difficulty"] = pred_diff
        c["llm_num_sentences_needed"] = int(num_needed)
        c["llm_difficulty_judge"] = dj

        # Score: 1.0 if difficulty matches, else 0.0.
        # Tiebreak: for Hard prefer more sentences, for Easy prefer fewer.
        match_score = 1.0 if pred_diff == target_difficulty else 0.0
        if target_difficulty == "Hard":
            tiebreak = int(num_needed) / 100.0
        elif target_difficulty == "Easy":
            tiebreak = -int(num_needed) / 100.0
        else:
            tiebreak = 0.0

        score = match_score + tiebreak
        if score > best_score:
            best_score = score
            best = c

    if best is not None:
        best["llm_rerank_selected"] = True

    return (best, candidates)


def random_rerank(
    candidates: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select a random quality-passing candidate as baseline.

    Parameters
    ----------
    candidates : list[dict]
        Each must have ``"quality_pass"`` already set.
    seed : int

    Returns
    -------
    (best_or_None, all_scored_candidates)
    """
    import random as _random

    rng = _random.Random(seed)

    for c in candidates:
        c["random_rerank_selected"] = False

    passing = [c for c in candidates if c.get("quality_pass", False)]
    if not passing:
        return (None, candidates)

    chosen = rng.choice(passing)
    chosen["random_rerank_selected"] = True
    return (chosen, candidates)
