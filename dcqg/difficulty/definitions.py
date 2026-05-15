"""Unified natural-language difficulty definitions.

Definitions follow the evidence-necessity framework:
  Easy:   1 evidence sentence, which alone suffices
  Medium: 2 evidence sentences
  Hard:   3+ evidence sentences

These definitions are used by baselines (as prompt instructions), classifiers
(as training targets), and evaluation (as ground truth). They must be
consistent with classify_difficulty() in fairytale_evidence_audit.py.
"""


DIFFICULTY_INSTRUCTIONS = {
    "Easy": (
        "Easy: the question can be answered from exactly one sentence. "
        "That sentence alone is sufficient — no other sentence is needed."
    ),
    "Medium": (
        "Medium: the question requires exactly two evidence sentences. "
        "Neither sentence alone is sufficient; both must be combined to answer."
    ),
    "Hard": (
        "Hard: the question requires three or more evidence sentences. "
        "The reader must synthesize information across multiple sentences "
        "to determine the answer."
    ),
}


DIFFICULTY_FRAMEWORK = """Difficulty is determined by the minimum number of evidence sentences a reader must consult to answer correctly.

Easy:
- exactly 1 evidence sentence is needed;
- that sentence alone is sufficient to answer.

Medium:
- exactly 2 evidence sentences are needed;
- neither sentence alone is sufficient; both must be combined.

Hard:
- 3 or more evidence sentences are needed;
- the reader must synthesize information across multiple sentences."""


def difficulty_instruction(difficulty):
    """Return the unified difficulty definition string."""
    return DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["Hard"])
