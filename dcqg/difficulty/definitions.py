"""Canonical difficulty definitions — single source of truth.

All prompts, classifiers, and evaluation must use the definitions below
verbatim. Never paraphrase, shorten, or rewrite them.

Definition framework:
  1. Answer acquisition: directly found in the text vs. inferred.
  2. Evidence scope: one necessary sentence vs. multiple necessary sentences.

These definitions are consistent with classify_difficulty() in
fairytale_evidence_audit.py.
"""


DIFFICULTY_DEFINITIONS = {
    "Easy": (
        "The answer can be directly found in the text; obtaining the "
        "answer requires relying on only one necessary evidence sentence."
    ),
    "Medium": (
        "Case 1: The answer cannot be directly found in the text; obtaining "
        "the answer requires relying on one necessary evidence sentence and "
        "making a simple inference. Case 2: The answer can be directly found "
        "in the text; however, obtaining the answer requires synthesizing "
        "information from multiple necessary evidence sentences."
    ),
    "Hard": (
        "The answer cannot be directly found in the text; obtaining the "
        "answer requires synthesizing information from multiple necessary "
        "evidence sentences and performing complex implicit reasoning or "
        "multi-step reasoning."
    ),
}


def difficulty_definition(level):
    """Return the canonical difficulty definition with label prefix.

    Example: difficulty_definition("Easy") returns
    "Easy: The answer can be directly found in the text; ..."
    """
    text = DIFFICULTY_DEFINITIONS.get(level, DIFFICULTY_DEFINITIONS["Hard"])
    return f"{level}: {text}"


def difficulty_definitions_block():
    """Return all three canonical difficulty definitions as a formatted block.

    Suitable for injecting into prompts as {difficulty_definitions}.
    """
    lines = []
    for level in ["Easy", "Medium", "Hard"]:
        lines.append(f"{level}:\n{DIFFICULTY_DEFINITIONS[level]}")
    return "\n\n".join(lines)

