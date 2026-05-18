"""Canonical difficulty definitions — single source of truth.

All prompts, classifiers, and evaluation must use the definitions below
verbatim. Never paraphrase, shorten, or rewrite them.

Definition framework:
  1. Minimal evidence set: the smallest sentence set needed to answer and justify.
  2. Answer acquisition: directly found in the text vs. inferred.
  3. Evidence scope: one evidence sentence vs. multiple evidence sentences.

These definitions are consistent with classify_difficulty() in
fairytale_evidence_audit.py.
"""


MINIMAL_EVIDENCE_SET_DEFINITION = (
    "A minimal evidence set is the smallest set of sentences from which a "
    "reader can correctly answer the question and justify the target answer."
)

EVIDENCE_SENTENCE_DEFINITION = (
    "An evidence sentence is any sentence in the minimal evidence set. Each "
    "evidence sentence is necessary: removing it would make the question "
    "unanswerable, ambiguous, or no longer justified."
)


DIFFICULTY_DEFINITIONS = {
    "Easy": (
        "The answer can be directly found in the text; the minimal evidence "
        "set contains one evidence sentence."
    ),
    "Medium": (
        "Case 1: The answer cannot be directly found in the text; the "
        "minimal evidence set contains one evidence sentence and requires a "
        "simple inference. Case 2: The answer can be directly found in the "
        "text; however, the minimal evidence set contains multiple evidence "
        "sentences."
    ),
    "Hard": (
        "The answer cannot be directly found in the text; the minimal "
        "evidence set contains multiple evidence sentences and requires at "
        "least one inference."
    ),
}


def evidence_definitions_block():
    """Return the canonical evidence-set and evidence-sentence definitions."""
    return (
        "Minimal Evidence Set:\n"
        f"{MINIMAL_EVIDENCE_SET_DEFINITION}\n\n"
        "Evidence Sentence:\n"
        f"{EVIDENCE_SENTENCE_DEFINITION}"
    )


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
