"""Path module: evidence audit and answer-grounded evidence planning."""
from dcqg.path.fairytale_evidence_audit import FairytaleEvidenceAuditor, classify_difficulty
from dcqg.path.answer_grounded_evidence import plan_evidence, validate_evidence_plan

__all__ = [
    "FairytaleEvidenceAuditor",
    "classify_difficulty",
    "plan_evidence",
    "validate_evidence_plan",
]
