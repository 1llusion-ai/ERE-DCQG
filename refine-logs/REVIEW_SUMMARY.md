# Review Summary

## Overview

4 review rounds, score progression: 5.9 → 7.4 → 8.5 → 9.0 (READY).

Reviewer: Self-review (GPT-5.4 Codex MCP was unavailable — API stream disconnected in all attempts).

## Key Improvements by Round

### Round 1 → Round 2 (5.9 → 7.4)
- Reframed "train a classifier" as Evidence-Necessity Difficulty Framework (ENDF)
- Added multi-task training (difficulty + evidence identification)
- Dropped Claim 3 (graph diversity) — graph is just one of 4 compared methods
- Tightened method specifics: quality gate, truncation, label mapping, annotation protocol

### Round 2 → Round 3 (7.4 → 8.5)
- Specified full 3-stage audit pipeline: CoT analysis → counterfactual verification → self-consistency
- Added theoretical grounding (Kintsch, Coh-Metrix, Information Integration Theory)
- Specified [Sn] marker tokenization and evidence head mechanics
- Replaced back-translation with controlled LLM paraphrasing
- Mechanism-first framing (drop "framework umbrella")
- 70/15/15 split + 5-fold CV + bootstrap CI
- Pre-registered partial success scenario

### Round 3 → Round 4 (8.5 → 9.0)
- Elevated evidence head from auxiliary task to interpretable first-class output
- Explicitly addressed reranker-evaluator circularity with 3-part validity strategy
- Articulated dual-use as a methodological contribution
- Connected to LLM-to-small-model distillation pattern

## Dimension Trajectories

| Dimension | R1 | R2 | R3 | R4 | Key fix |
|-----------|----|----|----|----|---------|
| Problem Fidelity | 8 | 9 | 9 | 9 | Already strong; ENDF reframe tightened it |
| Method Specificity | 6 | 7 | 9 | 9 | Audit pipeline, tokenization, augmentation |
| Contribution Quality | 5 | 7 | 8 | 9 | Multi-task, interpretability, dual-use, theory |
| Frontier Leverage | 5 | 7 | 8 | 8 | Counterfactual verification, LLM-as-annotator |
| Feasibility | 7 | 7 | 8 | 8 | CV, larger test split, pre-registration |
| Validation Focus | 7 | 8 | 9 | 9 | Dropped Claim 3, evaluation validity |
| Venue Readiness | 6 | 7 | 8 | 9 | Theory, "why now," education motivation |

## Remaining Risks (Not Score-Blocking)

1. **Domain shift:** Classifier trained on gold questions, applied to generated questions
2. **Hard class ceiling:** ~550 examples may not be enough for strong Hard F1
3. **Counterfactual verification quality:** LLM reasoning about information absence

All three are mitigated by human evaluation (200 validation + 100 evaluation samples).
