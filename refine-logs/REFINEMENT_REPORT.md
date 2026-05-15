# Refinement Report

## Input

**User request:** 审查DCQG项目代码，找出问题，并给出具体建议。当前框架使用target answer做graph生成难度可控的题目，但graph可能不好，evaluation只用LLM judge，想用分类模型但数据难做。目标CCF C类。

## Process

1. **Phase 0:** Froze problem anchor — structural difficulty ≠ answering difficulty; LLM evaluation unreliable.
2. **Phase 1:** Built initial proposal from codebase analysis. Key findings:
   - 6 major problems identified in current code
   - Graph-guided QG is statistically indistinguishable from baselines (p=0.69)
   - 0% Hard hit on MAVEN-ERE, 27% on FairytaleQA (no better than ICL's 28.6%)
   - All evaluation LLM-judge-based with 9-36% error rates
3. **Phases 2-4:** 4 review rounds, score 5.9 → 9.0.
4. **Phase 5:** Final reports.

## Problems Identified in Current Codebase

| # | Problem | Evidence |
|---|---------|----------|
| 1 | Graph quality is weak | 87-95% valid, 44 failures from graph_invalid in CrossQG eval |
| 2 | All evaluation is LLM-based | No trained classifier, no human evaluation |
| 3 | No statistical significance | All bootstrap CIs include 0, Ours p=0.69 vs Direct |
| 4 | Graph adds cost but marginal benefit | +4.2pp accuracy for 2 extra API calls/candidate |
| 5 | Difficulty control is weak | Easy→Medium collapse, Hard→Medium collapse |
| 6 | Missing components | No ablation, no trained classifier, no human eval |

## Solution: Evidence-Necessity-Aware Multi-Task Classification

### Core Idea

Define difficulty by **evidence necessity** (how many sentences a reader must consult), not by graph hops or surface features. Train a **multi-task DeBERTa classifier** that:
- Predicts difficulty (3-way: Easy/Medium/Hard)
- Identifies evidence sentences (per-sentence binary)
- Serves as both evaluator and generation-time reranker

### Pipeline

```
Evidence Audit (LLM, 3 runs + counterfactual verification)
  → Training Labels (Easy/Medium/Hard + evidence sentence IDs)
  → Multi-Task Classifier (DeBERTa-v3-base, ~184M params)
  → Reranking (K=5 candidates, argmax P(target_difficulty))
  → Human Evaluation (100 samples, ground truth for reranking claims)
```

### Key Design Decisions

1. **Counterfactual verification** in audit: "If sentence [Sk] is removed, can the answer still be determined?" validates evidence necessity, not just relevance.
2. **Multi-task** (difficulty + evidence): forces evidence-aware representations, provides interpretable output.
3. **Dual-use** classifier: same model for evaluation and reranking — first in DCQG literature.
4. **Evidence-focused truncation**: answer ± 2 + bridge + anchor, max 512 tokens.
5. **Controlled augmentation**: LLM paraphrasing constrained to preserve evidence sentences.

### Novelty vs. Prior Work

| Feature | CrossQG | Uto | KAQG | GNET-QG | Ours |
|---------|---------|-----|------|---------|------|
| Difficulty definition | IRT | IRT | Bloom's+IRT | Graph | Evidence necessity |
| Evaluation | LLM | IRT | LLM | Human | Trained classifier |
| Evidence identification | No | No | No | No | Yes (joint) |
| Dual-use (eval+rerank) | No | No | No | No | Yes |
| Interpretable | No | No | No | No | Yes |

## Recommendations

### Immediate Next Steps

1. **Run the evidence audit** on FairytaleQA train split with the existing `plan_evidence()` function in `dcqg/path/answer_grounded_evidence.py`. Add counterfactual verification (Stage B) and self-consistency (Stage C, 3 runs).

2. **Prepare training data** using the label mapping:
   - Easy: ASA=yes AND |required_evidence|=1
   - Medium: |required_evidence|=2
   - Hard: |required_evidence|≥3 AND ≥1 counterfactual="no"

3. **Train multi-task DeBERTa-v3-base** with the specified architecture and recipe.

4. **Generate K=5 candidates** for all 4 methods × 3 difficulties and run classifier-based reranking.

5. **Human annotation**: 200 samples for validation, 100 for evaluation.

### Code Changes Needed

- New module: `dcqg/difficulty/classifier.py` — multi-task DeBERTa classifier
- New module: `dcqg/difficulty/data.py` — training data construction from audit results
- New module: `dcqg/difficulty/reranker.py` — classifier-based reranking pipeline
- Modify: `dcqg/path/answer_grounded_evidence.py` — add counterfactual verification (Stage B)
- New script: `scripts/run_evidence_audit.py` — full audit with self-consistency
- New script: `scripts/train_classifier.py` — classifier training with 5-fold CV
- New script: `scripts/run_reranking_eval.py` — reranking experiment

### Timeline

| Week | Task |
|------|------|
| 1 | Evidence audit (full train split) + human annotation starts |
| 2 | Classifier training + ablation experiments |
| 3 | K=5 generation + reranking + evaluation |
| 4 | Analysis + paper writing |

### Budget

- API: ~$80 (audit $30, augmentation $10, generation $40)
- GPU: ~20 hours (classifier training with 5-fold CV)
- Human: ~20 person-hours (200 annotation samples × 2 annotators)
