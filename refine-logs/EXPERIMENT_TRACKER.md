# Experiment Tracker

## M0: Data Pipeline (Days 1-5, ~$30 API)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Implement Stage B (counterfactual verification) | Code | — | Unit test | MUST | DONE | `dcqg/path/counterfactual_verify.py` |
| R002 | M0 | Implement Stage C (self-consistency aggregation) | Code | — | Unit test | MUST | DONE | `dcqg/path/self_consistency.py` + `scripts/run_evidence_audit_full.py` |
| R003 | M0 | Stage A re-audit: run 2/3, 2166 implicit | Qwen-32B audit | train-implicit | Parse ≥90% | MUST | TODO | ~$5 API |
| R004 | M0 | Stage A re-audit: run 3/3, 2166 implicit | Qwen-32B audit | train-implicit | Parse ≥90% | MUST | TODO | ~$5 API, parallel with R003 |
| R005 | M0 | Stage B: counterfactual verification (all 3 runs) | Qwen-32B counterfactual | train-implicit | — | MUST | TODO | ~$10 API, after R003+R004 |
| R006 | M0 | Stage C: self-consistency → labels_implicit.jsonl | Aggregation | train-implicit | Coverage ≥80%, Hard ≥250 | MUST | TODO | GATE: stop if fail |
| R007 | M0 | Audit 2000 explicit items (Easy, simplified) | Qwen-32B audit | train-explicit | Easy ≥1500 | MUST | TODO | ~$5 API, parallel with R003-R006 |
| R008 | M0 | Merge → train_dataset.jsonl | Pipeline | train | Total ≥3000 | MUST | TODO | After R006+R007 |
| R009 | M0 | Hard augmentation (controlled paraphrasing) | Qwen-32B paraphrase | train-Hard | Hard ≥400, valid ≥50% | MUST | TODO | ~$5 API |

## M1: Classifier Training (Days 5-8, ~22 GPU-hours)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R010 | M1 | Implement data pipeline ([Sn] tokenization) | Code: `dcqg/difficulty/data.py` | — | Unit test | MUST | DONE | `dcqg/difficulty/data.py` |
| R011 | M1 | Implement multi-task classifier | Code: `dcqg/difficulty/classifier.py` | — | Unit test | MUST | DONE | `dcqg/difficulty/classifier.py` + `scripts/train_classifier.py` |
| R012 | M1 | Sanity: overfit on 100 items | Multi-task DeBERTa | 100 train | Loss < 0.1 | MUST | TODO | 0.5 GPU-hr |
| R013 | M1 | Train single-task (5-fold CV) | DeBERTa difficulty-only | 70/15/15 | Dev macro F1 ≥ 0.55 | MUST | TODO | 8 GPU-hr |
| R014 | M1 | Train multi-task (5-fold CV) | DeBERTa multi-task λ=0.3 | 70/15/15 | Dev macro F1 ≥ 0.60 | MUST | TODO | 10 GPU-hr, GATE |
| R015 | M1 | Train λ ablation (λ=0, λ=1.0) | DeBERTa multi-task variants | 70/15/15 | — | MUST | TODO | 4 GPU-hr |

## M2: Classifier Evaluation — Claim 1 (Days 8-12, ~$3 API + 20 person-hours)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R016 | M2 | Select 200 test samples (stratified) | Sampling | test-15% | 67E+67M+66H | MUST | TODO | Can start after R008 |
| R017 | M2 | Human annotation: 200 × 2 annotators | Human | test | κ ≥ 0.60 | MUST | TODO | 20 person-hr, start early |
| R018 | M2 | LLM judges on 200 samples | GPT-4o-mini + Qwen-32B | test | — | MUST | TODO | ~$3 API |
| R019 | M2 | Classifier predictions on 200 samples | Multi-task + single-task | test | — | MUST | TODO | After R014 |
| R020 | M2 | Compute Table 1 + bootstrap CI + McNemar's | Analysis | test | F1 gap ≥5pp | MUST | TODO | GATE: core claim |
| R021 | M2 | Compute ablation metrics (Table 1 rows) | Analysis | test | — | MUST | TODO | — |

## M3: Reranking — Claim 2 (Days 10-15, ~$50 API)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R022 | M3 | Implement reranker | Code: `dcqg/difficulty/reranker.py` | — | Unit test | MUST | DONE | `dcqg/difficulty/reranker.py` + `scripts/run_k5_generation.py` + `scripts/run_reranking_eval.py` |
| R023 | M3 | Generate K=5: Direct, 150 items | Direct QG × 5 | valid-matched | Parse ≥85% | MUST | TODO | ~$8, can start during M1 |
| R024 | M3 | Generate K=5: ICL, 150 items | ICL QG × 5 | valid-matched | Parse ≥85% | MUST | TODO | ~$8 |
| R025 | M3 | Generate K=5: SelfRefine, 150 items | SelfRefine QG × 5 | valid-matched | Parse ≥85% | MUST | TODO | ~$12 |
| R026 | M3 | Generate K=5: Ours, 150 items | Ours QG × 5 | valid-matched | Parse ≥85% | MUST | TODO | ~$12 |
| R027 | M3 | Quality gate: all 3000 candidates | Quality judge | valid | Pass ≥50% | MUST | TODO | ~$5 |
| R028 | M3 | Classifier reranking | Multi-task classifier | valid | — | MUST | TODO | After R014 |
| R029 | M3 | LLM judge reranking | GPT-4o-mini | valid | — | MUST | TODO | ~$5 |
| R030 | M3 | Random selection baseline | Random | valid | — | MUST | TODO | — |
| R031 | M3 | Compute Table 2 + paired bootstrap | Analysis | valid | ≥5pp all methods | MUST | TODO | GATE |

## M4: Human Evaluation & Analysis (Days 13-18, 10 person-hours)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R032 | M4 | Sample 100: 50 reranked + 50 K=1 | Sampling | valid | Stratified | MUST | TODO | — |
| R033 | M4 | Human annotation: 100 × 2 annotators | Human | valid | κ ≥ 0.55 | MUST | TODO | 10 person-hr |
| R034 | M4 | Compute Table 3 | Analysis | valid | Reranked > K=1 | MUST | TODO | — |
| R035 | M4 | Error analysis: 30 samples | Analysis | valid | Error taxonomy | NICE | TODO | — |
| R036 | M4 | Interpretability: Figure 1 | Analysis | valid | 6 examples | NICE | TODO | — |
| R037 | M4 | Final tables assembly | Paper | all | All gates pass | MUST | TODO | — |

## Summary

- **Total runs:** 37
- **MUST-RUN:** 33
- **NICE-TO-HAVE:** 4 (R035, R036, and associated)
- **Estimated cost:** ~$83 API + 22 GPU-hours + 30 person-hours
- **Estimated time:** 18 days
