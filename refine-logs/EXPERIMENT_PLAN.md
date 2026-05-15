# Experiment Plan

**Problem**: Difficulty-controlled QG where difficulty = minimum evidence sentences needed. Current methods use surface/structural features; LLM judges are unreliable (9-36% error).

**Method Thesis**: A multi-task DeBERTa classifier that jointly predicts difficulty and identifies evidence sentences — trained on counterfactual-verified evidence-necessity labels — provides reproducible evaluation and enables generation-time reranking.

**Date**: 2026-05-15

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: Multi-task classifier > LLM judges for difficulty assessment | Core bottleneck: LLM judges are unreliable | Macro F1 improvement ≥5pp over best LLM judge, κ(classifier,human) > κ(LLM,human), on 200 human-annotated samples | B1, B2 |
| C2: Classifier-based reranking improves difficulty control for all QG methods | Closes the generation→evaluation loop | Difficulty accuracy improves ≥5pp for all 4 methods (K=1→K=5), validated by human evaluation on 100 samples | B3, B4 |

**Anti-claims to rule out:**
- "The classifier just memorizes LLM audit biases" → Addressed by human validation (200 samples) and classifier-human κ
- "Multi-task doesn't help, single-task is equally good" → Ablation B2
- "Reranking improvement is circular (optimized by classifier, evaluated by classifier)" → Human evaluation B4

---

## Paper Storyline

**Main paper must prove:**
1. The classifier outperforms LLM judges on human-annotated test set (Table 1)
2. Multi-task > single-task (Table 1, ablation row)
3. Reranking improves all 4 QG methods (Table 2)
4. Human evaluation confirms reranking benefit (Table 3)

**Appendix can support:**
- Evidence head interpretability examples (Figure A1)
- Audit self-consistency statistics (Table A1)
- Per-story-type analysis
- Augmentation quality analysis

**Experiments intentionally cut:**
- Cross-dataset transfer (out of scope for CCF C)
- Comparison with IRT-based methods (no learner data)
- Graph ablation (graph is not a contribution)

---

## Existing Assets

| Asset | Location | Status | Reusable? |
|-------|----------|--------|-----------|
| FairytaleQA data | HuggingFace `WorkInTheDark/FairytaleQA` | Available at runtime | Yes |
| Stage A audit (2166 implicit, 1 run) | `outputs/runs/fairytale_evidence_audit_train_implicit_2166_20260511/candidates.jsonl` | Complete (Easy=659, Med=1153, Hard=354) | Yes (as run 1 of 3) |
| Stage A code | `dcqg/path/answer_grounded_evidence.py` + `scripts/run_answer_grounded_evidence_audit.py` | Working | Yes |
| Difficulty definitions | `dcqg/difficulty/definitions.py` | Working | Yes |
| CrossQG eval results (K=1) | `outputs/runs/fairytale_qg_crossqg_eval_20260511_v1/` | 1800 items (450×4 methods) | Yes, as K=1 baseline |
| 4 QG methods | `dcqg/generation/fairytale_qg.py` + `dcqg/generation/baselines.py` | Working | Yes |
| FairytaleQA train stats | 8548 total: 6382 explicit + 2166 implicit | — | — |

**Must build:**
- Stage B: counterfactual verification module
- Stage C: self-consistency aggregation
- Multi-task DeBERTa classifier + [Sn] tokenization
- Reranking pipeline
- K=5 generation script

---

## Experiment Blocks

### Block 1 (B1): Classifier vs. LLM Judges — Main Anchor

- **Claim tested:** C1 — multi-task classifier provides more reliable difficulty assessment than LLM judges
- **Why this block exists:** Core contribution. If the classifier doesn't beat LLM judges, the paper has no anchor.
- **Dataset / split / task:** 200 human-annotated FairytaleQA QA pairs (stratified: 67 Easy + 67 Medium + 66 Hard from the 15% held-out test split). 2 annotators, Cohen's κ reported.
- **Compared systems:**
  1. **Multi-task DeBERTa-v3-base** (difficulty + evidence heads) — proposed method
  2. **GPT-4o-mini judge** (zero-shot difficulty prompt, same as current pipeline)
  3. **Qwen-32B judge** (zero-shot difficulty prompt)
  4. **DeBERTa difficulty-only** (single-task, same data — ablation, also B2)
  5. **Majority baseline** (predict most common class)
- **Metrics:**
  - Primary: Macro F1, per-class F1 (Easy/Medium/Hard)
  - Secondary: Cohen's κ vs. human, accuracy
  - Evidence-specific: evidence sentence recall, precision, F1 (multi-task only)
  - All with bootstrap 95% CI (1000 resamples)
- **Setup details:**
  - Classifier: DeBERTa-v3-base, 184M params, [S0]-[S30] markers, evidence-focused truncation (max 512 tokens)
  - Training: 5-fold stratified CV on audit-labeled train data (~3500 items)
  - Report: mean ± std across 5 folds on the 200-sample human test set
  - LLM judges: temperature=0.0, run 3 times, majority vote
- **Success criterion:** Multi-task macro F1 > best LLM judge macro F1 by ≥5pp; κ(classifier,human) > κ(LLM,human)
- **Failure interpretation:** If classifier ≤ LLM judge, evidence-necessity labels may be too noisy. Check: (a) human-audit agreement, (b) per-class breakdown, (c) whether single-task also fails.
- **Table / figure target:** Table 1 (main paper)
- **Priority:** MUST-RUN

### Block 2 (B2): Multi-Task vs. Single-Task — Novelty Isolation

- **Claim tested:** C1 (sub-claim) — multi-task training (difficulty + evidence) outperforms single-task (difficulty only)
- **Why this block exists:** The multi-task design is the core novelty. If single-task performs equally, the evidence head is decorative.
- **Dataset / split / task:** Same 200-sample human test set as B1. Also report on full 15% test split from audit data.
- **Compared systems:**
  1. **Multi-task DeBERTa** (difficulty + evidence, λ=0.3)
  2. **Single-task DeBERTa** (difficulty only, same architecture minus evidence head)
  3. **Multi-task with λ=0** (evidence head exists but loss weight is 0 — architecture ablation)
  4. **Multi-task with λ=1.0** (equal weight — sensitivity check)
- **Metrics:** Macro F1, per-class F1, evidence recall
- **Setup details:** Same 5-fold CV. All variants use identical training data, backbone, lr, batch size. Only difference: loss configuration.
- **Success criterion:** Multi-task (λ=0.3) improves Hard F1 by ≥3pp over single-task
- **Failure interpretation:** If no improvement, evidence signal doesn't help for this data size. Pivot to single-task classifier as contribution; drop multi-task angle.
- **Table / figure target:** Table 1 (ablation rows)
- **Priority:** MUST-RUN

### Block 3 (B3): Reranking Effectiveness — Generation-Time Impact

- **Claim tested:** C2 — classifier-based reranking improves difficulty accuracy for all 4 QG methods
- **Why this block exists:** The dual-use claim. Without this, the classifier is only an evaluator.
- **Dataset / split / task:** FairytaleQA validation split, story-matched CrossQG setup. 150 QA items (50 Easy + 50 Medium + 50 Hard) × 4 methods = 600 items.
- **Compared systems:**
  1. **K=1 (no reranking)** — single generation per item (reuse existing CrossQG eval)
  2. **K=5 + classifier reranking** — generate 5, quality gate, argmax P(target_difficulty)
  3. **K=5 + LLM judge reranking** — generate 5, quality gate, LLM picks best match
  4. **K=5 + random selection** — generate 5, quality gate, random pick
- **Metrics:**
  - Primary: Difficulty accuracy (classifier-evaluated), macro accuracy
  - Secondary: Per-level hit rate, Spearman ρ
  - Per-method breakdown: 4 methods × 4 conditions
  - All with bootstrap 95% CI, paired bootstrap test
- **Setup details:**
  - Generation: Qwen-32B API, K=5 independent samples, temperature=0.7
  - Quality gate: answerable ∧ asks_expected_answer ∧ fluent (binary)
  - Classifier reranking: argmax P(target_difficulty) among quality-passing
  - K=1 baseline: reuse existing CrossQG eval results
- **Success criterion:** Classifier reranking improves macro accuracy ≥5pp over K=1 for ALL 4 methods; Hard hit rate ≥+10pp for ≥3 methods
- **Failure interpretation:** If K=5 pool lacks diversity, increase temperature or K. If helps only some methods, report partial.
- **Table / figure target:** Table 2 (main paper)
- **Priority:** MUST-RUN

### Block 4 (B4): Human Evaluation — Ground-Truth Validation

- **Claim tested:** C2 ground-truth — reranking benefit holds when measured by humans, not just classifier
- **Why this block exists:** Breaks reranker-evaluator circularity.
- **Dataset / split / task:** 100 questions: 50 classifier-reranked + 50 K=1. Sampled from B3, stratified by difficulty and method. Blind to annotators.
- **Compared systems:** Same questions evaluated by (a) 2 human annotators, (b) classifier, (c) LLM judge
- **Metrics:** Difficulty accuracy (human-judged), κ(human-human), κ(classifier-human), κ(LLM-human)
- **Setup details:**
  - Present: story + question + answer. Annotator labels Easy/Medium/Hard.
  - 2 annotators, disagreements resolved by discussion, order randomized, source blinded.
- **Success criterion:** Human-evaluated difficulty accuracy higher for reranked by ≥5pp; κ(classifier,human) > κ(LLM,human)
- **Failure interpretation:** If humans see no difference, classifier captures surface patterns, not cognitive difficulty.
- **Table / figure target:** Table 3 (main paper)
- **Priority:** MUST-RUN

### Block 5 (B5): Error Analysis and Interpretability

- **Claim tested:** Interpretability claim + failure diagnosis
- **Why this block exists:** Supports "interpretable assessment" framing; diagnoses remaining gaps.
- **Analysis:**
  1. Evidence head accuracy vs. audit gold (recall, precision, F1)
  2. Error taxonomy: wrong difficulty + correct/wrong evidence
  3. 6 qualitative examples (2 per difficulty)
  4. Failure pattern identification
- **Table / figure target:** Table 4 (error analysis), Figure 1 (examples)
- **Priority:** NICE-TO-HAVE

---

## Run Order and Milestones

### M0: Data Pipeline (Days 1-5, ~$30 API)

**Goal:** Produce finalized training labels with self-consistency and counterfactual verification.

| Run | Task | Depends On | Cost | Gate |
|-----|------|-----------|------|------|
| R001 | Implement Stage B (counterfactual verification) in `dcqg/path/answer_grounded_evidence.py` | — | — | Unit test |
| R002 | Implement Stage C (self-consistency aggregation) script | — | — | Unit test |
| R003 | Stage A re-audit: run 2 on 2166 implicit items | R001 | ~$5 | Parse ≥90% |
| R004 | Stage A re-audit: run 3 on 2166 implicit items | R001 | ~$5 | Parse ≥90% |
| R005 | Stage B: counterfactual verification on all 3 runs | R001,R003,R004 | ~$10 | — |
| R006 | Stage C: self-consistency aggregation → `labels_implicit.jsonl` | R005 | — | Coverage ≥80%, Hard ≥250 |
| R007 | Audit 2000 explicit items (Easy, simplified audit) | — | ~$5 | Easy ≥1500 |
| R008 | Merge implicit + explicit → `train_dataset.jsonl` | R006,R007 | — | Total ≥3000 |
| R009 | Hard augmentation (controlled paraphrasing + re-audit) | R008 | ~$5 | Hard ≥400 |

**Parallelism:** R001+R002 (code) in parallel. Then R003+R004+R007 (API) in parallel. Then R005→R006→R008→R009.

**STOP gate:** If coverage < 70% OR Hard raw < 200 after R006, diagnose before M1.

### M1: Classifier Training (Days 5-8, ~22 GPU-hours)

**Goal:** Trained multi-task and single-task classifiers with 5-fold CV.

| Run | Task | Depends On | Cost | Gate |
|-----|------|-----------|------|------|
| R010 | Implement `dcqg/difficulty/data.py` ([Sn] tokenization + dataset class) | R008 | — | Unit test |
| R011 | Implement `dcqg/difficulty/classifier.py` (multi-task DeBERTa) | — | — | Unit test |
| R012 | Sanity: overfit on 100 items | R010,R011 | 0.5h | Loss < 0.1 |
| R013 | Train single-task (5-fold CV) | R012 | 8h | Dev macro F1 ≥ 0.55 |
| R014 | Train multi-task (5-fold CV) | R012 | 10h | Dev macro F1 ≥ 0.60 |
| R015 | Train λ ablation (λ=0, λ=1.0, 5-fold CV) | R012 | 4h | — |

**STOP gate:** If multi-task dev macro F1 < 0.55, try DeBERTa-v3-large or simplify to single-task.

### M2: Classifier Evaluation — Claim 1 (Days 8-12, ~$3 API + 20 person-hours)

**Goal:** Prove classifier > LLM judges on human test set.

| Run | Task | Depends On | Cost | Gate |
|-----|------|-----------|------|------|
| R016 | Select 200 test samples (stratified from 15% test split) | R008 | — | — |
| R017 | Human annotation: 200 samples × 2 annotators | R016 | 20h human | κ ≥ 0.60 |
| R018 | LLM judges (GPT-4o-mini + Qwen-32B) on 200 samples | R016 | ~$3 | — |
| R019 | Classifier predictions on 200 samples | R014,R016 | <1 min | — |
| R020 | Compute Table 1 metrics + bootstrap CI + McNemar's | R017-R019 | — | F1 gap ≥5pp |
| R021 | Compute ablation metrics (Table 1 rows) | R013-R015,R017 | — | — |

**Note:** R016-R017 can start as soon as R008 (dataset) is ready, overlapping with M1.

### M3: Reranking — Claim 2 (Days 10-15, ~$50 API)

**Goal:** Show reranking improves all 4 QG methods.

| Run | Task | Depends On | Cost | Gate |
|-----|------|-----------|------|------|
| R022 | Implement `dcqg/difficulty/reranker.py` | R014 | — | Unit test |
| R023 | Generate K=5: Direct, 150 items | — | ~$8 | Parse ≥85% |
| R024 | Generate K=5: ICL, 150 items | — | ~$8 | Parse ≥85% |
| R025 | Generate K=5: SelfRefine, 150 items | — | ~$12 | Parse ≥85% |
| R026 | Generate K=5: Ours, 150 items | — | ~$12 | Parse ≥85% |
| R027 | Quality gate on all 3000 candidates | R023-R026 | ~$5 | Pass ≥50% |
| R028 | Classifier reranking | R027,R014 | <1 min | — |
| R029 | LLM judge reranking | R027 | ~$5 | — |
| R030 | Random selection baseline | R027 | — | — |
| R031 | Compute Table 2 metrics + paired bootstrap | R028-R030 + K=1 data | — | ≥5pp for all |

**Note:** R023-R026 (generation) can start during M1/M2 since they don't depend on the classifier.

### M4: Human Evaluation & Analysis (Days 13-18, 10 person-hours)

**Goal:** Ground-truth validation + interpretability analysis.

| Run | Task | Depends On | Cost | Gate |
|-----|------|-----------|------|------|
| R032 | Sample 100 items: 50 reranked + 50 K=1 (blind, stratified) | R028 | — | — |
| R033 | Human annotation: 100 items × 2 annotators | R032 | 10h human | κ ≥ 0.55 |
| R034 | Compute Table 3 metrics | R033 | — | Reranked > K=1 |
| R035 | Error analysis: 30 samples | R028,R014 | — | — |
| R036 | Interpretability examples: Figure 1 | R035 | — | — |
| R037 | Final paper tables assembly | R020,R021,R031,R034 | — | All gates pass |

---

## Timeline Gantt

```
Day:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
M0:  [========R001-R009========]
M1:                      [======R010-R015======]
M2:               [R016-R017 annotation ===========================]
      (annotation can overlap)  [R018-R021]
M3:                    [R023-R026 generation=======][R027-R031]
M4:                                                      [====R032-R037====]
```

**Critical path:** M0 → M1 → M2(R020) + M3(R028) → M4 → paper

**Parallelizable:** Human annotation (R017) overlaps M1. Generation (R023-R026) overlaps M1/M2. Explicit audit (R007) overlaps implicit re-audits.

---

## Compute and Data Budget

| Category | Items | Cost |
|----------|-------|------|
| API: Evidence audit | 2166×2 re-audits + ~2166 counterfactual + 2000 explicit + augmentation | ~$30 |
| API: K=5 generation | 150×4×5=3000 generations + quality gate + LLM reranking | ~$50 |
| API: LLM judges | 200 × 2 judges × 3 runs | ~$3 |
| GPU: Classifier | 5-fold CV × (multi+single+2 ablations) × 10 epochs | ~22 hours |
| Human: Annotation | 200 validation + 100 evaluation = 300 × 2 annotators × 3 min | ~30 person-hours |
| **Total** | | **~$83 API, ~22 GPU-hours, ~30 person-hours** |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hard class too small (<250 post-consistency) | Medium | High | Relax to 2/3 agreement; augment more; pre-register partial success |
| Counterfactual verification inconsistent | Medium | Medium | 3 runs + majority; if too noisy, use Stage A labels directly |
| Classifier doesn't beat LLM judges | Low | Critical | Check per-class; pivot to "reproducibility + speed" argument |
| K=5 pool lacks diversity | Medium | Medium | Increase temp to 0.8; try K=10 |
| Human annotators disagree (κ < 0.50) | Low | High | Improve guidelines; pilot round with 20 items |
| DeBERTa truncation loses evidence | Low | Low | FairytaleQA sections are 5-15 sentences; truncation handles this |

---

## Paper Table Layout

| Table | Content | Blocks | Position |
|-------|---------|--------|----------|
| Table 1 | Classifier vs. LLM judges + ablation: macro F1, per-class F1, κ, evidence metrics | B1, B2 | Main |
| Table 2 | Reranking: 4 methods × 4 conditions, macro acc, per-level hit, Spearman | B3 | Main |
| Table 3 | Human evaluation: reranked vs. K=1, accuracy, κ agreements | B4 | Main |
| Table 4 | Error analysis: types, patterns, confusion matrix | B5 | Main/Appendix |
| Figure 1 | Interpretability examples: story + question + predicted difficulty + evidence | B5 | Main |
| Table A1 | Audit self-consistency statistics | M0 | Appendix |

---

## Final Checklist

- [x] Main paper tables covered (Tables 1-4, Figure 1)
- [x] Novelty isolated (B2: multi-task vs. single-task)
- [x] Simplicity defended (no unnecessary components)
- [x] Frontier contribution justified (LLM-as-annotator → small model)
- [x] Nice-to-have separated from must-run (B5)
- [x] Reranker-evaluator circularity addressed (B4: human evaluation)
- [x] Pre-registered partial success for Hard class
- [x] All gates specified with fallback actions
- [x] Dependencies and parallelism mapped
