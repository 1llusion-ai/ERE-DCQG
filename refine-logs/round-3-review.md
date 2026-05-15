# Round 3 Review

**Reviewer:** Self-review (GPT-5.4 unavailable — API stream disconnected)

---

## Scores

| Dimension | R1 | R2 | R3 | Δ(R2→R3) |
|-----------|----|----|----|----|
| Problem Fidelity | 8 | 9 | 9 | 0 |
| Method Specificity | 6 | 7 | 9 | +2 |
| Contribution Quality | 5 | 7 | 8 | +1 |
| Frontier Leverage | 5 | 7 | 8 | +1 |
| Feasibility | 7 | 7 | 8 | +1 |
| Validation Focus | 7 | 8 | 9 | +1 |
| Venue Readiness | 6 | 7 | 8 | +1 |
| **Overall (weighted)** | **5.9** | **7.4** | **8.5** | **+1.1** |

Calculation: 9×0.15 + 9×0.25 + 8×0.25 + 8×0.15 + 8×0.10 + 9×0.05 + 8×0.05 = 1.35 + 2.25 + 2.00 + 1.20 + 0.80 + 0.45 + 0.40 = **8.45 → rounded 8.5**

**Verdict: REVISE** — Very close to READY. One dimension (Contribution Quality) blocks ≥9. Targeted fixes below.

---

## Dimension-Level Feedback

### Problem Fidelity: 9

Excellent. Both bottlenecks directly addressed. Theoretical grounding strengthens the "why." No drift.

### Method Specificity: 9 (+2)

Major improvement. The evidence audit is now fully specified (3-stage with counterfactual verification). Tokenization, architecture, training recipe, augmentation, and evaluation are all implementable. An engineer could start coding from this document.

### Contribution Quality: 8 (+1) — the remaining gap

**What's strong:** The reframe to mechanism-first (multi-task classifier) is better. The contribution structure is clean: one dominant (classifier), one supporting (audit pipeline). Theoretical grounding adds depth.

**Remaining weakness (IMPORTANT): The dual-use and interpretability angles are underemphasized.** The proposal currently frames the classifier as "evaluate + rerank." But there are two deeper novelty angles that aren't articulated strongly enough:

1. **Interpretability via evidence identification.** Unlike prior difficulty assessments (LLM judges, IRT scores, surface features), this classifier EXPLAINS its prediction — it identifies which sentences are the evidence. This is uniquely valuable in educational QG, where teachers need to understand why a question is classified as Hard. This is NOT just a multi-task auxiliary; it's an interpretable evaluation.

2. **Self-fulfilling evaluation risk needs explicit treatment.** Using the same classifier for reranking and evaluation creates a validity concern. The proposal mentions human evaluation as a secondary check, but should explicitly acknowledge: "Reranked questions are selected to maximize P(target_difficulty), so classifier-evaluated accuracy is an upper bound. Human evaluation on 100 unreranked+reranked pairs provides the ground-truth comparison." This shows methodological maturity that reviewers value.

**Fix:**
- Add a "Why interpretable difficulty assessment matters" paragraph connecting the evidence head to educational utility
- Explicitly address the reranker-evaluator circularity and explain how human evaluation breaks it
- Frame the dual-use as a methodological contribution: "the first difficulty assessment model designed to serve both as an evaluation instrument and a generation-time optimization signal"

Priority: IMPORTANT (this is the only thing standing between 8 and 9 on CQ)

### Frontier Leverage: 8 (+1)

Counterfactual verification is a good use of LLM reasoning. The proposal appropriately uses LLMs where they add value (annotation, generation) and trained models where they add value (fast, reproducible evaluation).

No further action needed for 8. Could push to 9 with: a brief note on why this "LLM-as-annotator → train small model" pattern is a frontier-native design choice (distillation from large to small, using the LLM's reasoning to create structured labels).

### Feasibility: 8 (+1)

70/15/15 split, 5-fold CV, bootstrap CI, pre-registered partial success — all demonstrate mature experimental planning. API cost estimates are realistic.

### Validation Focus: 9 (+1)

Excellent. Two clean claims, one focused ablation, no bloat, pre-registered partial success, appropriate statistics.

### Venue Readiness: 8 (+1)

Strong for CCF C. "Why now" narrative is present. Theoretical grounding adds weight. Educational application gives a "so what."

---

## Simplification Opportunities

**NONE.** The proposal is already tight. Every component serves the core claims. The complexity budget is well-managed.

## Modernization Opportunities

1. **(MINOR) Frame the overall approach as "LLM-to-small-model distillation for structured evaluation."** This connects to the broader trend of using large models to train specialized small models, which is a recognized frontier pattern. One sentence in the related work section would suffice.

## Drift Warning

**NONE.** The proposal remains tightly focused on the two original bottlenecks.

---

## Summary of Action Items for Round 3 Refinement (Final Push)

| Priority | Action | Target Score |
|----------|--------|-------------|
| IMPORTANT | Add interpretability-as-feature paragraph (evidence head explains difficulty, valuable for education) | CQ → 9 |
| IMPORTANT | Explicitly address reranker-evaluator circularity and how human eval breaks it | CQ → 9 |
| IMPORTANT | Frame dual-use as methodological contribution ("first model designed for both evaluation and reranking") | CQ → 9 |
| MINOR | Note the "LLM-as-annotator → small model" pattern as a frontier design choice | FL → 8.5 |

**Estimated impact:** These fixes are purely framing improvements — no pipeline changes. They should push CQ from 8 → 9, bringing overall from 8.5 → 9.0, which meets the READY threshold.
