# Round 2 Review

**Reviewer:** Self-review (GPT-5.4 unavailable — API stream disconnected × 2)

**Review basis:** Same rubric as Round 1, applied to `round-1-refinement.md`. Focus on what improved and what still blocks a score ≥ 9.

---

## Scores

| Dimension | Round 1 | Round 2 | Δ |
|-----------|---------|---------|---|
| Problem Fidelity | 8 | 9 | +1 |
| Method Specificity | 6 | 7 | +1 |
| Contribution Quality | 5 | 7 | +2 |
| Frontier Leverage | 5 | 7 | +2 |
| Feasibility | 7 | 7 | 0 |
| Validation Focus | 7 | 8 | +1 |
| Venue Readiness | 6 | 7 | +1 |
| **Overall (weighted)** | **5.9** | **7.4** | **+1.5** |

Weighting: PF 15%, MS 25%, CQ 25%, FL 15%, F 10%, VF 5%, VR 5%.

Calculation: 9×0.15 + 7×0.25 + 7×0.25 + 7×0.15 + 7×0.10 + 8×0.05 + 7×0.05 = 1.35 + 1.75 + 1.75 + 1.05 + 0.70 + 0.40 + 0.35 = **7.35**

**Verdict: REVISE** — substantial improvement from 5.9 → 7.35, direction is right, but three dimensions still block ≥ 9.

---

## Dimension-Level Feedback

### Problem Fidelity: 9 (+1)

Strong. The ENDF reframing directly addresses both bottlenecks: (a) evidence necessity replaces structural hops, (b) trained classifier replaces LLM judges. The anchor check in the refinement doc is explicit and convincing.

No action needed.

### Method Specificity: 7 (+1) — needs 9

**Weakness 1 (CRITICAL): Evidence audit pipeline is still a black box.** The proposal says "LLM evidence analysis" but doesn't specify: What is the exact audit prompt? How are borderline cases handled? What is the self-consistency protocol? The `answer_grounded_evidence.py` module exists in code, but the proposal doesn't connect it to the training data pipeline or specify how audit quality is assured beyond the 200-sample human check.

**Fix:** Add a concrete audit procedure section:
1. Show the audit prompt (or reference the existing `build_answer_grounded_evidence_prompt`)
2. Specify the self-consistency protocol: run each item N=3 times, take majority vote for difficulty label. For evidence sentences, take the union of required sentences across N runs.
3. Define the audit→label mapping rules explicitly (these exist in the code but aren't in the proposal):
   - Easy: ASA=yes AND |required_evidence|=1
   - Medium: |required_evidence|=2 AND bridge_removal_effect ∈ {helpful, somewhat_helpful}
   - Hard: |required_evidence|≥3 AND bridge_removal_effect ∈ {ambiguous, unanswerable}
4. Report audit coverage: what fraction of FairytaleQA pairs get a valid, confident label?

**Weakness 2 (IMPORTANT): [Sn] marker tokenization is underspecified.** The proposal says "sentence markers [S0]...[Sn]" but doesn't clarify whether these are added as new special tokens to DeBERTa's vocabulary, how the evidence head locates them, or what happens when markers are split across subword tokens.

**Fix:** Specify:
- Add [S0]...[S30] as special tokens via `tokenizer.add_special_tokens({"additional_special_tokens": [f"[S{i}]" for i in range(31)]})`
- Resize model embeddings: `model.resize_token_embeddings(len(tokenizer))`
- Evidence head: at inference, collect the hidden states at positions of [Sn] tokens, apply binary classification head
- If a story has >30 sentences, merge the last sentences under [S30]

**Weakness 3 (MINOR): Hard class augmentation is hand-wavy.** "Back-translation Hard→Chinese→English" is not a proven technique for preserving difficulty labels. A question paraphrased via back-translation might become easier if the paraphrase simplifies the reasoning.

**Fix:** Replace back-translation with a more controlled augmentation:
- Use the LLM to generate alternative phrasings of the SAME Hard question, constrained to require the SAME evidence sentences (validate with the audit pipeline)
- Or: use synonym substitution + sentence reordering in the story (which changes surface but not evidence structure)
- Report augmented-Hard dev F1 vs. raw-Hard dev F1 to verify augmentation quality

### Contribution Quality: 7 (+2) — needs 9

**Weakness 1 (CRITICAL): The evidence-necessity definition needs theoretical grounding.** The proposal operationalizes difficulty as "minimum evidence sentences," but doesn't connect this to any theory of reading comprehension difficulty. This makes the framework feel empirically motivated but theoretically thin. A reviewer would ask: "Why is sentence count the right proxy for difficulty?"

**Fix:** Add a brief theoretical grounding section:
- Connect to Kintsch's Construction-Integration model: difficulty increases when readers must integrate information from more text segments (propositions across sentences)
- Connect to Coh-Metrix text difficulty dimensions: referential cohesion, text connectivity, situation model difficulty — all relate to how many text segments must be connected
- Cite FairytaleQA's own design principles (Xu et al., 2022): implicit questions are harder BECAUSE the answer requires reading multiple sentences
- The evidence-necessity count operationalizes "integration load" — a reader-centric metric, not a text-centric one

**Weakness 2 (IMPORTANT): "Framework" framing is slightly overclaimed for the mechanism.** ENDF is presented as a 3-part framework (definition + audit + classifier), but the audit is a one-time LLM call and the definition is a paragraph. The real technical novelty is the multi-task classifier and its use for reranking. Overclaiming the framework risks a "this is just labeling + fine-tuning" dismissal.

**Fix:** Tighten the framing:
- Lead with the technical mechanism: "We propose a multi-task classifier that jointly predicts question difficulty and identifies evidence sentences, trained on evidence-necessity labels"
- Present the definition and audit as the principled data construction method, not a separate contribution
- The novelty is the JOINT training + the evidence-grounded labels + the reranking application — not the framework umbrella

### Frontier Leverage: 7 (+2) — could push to 8

**Observation:** The proposal appropriately uses LLMs for audit (data labeling) and generation, and a trained classifier for evaluation and reranking. This is a sensible allocation.

**Weakness (IMPORTANT): The audit pipeline doesn't leverage LLM reasoning capabilities well enough.** A single LLM call per item is fragile. Modern best practice uses chain-of-thought + self-consistency.

**Fix:** Upgrade the audit to a 2-stage protocol:
- Stage A: Chain-of-thought evidence analysis (the existing `answer_grounded_evidence.py` prompt)
- Stage B: Verification — given the plan from Stage A, ask: "If I remove sentence [Sk] from the evidence, can the question still be answered?" for each bridge/support sentence. This is a targeted counterfactual that directly validates evidence necessity.
- This makes the audit quality higher AND provides a stronger training signal for the classifier.

### Feasibility: 7 (unchanged)

**Weakness (IMPORTANT): Hard class size (330→660 with augmentation) is the single largest risk.** With a 80/10/10 split, the Hard test set has only ~66 examples. Per-class F1 on 66 examples has wide confidence intervals.

**Fix:** 
- Report bootstrap 95% CI for all per-class metrics
- Consider a larger test set for Hard: use 70/10/20 split instead of 80/10/10, giving ~132 Hard test examples
- Or: run 5-fold cross-validation and report mean ± std

### Validation Focus: 8 (+1)

Clean. Two claims, one ablation, minimal bloat. The negative-result protocol could be slightly stronger.

**Minor fix:** Pre-register what happens if ENDF only helps Easy/Medium but not Hard (possible given the class size). State this as an acceptable partial finding rather than a failure.

### Venue Readiness: 7 (+1) — could push to 8

**Weakness (MINOR): The "why now" narrative is weak.** The proposal should articulate why evidence-necessity is timely.

**Fix:** One paragraph: "As LLM-based evaluation becomes standard but its unreliability is increasingly documented (Liu et al., 2023; Wang et al., 2024), difficulty-controlled QG urgently needs reproducible evaluation. Simultaneously, the availability of LLMs as data annotators makes it feasible to create evidence-necessity labels at scale — something that would have required prohibitive human annotation five years ago. ENDF exploits both trends."

---

## Simplification Opportunities

1. **Merge the quality gate into the reranking score.** Instead of binary quality gate → argmax P(difficulty), use a single ranking: `score = is_quality_pass * P(target_difficulty)` where `is_quality_pass ∈ {0, 1}`. This removes the separate fallback logic and simplifies the pipeline description.

2. **Drop the Longformer fallback.** The proposal mentions switching to Longformer if truncation is an issue. This is a distraction. FairytaleQA story sections are typically 5-15 sentences (100-300 words). With evidence-focused truncation, 512 tokens is almost certainly sufficient. State this and drop the fallback.

## Modernization Opportunities

1. **Upgrade audit to counterfactual verification (as described above).** This is the highest-leverage modernization — it makes the training data more reliable without adding pipeline complexity.

2. **Consider using DeBERTa-v3-large instead of base if Hard F1 is too low.** The compute increase (from ~4 to ~12 GPU-hours) is within budget and may help with the small Hard class.

## Drift Warning

**NONE.** The proposal still directly addresses both original bottlenecks: evidence-necessity replaces structural hops, trained classifier replaces LLM judges.

---

## Summary of Action Items for Round 2 Refinement

| Priority | Action | Dimension |
|----------|--------|-----------|
| CRITICAL | Specify the evidence audit procedure concretely (prompt, self-consistency, label mapping, coverage) | Method Specificity |
| CRITICAL | Ground evidence-necessity in reading comprehension theory (Kintsch, Coh-Metrix) | Contribution Quality |
| IMPORTANT | Specify [Sn] marker tokenization and evidence head mechanics | Method Specificity |
| IMPORTANT | Replace back-translation augmentation with controlled paraphrasing | Method Specificity |
| IMPORTANT | Tighten framework framing — lead with mechanism, not umbrella | Contribution Quality |
| IMPORTANT | Add counterfactual verification to audit pipeline | Frontier Leverage |
| IMPORTANT | Address Hard class size risk: larger test split or cross-validation | Feasibility |
| MINOR | Add "why now" paragraph | Venue Readiness |
| MINOR | Pre-register partial success scenario (Easy/Medium only) | Validation Focus |
| MINOR | Drop Longformer fallback | Simplification |
