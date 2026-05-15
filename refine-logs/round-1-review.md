# Round 1 Review

## Scores

| Dimension | Score | Assessment |
|-----------|------:|------------|
| Problem Fidelity | 8 | The proposal directly attacks the evidence-necessity bottleneck. Good. |
| Method Specificity | 6 | Classifier architecture is clear, but training data pipeline has gaps (see below). |
| Contribution Quality | 5 | The dominant contribution (classifier + reranking) is incremental — fine-tuning RoBERTa on LLM-generated labels is standard practice. Novelty needs sharpening. |
| Frontier Leverage | 5 | Misses the opportunity to use LLMs more naturally. The evidence audit uses LLM→labels→classifier pipeline, but distillation or direct LLM-as-judge calibration may be more elegant. |
| Feasibility | 7 | Feasible within constraints but Hard class size (~330) is risky for fine-tuning. |
| Validation Focus | 7 | Three claims are clean and minimal. Claim 3 may not add much. |
| Venue Readiness | 6 | For CCF C, contribution needs to be clearer. The current framing sounds like "we trained a classifier on LLM labels" which is thin. |

**OVERALL: 5.9** (weighted)

## Weaknesses and Fixes

### 1. Contribution Quality (score 5) — CRITICAL

**Weakness:** "Train a RoBERTa classifier on LLM-generated difficulty labels" is a well-known distillation pattern. The evidence-necessity difficulty definition is the real conceptual contribution, but the proposal buries it as a data construction step rather than highlighting it as the methodological core.

**Fix:** Reframe the dominant contribution as the **evidence-necessity difficulty framework** itself — the operationalized definition (Easy=1 sent, Medium=2, Hard=3+ with bridge removal test) plus the automated audit pipeline that produces these labels. The classifier is the instantiation, not the idea. The paper story should be: (1) we define difficulty by evidence necessity, (2) we build an automated audit to label existing QA data, (3) we train a classifier and use it for evaluate-and-rerank. The novelty is in the difficulty framework, not the classifier architecture.

### 2. Method Specificity (score 6) — IMPORTANT

**Weakness:** Several concrete questions remain unanswered:
- How exactly is `quality_score` computed in the reranking formula `score = P(target_difficulty) * quality_score`? Is it the quality_judge output? A binary pass/fail? A continuous score?
- The training data includes "~2000 explicit Easy" but how are explicit Easy difficulty labels assigned? Just by `ex_or_im == "explicit"`? That conflates question style with difficulty.
- What is the exact input format? FairytaleQA story sections can be 500+ tokens. Does RoBERTa handle the full context, or is truncation applied? At what point?
- Human validation: 200 samples is reasonable for calibration but how will disagreements be resolved? What's the expected inter-annotator agreement?

**Fix:**
- Define quality_score: use quality_judge's `quality_pass` as a gate (binary) and `P(target_difficulty)` for ranking among quality-passing candidates.
- For explicit Easy: use the evidence audit result (`answer_sentence_alone_sufficient=yes` AND `num_required_sentences=1`) rather than the `ex_or_im` metadata field.
- Specify truncation: DeBERTa-v3-base handles 512 tokens. For long contexts, truncate story_section to the evidence-relevant sentences (answer + bridge + anchor ± 1 neighboring sentence) rather than the full section.
- For human annotation: specify that 2 annotators label independently, compute Cohen's kappa, resolve disagreements by discussion.

### 3. Frontier Leverage (score 5) — IMPORTANT

**Weakness:** The proposal uses LLM (Qwen-32B) for evidence audit → labels, then trains a small classifier on those labels. This is a standard distillation pattern that doesn't leverage LLMs deeply. Two missed opportunities:

(a) **LLM confidence calibration:** Instead of training a separate classifier, calibrate the LLM judge directly using the human validation set. Temperature scaling or Platt scaling on LLM logits/confidence could produce a reliable difficulty estimator without training a new model. This would be simpler and more directly address the "unreliable LLM judge" bottleneck.

(b) **Evidence-aware input for the classifier:** The current classifier input is just `(story, question, answer)`. But the evidence audit already identifies which sentences are required. Including sentence-level evidence features (which sentences the model attends to) would make the classifier evidence-grounded rather than surface-grounded.

**Fix:** Consider one of these as the primary mechanism:

*Option A (Simpler):* Keep the classifier but enrich its input with explicit evidence features: `[CLS] story_section [SEP] question [SEP] answer [SEP] evidence_sentence_1 [SEP] evidence_sentence_2 [SEP] ...`. This makes the classifier truly evidence-aware.

*Option B (More elegant):* Replace the classifier with a calibrated LLM judge. Use the human-labeled 200 samples as a calibration set. Apply Platt scaling to convert raw LLM judge confidence into calibrated probabilities. Use these calibrated probabilities for reranking. This approach is simpler (no training), more modern, and directly addresses the "unreliable judge" problem.

For CCF C, Option A is safer and more concrete. Option B is more novel but riskier.

### 4. Venue Readiness (score 6) — MINOR

**Weakness:** The paper framing needs a stronger "so what" — who benefits from difficulty-controlled narrative QG? The proposal assumes the reader cares about generating Easy/Medium/Hard questions from stories, but doesn't articulate the application context (education, adaptive testing, reading comprehension practice).

**Fix:** Add 1-2 sentences in the introduction motivating the application: "Difficulty-controlled QG is essential for adaptive reading comprehension systems used in education. FairytaleQA is explicitly designed for such applications (Xu et al., 2022)."

## Simplification Opportunities

1. **Drop Claim 3 entirely.** The graph-guided "Ours" method is not clearly better than baselines. Rather than trying to rescue it as a contribution, treat it as one of 4 methods being compared. The paper is about the difficulty framework + classifier + reranking, not about the graph.

2. **Merge the evidence audit and classifier training into one contribution block** instead of presenting them as separate stages. The audit IS the difficulty framework; the classifier IS the evaluation mechanism.

3. **Use the same difficulty classifier for both training labels AND final evaluation.** Train on FairytaleQA train, evaluate on validation. Don't need separate "evidence audit labels" and "classifier labels" — they're the same thing projected through the model.

## Modernization Opportunities

1. **Evidence-enriched classifier input** (see Frontier Leverage fix Option A). Use the evidence sentence IDs from the audit to construct a focused input, rather than feeding the full 512-token context and hoping the model figures out which sentences matter.

2. **Multi-task training:** Train the classifier not just for difficulty prediction, but also for evidence sentence identification (auxiliary task). Input: (story, question, answer) → Output: (difficulty label, which sentence IDs are evidence). This would be a stronger contribution than a pure classification head, and the evidence audit data already provides both labels.

3. **LLM-as-judge calibration** (Option B from Frontier Leverage). If feasible within compute constraints, calibrating an LLM judge is more publishable than fine-tuning BERT in 2026.

## Drift Warning

**Minor drift risk:** The proposal's emphasis on the classifier as the "dominant contribution" risks drifting from the original problem (difficulty-controlled QG) toward a difficulty prediction problem. Keep the framing on QG quality — the classifier is a means, not an end.

## Verdict

**REVISE** — The direction is promising, but the contribution needs reframing (evidence-necessity framework as the core, classifier as instantiation), the method specificity needs tightening (input format, quality score, training data), and the frontier leverage should be improved (evidence-enriched input or multi-task). For CCF C, these revisions should be sufficient to make a solid submission.
