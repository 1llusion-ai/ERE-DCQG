# Round 3 Refinement (Final)

## Problem Anchor (Revised Definition)

- **Bottom-line problem:** How to generate reading comprehension questions whose answering difficulty (Easy/Medium/Hard) is reliably controllable, where "difficulty" is defined by a two-dimensional combination of answer explicitness and necessary evidence scope — not by surface features or graph hop count alone.
- **Must-solve bottleneck:** No mechanism verifies evidence necessity during generation; evaluation relies on unreliable LLM judges.
- **Non-goals:** General-purpose QG, SOTA, new LLM, multilingual.
- **Constraints:** Single-GPU, 4-6 weeks, CCF C, <$100 API.
- **Success condition:** Independent difficulty assessment (trained model) correctly predicts target difficulty at significantly higher rate than baselines.

## Anchor Check

- **Original bottleneck:** Structural difficulty ≠ answering difficulty; LLM evaluation unreliable.
- **Still addressed:** Evidence-necessity operationalizes answering difficulty. Multi-task classifier provides reproducible evaluation. Counterfactual verification strengthens labels. Reranking closes the loop.
- **Drift:** None.

## Simplicity Check

- **Dominant contribution:** Multi-task difficulty-evidence classifier (mechanism) + evidence-necessity labels (data). The classifier is interpretable (identifies evidence sentences) and dual-use (evaluator + reranker).
- **No components removed in this round.** Round 2 already achieved tight design. Round 3 is framing improvements only.

## Changes Made

### 1. Interpretability as a concrete feature, not an afterthought (IMPORTANT)

- **Reviewer said:** The evidence head's interpretability angle is underemphasized.
- **Action:** Add explicit paragraph:

> Unlike prior difficulty assessments (LLM judges return a label; IRT returns a score; surface features return statistics), our multi-task classifier provides an **interpretable difficulty assessment**: alongside the Easy/Medium/Hard prediction, it identifies which sentences constitute the required evidence. This interpretability has direct practical value in educational QG, where teachers need to understand WHY a question is classified as Hard to decide whether it is appropriate for their students. The evidence head is not merely an auxiliary training signal — it is a first-class output that makes the difficulty prediction auditable and actionable.

- **Impact:** Framing only. Elevates the evidence head from "auxiliary task" to "interpretable output."

### 2. Explicitly address reranker-evaluator circularity (IMPORTANT)

- **Reviewer said:** Using the same classifier for reranking and evaluation creates a validity concern.
- **Action:** Add methodological note:

> **Evaluation validity.** Reranked questions are selected to maximize P_classifier(target_difficulty), so evaluating them with the same classifier produces an optimistic upper bound. We address this by design:
>
> 1. **Human evaluation is the ground-truth comparison.** We evaluate 100 samples (50 reranked + 50 non-reranked, blind to source) with 2 human annotators. The human evaluation compares reranked vs. non-reranked directly — the classifier is not in the evaluation loop.
> 2. **Classifier evaluation serves as a consistency check.** We report classifier-evaluated accuracy to measure the reranking mechanism's self-consistency, but all claims about improvement are validated against human judgment.
> 3. **Cross-validation guards against overfitting.** The 5-fold CV ensures the classifier wasn't tuned on the evaluation data.
>
> This separation of "optimization instrument" (classifier for reranking) from "ground-truth evaluation" (human + held-out classifier fold) is essential for valid claims.

- **Impact:** Shows methodological maturity. No pipeline change.

### 3. Dual-use as a methodological contribution (IMPORTANT)

- **Reviewer said:** Frame the dual-use more explicitly as a contribution.
- **Action:** Revise contribution statement:

> **Dominant contribution:** A multi-task classifier that jointly predicts question difficulty and identifies evidence sentences — the first difficulty assessment model designed to serve as both an evaluation instrument (replacing unreliable LLM judges) and a generation-time optimization signal (reranking candidates by predicted difficulty). The model provides interpretable assessments: it doesn't just classify difficulty, it explains it by identifying the required evidence.

- **Impact:** Cleaner articulation of why this is more than "fine-tune a classifier."

### 4. LLM-to-small-model pattern as frontier design choice (MINOR)

- **Reviewer said:** Connect the approach to the broader trend.
- **Action:** Add one sentence in related work framing:

> Our approach follows the "LLM-as-annotator → train specialized small model" paradigm that has proven effective across NLP evaluation tasks (e.g., automatic essay scoring, factuality checking). We extend this pattern with counterfactual verification and multi-task training to produce evidence-grounded labels, not just surface-level annotations.

### 5. Difficulty definition revised to evidence scope + answer explicitness (IMPORTANT)

- **Issue found:** The previous definition was too close to pure evidence-sentence count, which made single-sentence implicit reasoning collapse into Easy and made Medium/Hard unnecessarily scarce.
- **Action:** Replace the sentence-count-only mapping with a two-dimensional definition:

| Difficulty | Definition |
| ---------- | ---------- |
| **Easy** | The answer can be directly found in the text; obtaining the answer requires relying on only one necessary evidence sentence. |
| **Medium** | **Case 1:** The answer cannot be directly found in the text; obtaining the answer requires relying on one necessary evidence sentence and making a simple inference. **Case 2:** The answer can be directly found in the text; however, obtaining the answer requires synthesizing information from multiple necessary evidence sentences. |
| **Hard** | The answer cannot be directly found in the text; obtaining the answer requires synthesizing information from multiple necessary evidence sentences and making at least one inference. |

- **Impact:** Aligns the method more naturally with FairytaleQA's explicit/implicit distinction and CrossQG-style difficulty control while keeping evidence necessity auditable.

---

## Final Revised Proposal

# Evidence-Necessity-Aware Multi-Task Classification for Difficulty-Controlled Question Generation

## Problem Anchor

- **Bottom-line problem:** How to generate reading comprehension questions whose answering difficulty (Easy/Medium/Hard) is reliably controllable, where "difficulty" is defined by a two-dimensional combination of answer explicitness and necessary evidence scope — not by surface features or graph hop count alone.
- **Must-solve bottleneck:** No mechanism verifies evidence necessity during generation; evaluation relies on unreliable LLM judges.
- **Non-goals:** General-purpose QG, SOTA, new LLM, multilingual.
- **Constraints:** Single-GPU, 4-6 weeks, CCF C, <$100 API.
- **Success condition:** Independent difficulty assessment (trained model) correctly predicts target difficulty at significantly higher rate than baselines.

## Technical Gap

### Why Current Methods Fail

**Surface and structural difficulty definitions don't guarantee answering difficulty.** Prior DCQG work defines difficulty via surface features (question type, word count, answer length) or structural features (graph hops, path length). These correlate with but do not ensure answering difficulty. Our experiments confirm: 3-hop event paths on MAVEN-ERE produce 0% verified Hard questions; on FairytaleQA, graph-guided QG is statistically indistinguishable from prompt-based baselines (bootstrap p=0.69).

**LLM judges are unreliable evaluators.** All current DCQG evaluation uses LLM judges that are prompt-sensitive (changing the judge prompt shifted Pred Hard from 0% to 24.5%), irreproducible across runs, and show 9-36% error rates.

**IRT-based difficulty requires learner response data.** Methods like CrossQG (2024) and Uto et al. (2023) use Item Response Theory, which requires data on how learners answer questions. This is unavailable for newly generated questions, creating a cold-start problem.

### Why Evidence Necessity Captures Difficulty (Theoretical Grounding)

The connection between evidence-inference load and question difficulty is grounded in established reading comprehension theory:

- **Kintsch's Construction-Integration model (1988, 1998):** Comprehension requires building a textbase from propositions and integrating them into a situation model. Difficulty increases when the reader must integrate propositions from multiple text segments, especially when inference bridges gaps between segments.

- **Coh-Metrix dimensions (McNamara et al., 2014):** Text difficulty correlates with referential cohesion and situation model coherence. Questions requiring more evidence sentences demand more referential tracking and inference.

- **Information Integration Theory (Anderson, 1981):** Cognitive difficulty increases with the number of information sources that must be integrated.

- **FairytaleQA design rationale (Xu et al., 2022):** The dataset distinguishes explicit from implicit questions. Implicit questions are empirically harder because the answer cannot be directly found in text and must be inferred from evidence.

**Evidence-inference load operationalizes difficulty along two axes** — whether the answer can be directly found in text and whether the reader must rely on one or multiple necessary evidence sentences.

| Difficulty | Definition |
| ---------- | ---------- |
| **Easy** | The answer can be directly found in the text; obtaining the answer requires relying on only one necessary evidence sentence. |
| **Medium** | **Case 1:** The answer cannot be directly found in the text; obtaining the answer requires relying on one necessary evidence sentence and making a simple inference. **Case 2:** The answer can be directly found in the text; however, obtaining the answer requires synthesizing information from multiple necessary evidence sentences. |
| **Hard** | The answer cannot be directly found in the text; obtaining the answer requires synthesizing information from multiple necessary evidence sentences and making at least one inference. |

### Why This Is Timely

As LLM-based evaluation becomes standard but its unreliability is increasingly documented, difficulty-controlled QG urgently needs reproducible assessment. Simultaneously, LLMs as data annotators make it feasible to create evidence-necessity labels at scale — prohibitively expensive with human annotation alone. Our approach follows the "LLM-as-annotator → train specialized small model" paradigm, extending it with counterfactual verification and multi-task training to produce evidence-grounded labels.

## Method Thesis

A multi-task classifier that jointly predicts question difficulty and identifies required evidence sentences — trained on evidence-necessity labels from automated counterfactual-verified audits — provides reproducible, interpretable difficulty evaluation and enables generation-time reranking that improves difficulty control for any QG method.

## Contribution Focus

- **Dominant contribution:** A multi-task classifier that jointly predicts question difficulty and identifies evidence sentences — the first difficulty assessment model designed to serve as both an evaluation instrument (replacing unreliable LLM judges) and a generation-time optimization signal (reranking candidates by predicted difficulty). The model provides interpretable assessments: it doesn't just classify difficulty, it explains it by identifying the required evidence.

- **Supporting contribution:** An automated evidence audit pipeline with counterfactual verification that produces evidence-necessity labels from existing QA datasets without human annotation or learner response data.

- **Explicit non-contributions:**
  - The narrative evidence graph (one of 4 compared QG methods)
  - New QG generation architectures
  - General QG quality improvements

## Proposed Method

### Complexity Budget

- **Frozen / reused:** FairytaleQA dataset, Qwen-32B as generator, 4 existing QG methods, revised difficulty definitions
- **New trainable:** Multi-task DeBERTa-v3-base classifier (difficulty + evidence heads)
- **New pipeline:** Evidence audit with counterfactual verification (one-time offline)
- **Excluded:** RL optimization, multi-model ensemble, curriculum learning, graph-aware features, Longformer

### System Overview

```
Stage 1: Evidence Audit Pipeline (offline, one-time, ~$30 API)
══════════════════════════════════════════════════════════════
  FairytaleQA train QA pairs (2166 implicit + ~2000 explicit)
    ↓
  Stage A: Chain-of-thought evidence analysis (×3 runs)
    Prompt: story with [S0]...[Sn] + answer + difficulty
    → answer_sentence_id, required_evidence_sentences,
      bridge_sentence_ids, ASA, reasoning_operation
    ↓
  Stage B: Counterfactual verification (per bridge/support sentence)
    "If [Sk] is removed, can the answer still be determined?"
    → Confirms necessity: "no"/"partial" = necessary
    ↓
  Stage C: Self-consistency (majority vote across 3 runs)
    → Difficulty label: majority of 3 runs
    → Evidence sentences: intersection (≥2/3 agreement)
    → No majority → excluded (~10%)
    ↓
  Label mapping:
    Easy:   answer directly found + one necessary evidence sentence
    Medium: single-sentence simple inference OR direct answer requiring multiple necessary evidence sentences
    Hard:   answer not directly found + multiple necessary evidence sentences
    ↓
  Human validation: 200 samples, 2 annotators, Cohen's κ
  Coverage report: % with valid consistent labels

Stage 2: Multi-Task Classifier Training (~20 GPU-hours with 5-fold CV)
══════════════════════════════════════════════════════════════════════
  Input: [CLS] [S0] sent_0 [S1] sent_1 ... [Sn] sent_n [SEP] question [SEP] answer [SEP]
  
  Tokenization:
    31 special tokens [S0]...[S30] added to DeBERTa vocabulary
    model.resize_token_embeddings(len(tokenizer))
    marker_positions tensor records [Si] token positions
    Evidence-focused truncation: answer ± 2 + bridge + anchor (max 512)
  
  Architecture (DeBERTa-v3-base, 184M params):
    Head 1 (difficulty): h_CLS → MLP(768→256→3) → Softmax
    Head 2 (evidence): h_{[Si]} → Linear(768→2) per sentence marker
  
  Loss: L = L_diff(CE, class-weighted) + 0.3 × L_evidence(BCE)
  
  Data: class counts re-estimated after the revised audit definition
    Augmentation: LLM paraphrasing constrained to same evidence sentences, especially if Hard remains small
    Split: 70/15/15 | Class weights: inverse frequency | 5-fold CV
  
  Recipe: AdamW lr=2e-5 (backbone) / 5e-4 (heads), batch 16, 10 epochs
    Early stop on dev macro F1, warmup 10%

Stage 3: Question Generation + Reranking (~$40 API)
═══════════════════════════════════════════════════
  For each (story, answer, target_difficulty):
    → Generate K=5 candidates (any QG method)
    → Score: s = is_quality_pass × P_classifier(target_difficulty)
    → Select: argmax(s)
    → Output: 1 difficulty-controlled question

Stage 4: Evaluation (designed to avoid reranker-evaluator circularity)
════════════════════════════════════════════════════════════════════════
  Primary:     Human evaluation (100 samples: 50 reranked + 50 non-reranked, blind)
  Consistency: Trained classifier on held-out fold (measures self-consistency)
  Diagnostic:  LLM judge (GPT-4o-mini, Qwen-32B) for method comparison
  Statistics:  Bootstrap 95% CI, paired tests, McNemar's test
```

### Core Mechanism: Multi-Task Difficulty-Evidence Classifier

**Input example:**
```
[CLS] [S0] The girl lived with her grandmother. [S1] One day she went 
to the forest. [S2] A wolf wanted to eat her. [SEP] Why did the girl 
go to the forest? [SEP] to visit her grandmother [SEP]
```

**Output:**
- Difficulty: `P(Easy)=0.12, P(Medium)=0.75, P(Hard)=0.13` → Medium
- Evidence: `[S1]=required` → the question needs sentence [S1]

**Why multi-task matters:** The evidence head forces the encoder to learn which sentences are load-bearing, creating representations that are evidence-aware rather than surface-pattern-matching. The difficulty head benefits from these enriched representations. Ablation (Claim 1 experiment) directly tests this.

**Why interpretable difficulty assessment matters:** Unlike prior difficulty assessments — LLM judges that return a label, IRT models that return a score, surface features that return statistics — this classifier provides an **interpretable assessment**: alongside the Easy/Medium/Hard prediction, it identifies which sentences constitute the required evidence. This interpretability has direct practical value in educational QG, where teachers need to understand WHY a question is classified as Hard to decide whether it is appropriate for their students. The evidence head is not merely an auxiliary training signal — it is a first-class output that makes the difficulty prediction auditable and actionable.

**Why dual-use is a methodological contribution:** The same model architecture serves two roles:
1. **Evaluation instrument:** Replaces unreliable LLM judges with a fast (~10ms/item), reproducible, interpretable classifier
2. **Generation-time reranker:** Selects the candidate question that best matches the target difficulty from a pool of K=5

This dual-use is possible because the classifier produces calibrated probability distributions over difficulty levels — the same probabilities that make it a good evaluator also make it a natural reranking signal. No prior DCQG work uses the same model for both evaluation and optimization.

### Evidence Audit Pipeline (Data Construction)

**Stage A: Chain-of-thought evidence analysis.** Uses `build_answer_grounded_evidence_prompt()` from `dcqg/path/answer_grounded_evidence.py`. The prompt presents the story with numbered sentences, the target answer (NOT the original question), and asks the model to identify: answer_sentence_id, required_evidence_sentences, bridge_sentence_ids, anchor_sentence_ids, answer_sentence_alone_sufficient, bridge_required, reasoning_operation, necessity_type, evidence_plan_valid, target_difficulty_feasible. Validated by `validate_evidence_plan()`.

**Stage B: Counterfactual verification.** For each sentence [Sk] in required_evidence_sentences (excluding answer sentence):
```
Given: question="{question}", answer="{answer}"
Story: [S0]...[Sn]
Suppose sentence [Sk] is NOT available. Can you still determine the answer?
Reply: "yes" / "partial" / "no"
```
Sentence is confirmed necessary if response is "no" or "partial."

**Stage C: Self-consistency.** 3 runs per item. Majority vote for difficulty label. Intersection (≥2/3) for evidence sentences. No majority → excluded.

**Label mapping:**

| Label | Condition | Theoretical basis |
|-------|-----------|-------------------|
| Easy | Answer directly found AND \|req\|=1 | Single-sentence explicit answer extraction |
| Medium | Answer not directly found AND \|req\|=1 with simple inference; OR answer directly found AND \|req\|≥2 | Single-sentence implicit reasoning or simple multi-sentence synthesis |
| Hard | Answer not directly found AND \|req\|≥2 | Multi-evidence integration with implicit answer acquisition |

**Expected yield:** to be re-estimated after re-labeling under the revised two-dimensional definition. Medium should increase because single-sentence implicit reasoning no longer collapses into Easy.

### QG Methods Compared

| Method | Description | Graph? |
|--------|------------|--------|
| Direct | story + answer + difficulty definition | No |
| ICL | + few-shot examples | No |
| SelfRefine | Direct → critique → revise | No |
| Ours | + evidence graph scaffold | Yes |

All methods: same Qwen-32B API, same difficulty definitions, same K=5, same reranking.

### Evaluation Validity

**Reranker-evaluator circularity.** Reranked questions are selected to maximize P_classifier(target_difficulty), so evaluating them with the same classifier produces an optimistic upper bound. We address this:

1. **Human evaluation is the ground-truth.** 100 samples (50 reranked + 50 non-reranked, blind to source), 2 annotators. Claims about improvement are validated against human judgment, not the classifier.
2. **Classifier evaluation measures self-consistency.** Reported for completeness and cross-method comparison, but not as the primary evidence for reranking effectiveness.
3. **Cross-validation.** 5-fold CV ensures the classifier fold evaluating a reranked question was not trained on that question's data.

### Failure Modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Hard class too small | Per-class F1 <0.50 | Report as limitation; Easy+Medium cover >80% of QG demand |
| Overfitting to LLM labels | Classifier-human κ << Classifier-LLM κ | Human validation calibrates; report both |
| Augmented data noisy | Augmented dev F1 < raw dev F1 | Tighten validation or reduce augmentation |
| Audit coverage <70% | Coverage report | Relax consistency threshold |

### Novelty Argument

| Work | Difficulty Def | Evaluation | Evidence ID? | Dual-Use? | Interpretable? |
|------|---------------|------------|-------------|-----------|----------------|
| CrossQG (2024) | IRT | LLM judge | No | No | No |
| Uto et al. (2023) | IRT | IRT score | No | No | No |
| KAQG (2025) | Bloom's + IRT | LLM judge | No | No | No |
| GNET-QG (2025) | Graph structure | Human | No | No | No |
| **Ours** | **Evidence-inference necessity** | **Trained classifier** | **Yes** | **Yes** | **Yes** |

**What we uniquely contribute:**
1. Evidence-inference necessity as a theoretically grounded difficulty definition (Kintsch, Coh-Metrix, FairytaleQA explicit/implicit distinction)
2. Multi-task classifier: jointly predicts difficulty AND identifies evidence (interpretable)
3. Dual-use design: evaluation instrument + generation-time reranker
4. Counterfactual-verified evidence labels from automated LLM audit (no learner data)

## Claim-Driven Validation

### Claim 1: Multi-task classifier > LLM judges for difficulty assessment

- **Experiment:** Train classifier on audit labels. Evaluate on 200 human-annotated samples.
- **Baselines:** GPT-4o-mini judge, Qwen-32B judge, difficulty-only DeBERTa (ablation), majority baseline
- **Metrics:** Macro F1, per-class F1, Cohen's κ vs. human, evidence recall+precision
- **Statistics:** Bootstrap 95% CI, McNemar's test
- **Expected:** Multi-task macro F1 > LLM judges ≥5pp; evidence recall >70%
- **Ablation:** Multi-task vs. single-task → expected ≥3pp improvement on Hard F1

### Claim 2: Classifier reranking improves difficulty control for all QG methods

- **Experiment:** K=5 candidates, 4 methods × 3 difficulties. Compare K=1 vs. K=5+classifier reranking vs. K=5+LLM reranking.
- **Metrics:** Difficulty accuracy (human-evaluated on 100 samples), classifier-evaluated accuracy (consistency check), Spearman ρ, per-level hit rate
- **Statistics:** Paired bootstrap test per method
- **Expected:** Reranking improves difficulty accuracy ≥5pp; Hard hit rate ≥+10pp

### Pre-registered partial success

If the classifier improves Easy and Medium but not Hard (due to class size), we report this as a partial validation. Easy+Medium account for >80% of educational QG demand. The Hard limitation motivates future work on evidence augmentation.

## Experiment Handoff

- **Must-prove:** Classifier > LLM judges; reranking improves all methods
- **Must-run ablations:** Multi-task vs. single-task, K=1 vs. K=5, classifier vs. LLM reranking
- **Critical data:** FairytaleQA audit + 200 human annotations
- **Highest risk:** Hard class size, label quality, domain shift (gold → generated questions)

## Compute & Timeline

| Component | Cost | Time |
|-----------|------|------|
| Evidence audit (3× + counterfactual) | ~$30 | 2 days |
| Hard augmentation | ~$10 | 1 day |
| Classifier training (5-fold CV) | ~20 GPU-hours | 1 day |
| Human annotation (200 × 2) | ~20 person-hours | 3 days |
| Generation (K=5, 4 methods) | ~$40 | 2 days |
| Evaluation + analysis | — | 3 days |
| Writing | — | 1 week |
| **Total** | **<$100 API, 20 GPU-hours** | **4 weeks** |
