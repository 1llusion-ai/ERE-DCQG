# Round 2 Refinement

## Problem Anchor (Unchanged)

- **Bottom-line problem:** How to generate reading comprehension questions whose answering difficulty (Easy/Medium/Hard) is reliably controllable, where "difficulty" is defined by the minimum evidence sentences a reader must consult — not by surface features or graph hop count alone.
- **Must-solve bottleneck:** No mechanism verifies evidence necessity during generation; evaluation relies on unreliable LLM judges.
- **Non-goals:** General-purpose QG, SOTA, new LLM, multilingual.
- **Constraints:** Single-GPU, 4-6 weeks, CCF C, <$100 API.
- **Success condition:** Independent difficulty assessment (trained model) correctly predicts target difficulty at significantly higher rate than baselines.

## Anchor Check

- **Original bottleneck:** Structural difficulty ≠ answering difficulty; LLM evaluation is unreliable.
- **Why the revised method still addresses it:** Evidence-necessity directly operationalizes answering difficulty as integration load. The multi-task classifier makes assessment reproducible and evidence-grounded. Counterfactual verification strengthens training labels.
- **Reviewer suggestions rejected as drift:** None.

## Simplicity Check

- **Dominant contribution after revision:** Multi-task difficulty classifier trained on evidence-necessity labels, jointly predicting difficulty and identifying evidence sentences, usable as evaluator and reranker.
- **Components removed or merged:**
  - "ENDF framework" umbrella de-emphasized — lead with the mechanism (multi-task classifier + evidence-grounded labels)
  - Longformer fallback dropped (FairytaleQA sections fit 512 tokens)
  - Quality gate merged into reranking score (binary × P(difficulty))
  - Back-translation augmentation replaced with controlled LLM paraphrasing
- **Why the remaining mechanism is the smallest adequate route:** One classifier, one training pipeline, one reranking mechanism. No new generation architecture.

## Changes Made

### 1. Concrete evidence audit procedure (CRITICAL)

- **Reviewer said:** The audit pipeline is a black box — no prompt, no self-consistency protocol, no label mapping rules.
- **Action:** Specify the full audit procedure as a 3-stage pipeline:
  - Stage A: Chain-of-thought evidence analysis (existing `build_answer_grounded_evidence_prompt` in `dcqg/path/answer_grounded_evidence.py`). The prompt presents the story with numbered sentences [S0]...[Sn], the target answer, and the target difficulty. The LLM returns: answer_sentence_id, required_evidence_sentences, bridge_sentence_ids, answer_sentence_alone_sufficient, bridge_required, reasoning_operation, necessity_type, target_difficulty_feasible.
  - Stage B: Counterfactual verification. For each bridge/support sentence in the plan, ask: "If sentence [Sk] is removed from the available evidence, can the question still be answered?" This validates evidence necessity.
  - Stage C: Self-consistency. Run Stage A+B three times per item. Majority vote for difficulty label. For evidence sentences, take the intersection (conservative: only sentences flagged as required in ≥2/3 runs).
  - Label mapping:
    - Easy: ASA=yes AND |required_evidence|=1 (answer is locally extractable)
    - Medium: |required_evidence|=2 AND at least one bridge/support sentence has bridge_removal_effect ∈ {helpful, somewhat_helpful}
    - Hard: |required_evidence|≥3 AND at least one bridge sentence has bridge_removal_effect ∈ {ambiguous, unanswerable}
  - Coverage: Report the fraction of FairytaleQA pairs that receive a consistent, valid label across 3 runs. Items with no majority are excluded from training (expected: <10% exclusion).
- **Reasoning:** The audit is the foundation of all training data. Making it concrete, reproducible, and self-consistent is essential.
- **Impact:** Adds 2× more API calls to the audit (3 runs + counterfactual), but this is a one-time offline cost (~$15-20 extra). No architectural change.

### 2. Theoretical grounding for evidence-necessity (CRITICAL)

- **Reviewer said:** "Why is sentence count the right proxy for difficulty?" needs a theoretical answer.
- **Action:** Add a theoretical grounding section connecting evidence necessity to established reading comprehension theory:
  - **Kintsch's Construction-Integration model (1988, 1998):** Reading comprehension requires building a textbase from propositions and integrating them into a situation model. Difficulty increases when the reader must integrate propositions from more text segments, especially when the connection between segments requires inference. Evidence-necessity count directly operationalizes integration load.
  - **Coh-Metrix dimensions (McNamara et al., 2014):** Text difficulty correlates with referential cohesion (how many co-references link sentences) and situation model coherence (how much inference bridges gaps). When more sentences are required evidence, the question demands more referential tracking and inference.
  - **FairytaleQA design (Xu et al., 2022):** The dataset distinguishes explicit (answer stated in text) from implicit (answer requires inference) questions. Implicit questions are empirically harder because they require integrating information across sentences — exactly the evidence-necessity construct.
  - **Information Integration Theory (Anderson, 1981):** Cognitive difficulty of a judgment task increases with the number of information sources that must be integrated. Evidence-necessity count is a direct proxy for information integration load.
- **Reasoning:** This transforms the definition from an arbitrary metric into a theoretically grounded construct. Reviewers can see why sentence count approximates cognitive difficulty.
- **Impact:** Framing only. No pipeline change.

### 3. [Sn] marker tokenization and evidence head mechanics (IMPORTANT)

- **Reviewer said:** How the markers are added to DeBERTa and how the evidence head locates them is unclear.
- **Action:** Specify:
  - Add `[S0]` through `[S30]` (31 markers) as additional special tokens via `tokenizer.add_special_tokens({"additional_special_tokens": [f"[S{i}]" for i in range(31)]})`
  - Resize model embeddings: `model.resize_token_embeddings(len(tokenizer))`
  - During tokenization, record the token positions of each `[Si]` marker in a `marker_positions` tensor
  - Evidence head: at inference, extract the hidden states at `marker_positions`, apply `Linear(768, 2)` for binary classification (required evidence or not) per sentence
  - Stories with >30 sentences: concatenate remaining text under `[S30]` (rare in FairytaleQA — median section is 8-12 sentences)
- **Impact:** Implementation detail. No architectural change.

### 4. Controlled augmentation for Hard class (IMPORTANT)

- **Reviewer said:** Back-translation is unreliable for preserving difficulty labels.
- **Action:** Replace with controlled LLM paraphrasing:
  - For each Hard QA pair, prompt the LLM: "Rephrase this question while ensuring it still requires reading sentences [S2], [S5], [S8] to answer. Do not change the answer or the required evidence."
  - Validate: run the evidence audit on the paraphrased question. Keep only if the paraphrased version receives the same difficulty label and ≥80% overlap in required_evidence_sentences.
  - Target: 2× Hard examples (330 → 660). If validation rejects >50%, reduce target to 1.5× and report.
- **Reasoning:** Constrained paraphrasing with evidence-preservation check is more reliable than blind back-translation.
- **Impact:** More API calls for augmentation (~$5-10 extra), but higher-quality augmented data.

### 5. Mechanism-first framing (IMPORTANT)

- **Reviewer said:** "Framework" umbrella risks "this is just labeling + fine-tuning" dismissal. Lead with the mechanism.
- **Action:** Restructure the paper narrative:
  - **Title:** "Evidence-Necessity-Aware Multi-Task Classification for Difficulty-Controlled Question Generation" (mechanism-first)
  - **Dominant contribution:** A multi-task classifier that jointly predicts question difficulty and identifies required evidence sentences, trained on evidence-necessity labels derived from automated evidence audits of narrative QA data, applicable as both evaluator and generation-time reranker.
  - The evidence-necessity definition and audit pipeline are presented as the principled data construction method, not a separate contribution.
  - **Why this mechanism is novel:** Prior difficulty classifiers use surface features or learner-response labels. Ours is the first to jointly predict difficulty AND identify evidence sentences, creating an interpretable, evidence-grounded assessment.
- **Impact:** Framing change. The pipeline is the same; the narrative is sharper.

### 6. Counterfactual verification in audit (IMPORTANT)

- **Reviewer said:** A single LLM call per item is fragile. Add counterfactual verification.
- **Action:** Already incorporated in Change #1 (Stage B of audit). The counterfactual prompt:
  ```
  Given the question: "{question}"
  And the answer: "{answer}"
  
  The full story section is:
  [S0] ... [S1] ... [S2] ... [Sn] ...
  
  Now, suppose sentence [Sk] is NOT available (you cannot read it).
  With only the remaining sentences, can you still determine that 
  the answer is "{answer}"?
  
  Reply: "yes" (answerable without [Sk]), "partial" (hints remain), 
  or "no" (cannot determine the answer without [Sk]).
  ```
  Run for each sentence in `required_evidence_sentences` except the answer sentence.
  A sentence is confirmed as necessary evidence if the counterfactual returns "no" or "partial."
- **Impact:** Strengthens training labels. 1 extra LLM call per required evidence sentence per item.

### 7. Hard class size mitigation (IMPORTANT)

- **Reviewer said:** 66 Hard test examples give wide confidence intervals.
- **Action:**
  - Use 70/15/15 split instead of 80/10/10. This gives ~100 Hard test examples.
  - Report bootstrap 95% CI (1000 resamples) for all per-class F1 scores.
  - Run 5-fold stratified cross-validation for the classifier, report mean ± std macro F1.
  - Pre-register: if Hard test F1 < 0.50 even with augmentation, report this as a limitation and analyze whether it's due to class size or label noise.
- **Impact:** Better statistical reliability, no architectural change.

### 8. "Why now" narrative (MINOR)

- **Action:** Add paragraph: "As LLM-based evaluation becomes standard but its unreliability is increasingly documented, difficulty-controlled QG urgently needs reproducible assessment. Simultaneously, the availability of LLMs as data annotators makes it feasible to create evidence-necessity labels at scale — something that would have required prohibitive human annotation before the foundation-model era. Our approach exploits both trends: LLMs for scalable evidence analysis, a trained classifier for reproducible evaluation."

### 9. Partial success pre-registration (MINOR)

- **Action:** Add: "If the classifier improves difficulty prediction for Easy and Medium but not Hard (possible given the smaller Hard training set), we report this as a partial validation. The framework remains useful: Easy/Medium together account for >80% of educational QG demand, and the Hard limitation motivates future work on evidence augmentation."

### 10. Drop Longformer fallback (MINOR)

- **Action:** Removed. FairytaleQA story sections are typically 5-15 sentences (100-300 tokens). With evidence-focused truncation, 512 tokens is sufficient.

---

## Revised Proposal (Full)

# Evidence-Necessity-Aware Multi-Task Classification for Difficulty-Controlled Question Generation

## Problem Anchor
[Verbatim — unchanged from Round 0]

- **Bottom-line problem:** How to generate reading comprehension questions whose answering difficulty (Easy/Medium/Hard) is reliably controllable, where "difficulty" is defined by the minimum evidence sentences a reader must consult — not by surface features or graph hop count alone.
- **Must-solve bottleneck:** No mechanism verifies evidence necessity during generation; evaluation relies on unreliable LLM judges.
- **Non-goals:** General-purpose QG, SOTA, new LLM, multilingual.
- **Constraints:** Single-GPU, 4-6 weeks, CCF C, <$100 API.
- **Success condition:** Independent difficulty assessment (trained model) correctly predicts target difficulty at significantly higher rate than baselines.

## Technical Gap

### Why Current Methods Fail

**Surface and structural difficulty definitions don't guarantee answering difficulty.** Prior DCQG work defines difficulty via surface features (question type, word count, answer length) or structural features (graph hops, path length). These correlate with but do not ensure answering difficulty. Our experiments confirm: 3-hop event paths on MAVEN-ERE produce 0% verified Hard questions; on FairytaleQA, graph-guided QG is statistically indistinguishable from prompt-based baselines (bootstrap p=0.69).

**LLM judges are unreliable evaluators.** All current DCQG evaluation uses LLM judges that are prompt-sensitive (changing the judge prompt shifted Pred Hard from 0% to 24.5%), irreproducible across runs, and show 9-36% error rates across methods.

### Theoretical Grounding: Why Evidence Necessity Captures Difficulty

The connection between evidence sentence count and question difficulty is grounded in established reading comprehension theory:

- **Kintsch's Construction-Integration model (1988, 1998):** Comprehension requires building a textbase from propositions and integrating them into a situation model. Difficulty increases when the reader must integrate propositions from multiple text segments, especially when inference bridges gaps between segments.

- **Coh-Metrix dimensions (McNamara et al., 2014):** Text difficulty correlates with referential cohesion (co-reference chains across sentences) and situation model coherence (inference required to connect events). Questions requiring more evidence sentences demand more referential tracking and inference.

- **Information Integration Theory (Anderson, 1981):** Cognitive difficulty of a judgment task increases with the number of information sources that must be integrated.

- **FairytaleQA design rationale (Xu et al., 2022):** The dataset distinguishes explicit (answer stated) from implicit (answer requires inference) questions. Implicit questions are empirically harder because they require integrating information across sentences.

**Evidence-necessity count operationalizes integration load** — a reader-centric difficulty metric that directly measures how many text segments must be connected to answer the question.

### Why Naive Fixes Are Insufficient

- **More graph hops:** Adding hops adds structural complexity but doesn't ensure each hop is necessary for answering.
- **Better prompts:** Our project iterated 5+ Hard prompt strategies with diminishing returns.
- **Larger models:** Architecture doesn't change if there's no mechanism to verify evidence necessity.
- **IRT-based difficulty (CrossQG, Uto et al.):** Requires learner response data, which is unavailable for newly generated questions.

### Smallest Adequate Intervention

A **multi-task classifier** that jointly predicts difficulty and identifies evidence sentences, trained on **evidence-necessity labels** from an automated audit with counterfactual verification. This classifier serves as both a reliable evaluator and a generation-time reranker.

### Why This Is Timely

As LLM-based evaluation becomes standard but its unreliability is increasingly documented, difficulty-controlled QG urgently needs reproducible assessment. Simultaneously, LLMs as data annotators make it feasible to create evidence-necessity labels at scale — something that would have required prohibitive human annotation before the foundation-model era. Our approach exploits both trends: LLMs for scalable evidence analysis, a trained classifier for reproducible evaluation.

## Method Thesis

**One-sentence thesis:** A multi-task classifier that jointly predicts question difficulty and identifies required evidence sentences — trained on evidence-necessity labels from automated counterfactual-verified audits — provides reproducible difficulty evaluation and enables generation-time reranking that improves difficulty control for any QG method.

## Contribution Focus

- **Dominant contribution:** A multi-task difficulty-and-evidence classifier trained on evidence-necessity labels, the first to jointly predict difficulty AND identify which sentences constitute the required evidence, applicable as evaluator and reranker.

- **Supporting contribution:** An automated evidence audit pipeline with counterfactual verification that produces evidence-necessity labels from existing QA datasets without human annotation.

- **Explicit non-contributions:**
  - The narrative evidence graph (one of 4 compared QG methods, not a contribution)
  - New QG generation architectures
  - General QG quality improvements

## Proposed Method

### Complexity Budget

- **Frozen / reused:** FairytaleQA dataset, Qwen-32B as generator, existing 4 QG methods (Direct/ICL/SelfRefine/Ours), existing difficulty definitions
- **New trainable components:** (1) Multi-task DeBERTa-v3-base classifier with difficulty + evidence heads
- **New pipeline components:** (1) Evidence audit with counterfactual verification (one-time, offline)
- **Tempting additions intentionally excluded:** RL-based difficulty optimization, multi-model ensemble, curriculum learning, graph-aware classifier features, Longformer

### System Overview

```
Stage 1: Evidence Audit Pipeline (offline, one-time)
═══════════════════════════════════════════════════
  FairytaleQA train QA pairs (2166 implicit + ~2000 explicit)
    ↓
  Stage A: Chain-of-thought evidence analysis (×3 runs)
    Prompt: story with [S0]...[Sn] + answer + difficulty
    → answer_sentence_id, required_evidence_sentences,
      bridge_sentence_ids, ASA, reasoning_operation
    ↓
  Stage B: Counterfactual verification (per bridge/support sentence)
    "If [Sk] is removed, can you still determine the answer?"
    → Confirms each sentence's necessity (yes/partial/no)
    ↓
  Stage C: Self-consistency aggregation (majority vote)
    → Difficulty label (Easy/Medium/Hard)
    → Required evidence sentence IDs (intersection of ≥2/3 runs)
    → Items with no majority → excluded (<10% expected)
    ↓
  Label mapping:
    Easy:   ASA=yes AND |required_evidence|=1
    Medium: |required_evidence|=2
    Hard:   |required_evidence|≥3 AND ≥1 counterfactual="no"
    ↓
  + Human validation: 200 samples, 2 annotators, κ reported
  + Coverage report: fraction with valid consistent labels

Stage 2: Multi-Task Classifier Training
════════════════════════════════════════
  Input:  [CLS] [S0] sent_0 [S1] sent_1 ... [Sn] sent_n [SEP] question [SEP] answer [SEP]
  
  Tokenization:
    - Add [S0]...[S30] as special tokens to DeBERTa tokenizer
    - Resize embeddings: model.resize_token_embeddings(len(tokenizer))
    - Record marker_positions tensor during encoding
    - Evidence-focused truncation: answer sent ± 2 + bridge + anchor. Max 512 tokens.
  
  Architecture (DeBERTa-v3-base, 184M params):
    - Head 1 (difficulty): h_CLS → Linear(768,256) → ReLU → Linear(256,3) → Softmax
    - Head 2 (evidence):   h_{[Si]} → Linear(768,2) → Softmax (per marker position)
  
  Loss: L = L_difficulty(CE, class-weighted) + 0.3 × L_evidence(BCE, per-sentence)
  
  Training data:
    - Easy ~2000 | Medium ~1300 | Hard ~330 raw → ~550 with controlled paraphrasing
    - Augmentation: LLM generates alternative question phrasings constrained to 
      require the same evidence sentences; validated by re-running audit
    - Split: 70/15/15 (train/dev/test) for larger test set on Hard
    - Class weights: inverse frequency (Hard ≈ 5× Easy)
  
  Recipe:
    - AdamW, lr=2e-5 (backbone), lr=5e-4 (heads)
    - Warmup 10%, batch 16, epochs 10, early stop on dev macro F1
    - 5-fold stratified CV, report mean ± std

Stage 3: Question Generation + Reranking
═════════════════════════════════════════
  For each (story, answer, target_difficulty):
    → Generate K=5 candidates (same QG method, same prompt)
    → Score each: s = is_quality_pass × P_classifier(target_difficulty)
      where is_quality_pass = answerable ∧ asks_expected_answer ∧ fluent
    → Select: argmax(s)
    → Output: 1 difficulty-controlled question

Stage 4: Evaluation
═══════════════════
  Primary:     Trained classifier (reproducible, ~10ms/item)
  Secondary:   Human evaluation (100 samples, 2 annotators)
  Diagnostic:  LLM judge (GPT-4o-mini, Qwen-32B) for comparison
  Statistics:  Bootstrap 95% CI (1000 resamples), paired tests
```

### Core Mechanism: Multi-Task Difficulty-Evidence Classifier

**Input format:**
```
[CLS] [S0] The little girl lived with her grandmother. [S1] One day she 
went into the forest. [S2] She met a wolf who wanted to eat her. [SEP] 
Why did the girl go into the forest? [SEP] to visit her grandmother [SEP]
```

**Output:**
- `P(Easy)=0.12, P(Medium)=0.75, P(Hard)=0.13` → predicted difficulty: Medium
- `evidence_mask = [0, 1, 0, ...]` → sentence [S1] is required evidence

**Architecture detail:**
- DeBERTa-v3-base with disentangled attention (184M params)
- 31 additional special tokens: `[S0]` through `[S30]`
- Difficulty head: `[CLS]` hidden state → 2-layer MLP (768→256→3)
- Evidence head: hidden states at `[Si]` positions → Linear(768→2) per sentence
- Joint training with shared encoder; difficulty provides global signal, evidence provides local signal

**Why multi-task helps:** The evidence head forces the encoder to learn which sentences are load-bearing, creating representations that are evidence-aware rather than surface-pattern-matching. The difficulty head benefits from these enriched representations.

**Why this is the main novelty:**
1. First classifier to jointly predict difficulty AND identify evidence sentences for QG
2. Evidence-necessity labels come from counterfactual-verified LLM audits — not from human annotation or learner response data
3. The same classifier serves dual roles: evaluation (replacing LLM judges) and reranking (improving generation)

### Evidence Audit Pipeline (Data Construction)

**Stage A: Chain-of-thought evidence analysis.** Uses the existing `build_answer_grounded_evidence_prompt()` function in `dcqg/path/answer_grounded_evidence.py`. The prompt:
- Presents the story section with numbered sentences `[S0]...[Sn]`
- Provides the target answer (NOT the original question — answer-grounded, not question-dependent)
- Asks the model to identify: answer_sentence_id, required_evidence_sentences, bridge_sentence_ids, anchor_sentence_ids, answer_sentence_alone_sufficient, bridge_required, reasoning_operation, necessity_type, evidence_plan_valid, target_difficulty_feasible
- Validated by `validate_evidence_plan()` which checks index bounds, answer-in-evidence constraint, and difficulty consistency

**Stage B: Counterfactual verification.** For each sentence `[Sk]` in `required_evidence_sentences` (excluding the answer sentence):
- Prompt: "If sentence [Sk] is removed, can you still determine that the answer is '{answer}'? Reply: yes/partial/no."
- A sentence confirmed as necessary evidence if response is "no" or "partial"
- This directly validates evidence NECESSITY, not just evidence relevance

**Stage C: Self-consistency.** Run Stage A+B three times per item:
- Difficulty label: majority vote across 3 runs
- Required evidence sentences: intersection (sentence must appear in ≥2/3 runs)
- Items with no majority: excluded from training, reported in coverage stats

**Label mapping rules:**
| Label | Condition | Intuition |
|-------|-----------|-----------|
| Easy | ASA=yes AND \|req\|=1 | Answer is locally extractable from one sentence |
| Medium | \|req\|=2 | One supporting relation needed |
| Hard | \|req\|≥3 AND ≥1 counterfactual="no" | Multi-sentence integration required, verified |

**Expected yield:** ~3500 labeled examples from ~4166 FairytaleQA train pairs (~85% coverage, ~15% excluded due to inconsistency or parse failure).

### QG Methods Compared

| Method | Input | Graph? | New? |
|--------|-------|--------|------|
| Direct | story + answer + difficulty definition | No | No |
| ICL | story + answer + difficulty + few-shot examples | No | No |
| SelfRefine | Direct → critique → revise | No | No |
| Ours | story + answer + difficulty + evidence graph | Yes | No |

All methods: same Qwen-32B API, same ENDF-derived difficulty definitions, same K=5 generation, same reranking. The comparison tests whether reranking universally helps regardless of QG method.

### Failure Modes and Diagnostics

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| Hard class too small (<550 after augmentation) | Per-class F1 < 0.50 on dev | Report as limitation; Easy+Medium still valuable (>80% of educational QG) |
| Classifier overfits to LLM audit labels | Classifier-human κ << Classifier-LLM κ | Human validation of 200 samples calibrates; report both agreements |
| Reranking selects difficulty-matched but low-quality questions | Manual inspection of 50 reranked samples | Quality gate (binary) prevents this; merged into ranking score |
| Augmented Hard data is noisy | Augmented-Hard dev F1 < raw-Hard dev F1 | Reduce augmentation or tighten validation threshold |
| Evidence audit coverage too low (<70%) | Coverage report in Stage C | Relax consistency threshold (2/3 → any-agreement) for borderline cases |

### Novelty Argument

**Closest work and exact differences:**

| Work | Difficulty Definition | Evaluation | Evidence? | Reranking? |
|------|----------------------|------------|-----------|------------|
| CrossQG (2024) | IRT (learner response) | LLM judge | No | No |
| Uto et al. (2023) | IRT (learner response) | IRT score | No | No |
| KAQG (2025) | Bloom's taxonomy + IRT | LLM judge | No | No |
| GNET-QG (2025) | Graph structure | Human only | No | No |
| **Ours** | **Evidence necessity** | **Trained classifier** | **Yes (joint)** | **Yes** |

**What we add that no prior work has:**
1. Evidence-necessity as a theoretically grounded, operationalizable difficulty definition
2. Multi-task classifier that predicts difficulty AND identifies evidence (interpretable)
3. Dual-use classifier: evaluator + reranker
4. Counterfactual-verified evidence labels from automated LLM audits (no learner data needed)

## Claim-Driven Validation Sketch

### Claim 1: The multi-task classifier provides more reliable and evidence-grounded difficulty assessment than LLM judges

- **Experiment:** Train classifier on FairytaleQA train evidence audit labels. Evaluate on human-annotated 200-sample test set.
- **Baselines:**
  - GPT-4o-mini difficulty judge (zero-shot)
  - Qwen-32B difficulty judge (zero-shot)
  - DeBERTa difficulty-only classifier (no evidence task — ablation)
  - Majority class baseline
- **Metrics:** Macro F1, per-class F1, Cohen's κ vs. human, evidence sentence recall + precision (multi-task only)
- **Statistics:** Bootstrap 95% CI, McNemar's test for classifier vs. LLM judge
- **Expected result:** Multi-task macro F1 > LLM judges by ≥5pp; multi-task κ > LLM κ; evidence recall >70%

### Claim 2: Classifier-based reranking improves difficulty control for all QG methods

- **Experiment:** Generate K=5 candidates for all 4 methods × 3 difficulties. Compare:
  - K=1 (no reranking)
  - K=5 + reranking by multi-task classifier
  - K=5 + reranking by LLM judge
- **Metrics:** Difficulty accuracy (classifier-evaluated), macro accuracy, Spearman ρ, per-level hit rate
- **Statistics:** Paired bootstrap test (reranked vs. non-reranked, per method)
- **Expected result:** Classifier reranking improves difficulty accuracy by ≥5pp for all methods; Hard hit rate ≥+10pp; classifier reranking ≥ LLM reranking

### Ablation: Multi-task vs. single-task

- **Experiment:** Multi-task (difficulty + evidence) vs. difficulty-only DeBERTa
- **Metric:** Macro F1, per-class F1, evidence sentence recall
- **Expected result:** Multi-task improves Hard F1 by ≥3pp (evidence grounding helps most for Hard)

### Pre-registered partial success

If the classifier improves difficulty prediction for Easy and Medium but not Hard (possible given Hard class size ~550), we report this as a partial validation. Easy+Medium account for >80% of educational QG demand in the FairytaleQA setting. The Hard limitation would motivate future work on evidence augmentation and is itself a useful finding about the minimum data requirements for evidence-necessity training.

## Experiment Handoff Inputs

- **Must-prove claims:** Classifier > LLM judges (Claim 1); reranking improves all methods (Claim 2)
- **Must-run ablations:** Multi-task vs. single-task; K=1 vs K=5; classifier vs. LLM reranking
- **Critical data:** FairytaleQA train evidence audit (with counterfactual verification) + 200 human-annotated samples
- **Highest-risk assumptions:** (1) Hard class size sufficient for reliable training, (2) counterfactual verification produces consistent labels, (3) classifier generalizes to generated questions (trained on FairytaleQA gold questions)

## Compute & Timeline Estimate

| Component | Cost | Time |
|-----------|------|------|
| Evidence audit (3 runs + counterfactual) | ~$30 API | 2 days |
| Hard augmentation + validation | ~$10 API | 1 day |
| Classifier training (5-fold CV) | ~20 GPU-hours | 1 day |
| Human annotation (200 × 2 annotators) | ~20 person-hours | 3 days |
| Generation (K=5, 4 methods) | ~$40 API | 2 days |
| Evaluation + analysis | — | 3 days |
| Paper writing | — | 1 week |
| **Total** | **<$100 API, 20 GPU-hours** | **4 weeks** |
