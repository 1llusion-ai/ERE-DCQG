# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** How to generate reading comprehension questions whose answering difficulty (Easy/Medium/Hard) is reliably controllable, where "difficulty" is defined by the minimum evidence sentences a reader must consult — not by surface features or graph hop count alone.
- **Must-solve bottleneck:** No mechanism verifies evidence necessity during generation; evaluation relies on unreliable LLM judges.
- **Non-goals:** General-purpose QG, SOTA, new LLM, multilingual.
- **Constraints:** Single-GPU, 4-6 weeks, CCF C, <$100 API.
- **Success condition:** Independent difficulty assessment (trained model) correctly predicts target difficulty at significantly higher rate than baselines.

## Anchor Check

- **Original bottleneck:** Graph hop count ≠ answering difficulty; LLM judges unreliable.
- **Why the revised method still addresses it:** The evidence-necessity framework directly redefines difficulty as "how many sentences are genuinely required." The multi-task classifier makes this assessment reproducible. Reranking closes the generation→evaluation loop.
- **Reviewer suggestions rejected as drift:** None — all suggestions strengthen the core.

## Simplicity Check

- **Dominant contribution after revision:** Evidence-Necessity Difficulty Framework (ENDF) — a unified definition + automated audit + trained evaluator. The framework, not the classifier architecture, is the novelty.
- **Components removed or merged:**
  - Claim 3 (graph diversity) dropped — the graph-guided method is just one of 4 compared methods, not a separate contribution
  - Evidence audit and classifier training merged into one "ENDF" contribution block
- **Reviewer suggestions rejected as unnecessary complexity:**
  - LLM calibration (Option B) rejected — adds a dependency on LLM logit access which API-based models don't always provide; also less concrete for a systems paper
- **Why the remaining mechanism is still the smallest adequate route:** One framework definition + one multi-task classifier + one reranking mechanism. Each directly addresses the bottleneck.

## Changes Made

### 1. Reframe dominant contribution as Evidence-Necessity Difficulty Framework

- **Reviewer said:** "Train a classifier on LLM labels" is incremental. The difficulty definition is the real novelty.
- **Action:** Reframe the paper's core as ENDF: the operationalized evidence-necessity definition (Easy=1 sent alone sufficient, Medium=2 sent with one relation, Hard=3+ with bridge removal test) + automated audit pipeline. The classifier is the mechanism that instantiates this framework, not the contribution itself.
- **Reasoning:** This makes the novelty clearer: we don't just classify difficulty, we define what difficulty means in terms of verifiable evidence necessity.
- **Impact on core method:** Framing change only; no pipeline change.

### 2. Enrich classifier input with evidence features (multi-task)

- **Reviewer said:** Classifier input of (story, question, answer) is surface-only. Include evidence features.
- **Action:** Design a multi-task classifier:
  - Primary task: 3-class difficulty prediction
  - Auxiliary task: evidence sentence identification (which sentence IDs are required)
  - Input: `[CLS] story_section [SEP] question [SEP] answer [SEP]` with sentence-level markers `[S0]...[Sn]`
  - Output head 1: difficulty class (3-way softmax)
  - Output head 2: per-sentence binary (is this sentence required evidence?)
  - Joint loss: L = L_difficulty + λ * L_evidence (λ=0.3)
- **Reasoning:** The evidence audit data already provides both labels (difficulty + required_sentence_ids). Multi-task learning makes the classifier truly evidence-aware, not just surface-pattern-matching. This is a genuine step beyond "fine-tune BERT on labels."
- **Impact on core method:** Adds one auxiliary task head; training data already has both labels; no extra annotation cost.

### 3. Tighten method specifics

- **Reviewer said:** Quality score, input truncation, Easy label assignment, human annotation protocol are underspecified.
- **Action:**
  - Quality score: binary gate from quality_judge (answerable=yes AND asks_expected_answer=yes AND fluent=yes). Among quality-passing candidates, rank by P(target_difficulty).
  - Truncation: Use evidence-focused truncation — keep answer sentence ± 2 sentences, plus bridge sentences, plus anchor sentences. This keeps context relevant and within 512 tokens.
  - Easy labels: Use evidence audit result (answer_sentence_alone_sufficient=yes AND num_required_sentences=1) regardless of ex_or_im metadata.
  - Human annotation: 2 annotators, Cohen's kappa reported, disagreements resolved by discussion, 200 samples (stratified 67 Easy + 67 Medium + 66 Hard).
- **Impact on core method:** Specifications only; no architectural change.

### 4. Drop Claim 3 (graph diversity)

- **Reviewer said:** Graph doesn't clearly help. Drop as separate contribution.
- **Action:** Remove Claim 3. Keep graph-guided "Ours" as one of 4 QG methods in the comparison table. If it benefits more from reranking, report as an observation, not a claim.
- **Reasoning:** Simplifies the paper to 2 clean claims. The graph is part of the experimental setup, not the contribution.
- **Impact on core method:** Removes one claim and one experiment; tightens the story.

### 5. Strengthen application motivation

- **Reviewer said:** Paper needs a "so what" — who benefits?
- **Action:** Frame the work in the context of adaptive reading comprehension education. FairytaleQA is designed for K-8 education (Xu et al., 2022). Difficulty-controlled QG enables personalized reading practice.
- **Impact on core method:** Framing only.

## Revised Proposal

# Research Proposal: Evidence-Necessity-Grounded Difficulty-Controlled Question Generation for Narrative Reading Comprehension

## Problem Anchor
[Verbatim from round 0 — unchanged]

## Technical Gap

**The core gap in difficulty-controlled QG is the disconnect between structural difficulty (graph hops, path length) and answering difficulty (how many evidence sentences a reader must actually consult).** Prior work uses surface features (question type, word count) or structural features (graph hops) that correlate with but do not guarantee answering difficulty. Our own experiments confirm: 3-hop event paths produce 0% verified Hard questions on MAVEN-ERE; on FairytaleQA, graph-guided QG is statistically indistinguishable from prompt-based baselines on difficulty accuracy.

**The secondary gap is evaluation reliability.** All current DCQG evaluation uses LLM judges whose predictions are sensitive to prompt wording, irreproducible across runs, and show 9-36% error rates.

## Method Thesis

**One-sentence thesis:** We define question difficulty by evidence necessity (how many sentences must be consulted), build an automated audit pipeline to label existing QA data with this definition, and train a multi-task classifier that jointly predicts difficulty and identifies evidence sentences — usable as both a reliable evaluator and a generation-time reranker.

## Contribution Focus

- **Dominant contribution:** The Evidence-Necessity Difficulty Framework (ENDF):
  1. An operationalized difficulty definition grounded in evidence necessity
  2. An automated evidence audit pipeline that labels existing QA data
  3. A multi-task classifier (difficulty prediction + evidence identification) that instantiates this framework

- **Supporting contribution:** Classifier-based reranking that closes the generation→evaluation loop — applicable to any QG method

- **Explicit non-contributions:**
  - The narrative evidence graph (existing component, one of 4 compared methods)
  - General QG quality improvements
  - New QG architectures

## Proposed Method

### System Overview

```
                    ENDF: Evidence-Necessity Difficulty Framework
                    ==========================================

Stage 1: Evidence Audit (offline, one-time)
  FairytaleQA QA pairs → LLM evidence analysis →
    - difficulty label (Easy/Medium/Hard)
    - required_evidence_sentence_ids
    - bridge_sentence_ids
    - reasoning_operation, answer_sentence_alone_sufficient
    - bridge_removal_effect
  + Human validation (200 samples, 2 annotators, κ reported)

Stage 2: Multi-Task Classifier Training
  Input:  [CLS] story_section [SEP] question [SEP] answer [SEP]
          (with [S0]...[Sn] sentence markers in story_section)
  Task 1: Difficulty classification (3-way: Easy/Medium/Hard)
  Task 2: Evidence sentence identification (per-sentence binary)
  Loss:   L = L_difficulty + 0.3 * L_evidence
  Model:  DeBERTa-v3-base + 2 task heads

Stage 3: Question Generation (4 methods compared)
  For each (story, answer, difficulty):
    → Generate K=5 candidate questions
    → Quality gate: answerable ∧ asks_expected_answer ∧ fluent
    → Reranking: select argmax P_classifier(target_difficulty)
    → Output: difficulty-controlled question

Stage 4: Evaluation
  Primary:   Trained ENDF classifier (reproducible, fast)
  Secondary: Human evaluation (100 samples)
  Diagnostic: LLM judge (for comparison with classifier)
```

### Core Mechanism: Multi-Task Evidence-Aware Classifier

**Architecture:**
- Base: DeBERTa-v3-base (184M params)
- Head 1 (difficulty): [CLS] → Linear(768, 256) → ReLU → Linear(256, 3) → Softmax
- Head 2 (evidence): Per-token → Linear(768, 2) → Softmax (is this [Sn] token a required evidence sentence?)
- The evidence head operates on the [Sn] marker tokens only

**Input construction:**
- Story section with sentence markers: `[S0] First sentence. [S1] Second sentence. ...`
- Evidence-focused truncation: keep answer sentence ± 2 neighbors + bridge sentences + anchor sentence. Max 512 tokens.
- Full input: `[CLS] [S0]...[Sn] story_section [SEP] question [SEP] answer [SEP]`

**Training data:**
- Source: FairytaleQA train split evidence audit results
- Easy: QA pairs where answer_sentence_alone_sufficient=yes AND num_required_sentences=1 (~2000)
- Medium: QA pairs where num_required_sentences=2 (~1300)
- Hard: QA pairs where num_required_sentences≥3 AND bridge_removal_effect∈{ambiguous, unanswerable} (~330)
- Total: ~3630 labeled examples
- Stratified split: 80/10/10 train/dev/test
- Class weighting: inverse frequency (Hard weight ≈ 6x Easy weight)
- Augmentation for Hard: paraphrase questions using back-translation (Hard→Chinese→English) to 2x Hard examples (~660 total Hard)

**Training recipe:**
- Optimizer: AdamW, lr=2e-5 (DeBERTa), lr=5e-4 (task heads)
- Warmup: 10% of steps
- Epochs: 10, early stop on dev macro F1
- Batch: 16
- λ schedule: 0.3 constant (auxiliary evidence task)

**Inference (for reranking):**
- Input: (story_section, generated_question, target_answer)
- Output: P(Easy), P(Medium), P(Hard), evidence_sentence_ids
- Selection: among quality-passing candidates, pick argmax P(target_difficulty)

### Reranking Pipeline

For each candidate pool (K=5 questions per (story, answer, difficulty)):
1. Quality gate: keep candidates where quality_judge returns answerable=yes ∧ asks_expected_answer=yes ∧ fluent=yes
2. Run ENDF classifier on each surviving candidate
3. Select: argmax P(target_difficulty) among quality-passing candidates
4. Fallback: if 0 candidates pass quality gate, use the candidate with highest P(target_difficulty) regardless
5. Output: one selected question per (story, answer, difficulty)

### QG Methods Compared (Unchanged from Prior Work)

| Method | Input | Graph? |
|--------|-------|--------|
| Direct | story + answer + difficulty definition | No |
| ICL | story + answer + difficulty + few-shot examples | No |
| SelfRefine | Direct → critique → revise | No |
| Ours | story + answer + difficulty + evidence graph scaffold | Yes |

All methods use the same difficulty definitions (ENDF-derived), same API (Qwen-32B), same K=5 generation, same quality gate, same reranking.

### Failure Modes

1. **Hard class too small (~330 raw, ~660 augmented):** Monitor per-class F1 during training. If Hard F1 < 0.5 on dev, consider few-shot data augmentation via GPT-4o-mini question paraphrasing.
2. **Classifier overfits to LLM audit labels:** Human validation on 200 samples calibrates this. Report classifier-human agreement alongside classifier-LLM agreement.
3. **Reranking selects high-P but low-quality questions:** Quality gate prevents this — only quality-passing candidates are reranked.
4. **DeBERTa truncation loses critical evidence:** Evidence-focused truncation mitigates this. If too many Hard examples exceed 512 tokens, use Longformer-base instead.

### Novelty Argument

**Closest work:**
- CrossQG (Tomikawa & Uto, 2024): Difficulty-controlled QG using IRT-based difficulty estimation. Defines difficulty by item response, not evidence necessity. No evidence identification.
- Uto et al. (2023): IRT-based difficulty QG. Uses learner response data for difficulty labels, not evidence analysis.
- KAQG (Chen & Shiu, 2025): Knowledge graph + IRT + Bloom's taxonomy for MCQ. Focuses on knowledge coverage, not evidence necessity.
- GNET-QG (Jamshidi & Chali, 2025): GAT-based multi-hop QG. Uses graph structure for question complexity, no difficulty evaluation.

**Exact difference from all above:**
1. We define difficulty by **evidence necessity** (minimum sentences required), not by surface features, learner response, or structural complexity
2. We **jointly train for difficulty and evidence identification** — the classifier doesn't just predict difficulty, it explains it by identifying which sentences are evidence
3. We use the classifier for **generation-time reranking**, closing the evaluate→generate loop
4. Our difficulty labels come from an **automated evidence audit** on existing QA data — no learner response data needed

## Claim-Driven Validation Sketch

### Claim 1: The ENDF classifier provides more reliable and evidence-grounded difficulty assessment than LLM judges

- **Experiment:** Train ENDF classifier on FairytaleQA train evidence audit labels. Compare against 3 baselines on the human-annotated 200-sample test set:
  - GPT-4o-mini difficulty judge (current LLM judge)
  - Qwen-32B difficulty judge
  - DeBERTa classifier trained on difficulty ONLY (no evidence task, ablation)
- **Metrics:** Macro F1, per-class F1, Cohen's κ vs. human, evidence sentence recall (ENDF only)
- **Expected result:** ENDF macro F1 > LLM judges by ≥5pp; ENDF κ > LLM κ; evidence recall >70%

### Claim 2: Classifier-based reranking improves difficulty accuracy for all QG methods

- **Experiment:** Generate K=5 candidates for all 4 methods × 3 difficulties on FairytaleQA validation. Compare:
  - K=1 (no reranking, random selection)
  - K=5 + quality gate + reranking by ENDF classifier
  - K=5 + quality gate + reranking by LLM judge P(difficulty)
- **Metrics:** Difficulty accuracy, macro accuracy, Spearman ρ, per-level hit rate, macro F1
- **Expected result:** Reranking by ENDF improves difficulty accuracy by ≥5pp for all methods; Hard hit rate improves by ≥10pp; ENDF reranking > LLM reranking

### Ablation: Multi-task vs. single-task classifier

- **Experiment:** Compare ENDF (multi-task: difficulty + evidence) vs. difficulty-only classifier
- **Metric:** Macro F1 on difficulty, evidence sentence recall
- **Expected result:** Multi-task improves difficulty F1 by ≥2pp due to evidence grounding

## Experiment Handoff Inputs

- **Must-prove claims:** ENDF > LLM judges; reranking improves all methods
- **Must-run ablations:** Multi-task vs. single-task, K=1 vs K=5, ENDF reranking vs. LLM reranking
- **Critical data:** FairytaleQA train (evidence audit) + validation; 200 human-annotated samples
- **Highest-risk:** Hard class size, evidence audit label quality, truncation

## Compute & Timeline Estimate

- **Classifier training:** ~4 GPU-hours (multi-task, with augmentation, 10 epochs)
- **Human annotation:** ~200 samples × 2 annotators × 3 min = 20 person-hours
- **Generation (K=5):** 450 × 4 methods × 5 = 9000 API calls ≈ $40
- **Total:** 4 weeks, <$100 API, single GPU
