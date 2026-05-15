# Research Proposal: Evidence-Grounded Difficulty-Controlled Question Generation with Narrative Evidence Graphs

## Problem Anchor

- **Bottom-line problem:** How to generate reading comprehension questions whose answering difficulty (Easy/Medium/Hard) is reliably controllable, where "difficulty" is defined by the minimum evidence sentences a reader must consult — not by surface features or graph hop count alone.

- **Must-solve bottleneck:** Current graph-based QG methods define difficulty by structural properties (hop count, path length) that do not guarantee answering difficulty. A 3-hop event path can still yield an Easy question if the answer is locally extractable from one sentence. The project has demonstrated this empirically: on MAVEN-ERE, 0/112 blind Hard candidates were judged Hard; on FairytaleQA, the evidence-graph-guided "Ours" method achieves only 27% Hard hit rate, statistically indistinguishable from ICL baselines (28.6%). The core gap is: **there is no mechanism that verifies and enforces evidence necessity during generation**, and **the evaluation relies entirely on unreliable LLM judges** with no trained model to provide consistent difficulty assessment.

- **Non-goals:**
  - Building a general-purpose QG system (we focus on narrative reading comprehension)
  - Surpassing top-venue SOTA (target is CCF C-level venue)
  - Training a new LLM from scratch
  - Handling languages other than English

- **Constraints:**
  - Compute: no multi-GPU training; single-GPU fine-tuning (A100/4090 level) is feasible
  - Data: FairytaleQA dataset (~10K QA pairs), 2166 implicit train QA pairs available with evidence annotations
  - API budget: SiliconFlow/AIHUBMIX API for LLM calls (Qwen-32B/GPT-4o-mini)
  - Time: ~4-6 weeks to complete experiments
  - Venue target: CCF C-level (NLPCC, CoNLL, PRICAI, or equivalent)

- **Success condition:** The method should produce questions where an independent difficulty classifier (not the same LLM that generated the questions) correctly predicts the target difficulty at a significantly higher rate than baselines, AND the classifier is a trained model rather than a prompt-based LLM judge.

## Technical Gap

### Why Current Methods Fail

**Surface-level difficulty definitions fail.** Most existing DCQG work defines difficulty via surface features (question type, word count, answer length) or structural features (graph hops). These are necessary but not sufficient conditions for answering difficulty. A "why" question can be Easy if the reason is in the same sentence as the action.

**Graph hop count ≠ reasoning hops.** The project's own MAVEN-ERE experiments proved this conclusively: 3-hop event paths produced 0% verified Hard questions because the final event phrase was always locally identifiable. The pivot to evidence-necessity (answer_sentence_alone_sufficient=no) on FairytaleQA improved the situation (4% → 15.2% Hard rate on implicit subset), but the generation mechanism still doesn't strongly enforce that generated questions require multi-sentence reasoning.

**LLM judges are unreliable evaluators.** All current difficulty evaluation uses LLM judges (GPT-4o-mini / Qwen-32B). These judges:
- Are inconsistent across runs (no reproducibility guarantee)
- Can be gamed by prompt engineering (the project changed judge prompts and saw Pred Hard jump from 0% to 24.5%)
- Cannot serve as fair baselines vs. the generation method (same family of models)
- Show high error rates (8.9%-35.6% across methods in the CrossQG eval)

**Graph extraction is expensive but marginally effective.** The Ours method requires graph extraction + self-check (2 extra API calls per candidate) but achieves only +4.2pp overall accuracy over Direct, with no statistical significance (bootstrap p=0.69). The graph adds pipeline complexity without a clear payoff.

### Why Naive Fixes Are Insufficient

- **More data:** Simply scaling up from 450 to 4500 candidates won't fix the fundamental issue that the graph doesn't enforce evidence necessity.
- **Better prompts:** The project has already iterated through 5+ Hard prompt strategies. Prompt engineering has diminishing returns.
- **Larger models:** Using GPT-4 instead of Qwen-32B won't change the architecture's inability to verify evidence necessity.

### Smallest Adequate Intervention

The smallest mechanism that could fix the bottleneck is a **trained difficulty classifier** that:
1. Provides consistent, reproducible difficulty assessment during evaluation
2. Can be used as a **reranking/filtering signal** during generation to select candidates that actually achieve the target difficulty
3. Is trained on evidence-grounded difficulty labels derived from the existing FairytaleQA evidence audit data

### Core Technical Claim

**Evidence-grounded difficulty labels (derived from evidence-necessity analysis) + a trained difficulty classifier (used for both evaluation and generation-time reranking) produce significantly better difficulty-controlled questions than graph-hop-based or prompt-only approaches.**

## Method Thesis

- **One-sentence thesis:** A trained difficulty classifier, built on evidence-necessity labels from narrative QA, serves as both a reliable evaluator and a generation-time reranker to produce difficulty-controlled questions that withstand independent evaluation.

- **Why this is the smallest adequate intervention:** The evidence audit pipeline already exists and produces labeled data. Training a classifier on these labels is straightforward. Using the classifier as a reranker requires no changes to the generation architecture — it's a post-hoc selection mechanism that works with any generator.

- **Why this route is timely:** Recent work (CrossQG, CQG) shows that difficulty-controlled QG is an active area, but evaluation remains LLM-judge-dependent. A trained classifier provides a more rigorous evaluation foundation, which is increasingly demanded as the field moves past prompt-based evaluation.

## Contribution Focus

- **Dominant contribution:** Evidence-necessity-based difficulty definition + trained difficulty classifier for narrative QG, usable as both evaluator and reranker.

- **Supporting contribution:** Narrative evidence graph as optional scaffolding for Hard question generation (the "Ours" method), showing that graph guidance improves Hard hit rate when combined with classifier-based reranking.

- **Explicit non-contributions:**
  - We do not claim the graph alone controls difficulty (shown to be insufficient)
  - We do not claim SOTA on general QG quality metrics
  - We do not propose a new LLM architecture

## Proposed Method

### Complexity Budget

- **Frozen / reused:** FairytaleQA dataset, Qwen-32B/GPT-4o-mini as generators, existing evidence audit pipeline
- **New trainable components:** (1) Difficulty classifier (fine-tuned BERT/RoBERTa), (2) Optional: evidence-aware feature extraction
- **Tempting additions intentionally not used:** RL-based difficulty optimization, multi-model ensemble, curriculum learning for generation

### System Overview

```
FairytaleQA stories + QA pairs
  ──→ Evidence Necessity Audit (LLM-based, one-time)
       ├── Easy/Medium/Hard labels per QA pair
       ├── Evidence sentences, bridge IDs, reasoning operations
       └── Human validation subset (100-200 samples)
  ──→ Difficulty Classifier Training
       ├── Input: (story_section, question, answer)
       ├── Labels: Easy/Medium/Hard from evidence audit
       ├── Model: fine-tuned RoBERTa-base or DeBERTa-v3-base
       └── Output: 3-class difficulty prediction + confidence
  ──→ Candidate Generation (4 methods, same as current)
       ├── Direct QG
       ├── ICL QG
       ├── SelfRefine QG
       └── Ours (evidence graph guided)
  ──→ Classifier-Based Reranking (NEW)
       ├── Generate K=5 candidates per (story, answer, difficulty)
       ├── Run difficulty classifier on each candidate
       ├── Select candidate with highest P(target_difficulty)
       └── Tie-break by quality score
  ──→ Evaluation
       ├── Trained difficulty classifier (primary)
       ├── Human evaluation subset
       └── LLM judge (secondary/diagnostic only)
```

### Core Mechanism: Trained Difficulty Classifier

**Input:** Concatenation of `[CLS] story_section [SEP] question [SEP] answer [SEP]`

**Output:** 3-class softmax over {Easy, Medium, Hard}

**Architecture:** RoBERTa-base or DeBERTa-v3-base with a classification head (768 → 256 → 3). ~125M parameters, fine-tunable on single GPU in <1 hour.

**Training data construction:**
1. Start with FairytaleQA train split: 2166 implicit QA pairs with evidence audit labels
2. Add explicit QA pairs for Easy class balance (~2000 explicit Easy)
3. Total training pool: ~4000 labeled examples (Easy ~2000, Medium ~1300, Hard ~330)
4. Apply stratified split: 80% train, 10% dev, 10% test
5. **Human validation:** Annotate 200 random samples (50 per difficulty + 50 borderline) to calibrate label quality

**Training recipe:**
- Optimizer: AdamW, lr=2e-5, warmup 10%, weight decay 0.01
- Epochs: 5-10, early stopping on dev macro F1
- Class weighting: inverse frequency to handle Hard class imbalance (Hard:Medium:Easy ≈ 1:4:6)
- Augmentation: paraphrase-based augmentation for Hard class if needed

**Why this is the main novelty:** Prior DCQG work evaluates difficulty via LLM judges or surface heuristics. A trained classifier provides: (1) reproducibility, (2) faster inference (~10ms vs ~2s per LLM call), (3) no prompt sensitivity, (4) can be used as a reranking signal during generation.

### Supporting Component: Classifier-Based Reranking

**Input:** K=5 generated candidate questions per (story, answer, target_difficulty)

**Process:**
1. Generate K candidates using the QG method (Direct/ICL/SelfRefine/Ours)
2. Run quality filter (grammar, answer consistency, etc.)
3. Run trained difficulty classifier on each quality-passing candidate
4. Score each candidate: `score = P(target_difficulty) * quality_score`
5. Select the highest-scoring candidate

**Why this does not create contribution sprawl:** Reranking is a direct application of the classifier — no new model, no new training, just inference. It's the natural way to close the loop between evaluation and generation.

### Narrative Evidence Graph (Existing Component)

The evidence graph is retained as the scaffolding for the "Ours" method. It provides:
- Difficulty-aware substructure selection (Easy: 1 node, Medium: 2 nodes, Hard: 3+ chain)
- Evidence role annotations (anchor/bridge/answer)
- Relation chain for prompt construction

**Key insight from current results:** The graph alone doesn't control difficulty (Ours ≈ baselines on accuracy), but it may help when combined with reranking: the graph generates more structurally diverse candidates, increasing the chance that at least one candidate achieves the target difficulty.

### Integration: Full Pipeline

1. **Offline (one-time):** Evidence audit → difficulty labels → train classifier
2. **Generation time:** Story + answer + difficulty → QG method → K candidates → quality filter → classifier reranking → final question
3. **Evaluation:** Classifier prediction (primary) + human evaluation (secondary) + LLM judge (diagnostic)

### Training Plan

**Stage 1: Classifier training (1 week)**
- Prepare training data from evidence audit results
- Human validation of 200 samples
- Train classifier, tune hyperparameters on dev set
- Report: accuracy, macro F1, per-class F1, confusion matrix

**Stage 2: Reranking integration (1 week)**
- Implement reranking pipeline
- Run K=5 generation for all 4 methods × 3 difficulties
- Compare reranked vs. non-reranked results

**Stage 3: Full evaluation (2 weeks)**
- Story-matched CrossQG evaluation with classifier
- Human evaluation on 100 samples
- Ablation studies

### Failure Modes and Diagnostics

1. **Classifier overfits to evidence audit labels:** Mitigate by human validation + cross-validation. Diagnostic: compare classifier predictions with human labels on held-out set.

2. **Hard class too small for reliable training:** Mitigate by class weighting + augmentation. Diagnostic: per-class F1 on Hard, learning curve analysis.

3. **Reranking doesn't improve over single-best:** Mitigate by increasing K. Diagnostic: reranking lift as a function of K.

4. **Graph still doesn't help even with reranking:** This is an acceptable finding — it shows the classifier is the key component, not the graph. Paper can report this as an ablation.

### Novelty and Elegance Argument

**Closest work:**
- CrossQG (2024): difficulty-controlled QG using LLM judges for evaluation — no trained classifier, no evidence-necessity definition
- CQG-style papers: use surface features (question type, lexical complexity) for difficulty — not evidence-grounded
- Multi-hop QG papers: use graph structure for question design — no difficulty control or evaluation

**Exact difference:** We define difficulty by evidence necessity (how many sentences a reader MUST consult), train a dedicated classifier on these labels, and use it for both evaluation and generation-time reranking. This closes the loop that prior work leaves open: prior methods generate at a target difficulty but evaluate with unreliable LLM judges.

**Why focused:** One new trainable component (classifier), one new mechanism (reranking), one clear evaluation improvement (trained vs. LLM judge). The graph is existing work, not a new contribution.

## Claim-Driven Validation Sketch

### Claim 1: The trained difficulty classifier provides more reliable difficulty assessment than LLM judges

- **Minimal experiment:** Train classifier on evidence audit labels. Compare classifier accuracy vs. LLM judge accuracy against human annotations on the same 200-sample test set.
- **Baselines:** GPT-4o-mini judge, Qwen-32B judge, random baseline, majority baseline
- **Metric:** Macro F1, per-class accuracy, inter-annotator agreement (classifier vs. human)
- **Expected evidence:** Classifier macro F1 > LLM judge macro F1 by ≥ 5pp; classifier per-class agreement with human > LLM judge per-class agreement

### Claim 2: Classifier-based reranking improves difficulty hit rate for all QG methods

- **Minimal experiment:** Generate K=5 candidates per item for all 4 methods. Compare single-best (K=1) vs. reranked (K=5) difficulty accuracy using the trained classifier as evaluator.
- **Baselines:** K=1 (no reranking) for each method
- **Metric:** Difficulty accuracy, macro accuracy, Spearman, per-level hit rate
- **Expected evidence:** Reranking improves difficulty accuracy by ≥ 5pp for all methods; Hard hit rate improves by ≥ 10pp

### Claim 3 (Optional): Evidence graph guidance generates more structurally diverse candidates, benefiting reranking

- **Minimal experiment:** Compare candidate diversity (lexical diversity, predicted difficulty distribution) across K=5 candidates for Ours vs. Direct. Measure whether Ours benefits more from reranking.
- **Ablation:** Ours with reranking vs. Direct with reranking
- **Metric:** Reranking lift (accuracy_reranked - accuracy_single_best)
- **Expected evidence:** Ours shows higher candidate diversity and larger reranking lift than Direct

## Experiment Handoff Inputs

- **Must-prove claims:** Classifier > LLM judge; reranking improves difficulty accuracy
- **Must-run ablations:** (1) Classifier vs. LLM judge evaluation, (2) K=1 vs K=5 reranking, (3) Ours+reranking vs. Direct+reranking
- **Critical datasets/metrics:** FairytaleQA train/validation splits, macro F1, per-class accuracy, human agreement
- **Highest-risk assumptions:** (1) Evidence audit labels are reliable enough to train on, (2) Hard class has enough examples, (3) Classifier generalizes beyond training distribution

## Compute & Timeline Estimate

- **Classifier training:** ~2 GPU-hours on single A100/4090
- **Human annotation:** ~200 samples × 2 annotators × 3 min/sample = ~20 person-hours
- **Generation (K=5 reranking):** 450 candidates × 4 methods × 5 candidates = 9000 API calls ≈ $30-50
- **Classifier inference for reranking:** Negligible (~10ms per candidate)
- **Total timeline:** 4-6 weeks
- **Total compute cost:** < $100 API + ~2 GPU-hours
