# DCQG Claude Instructions

This repository is the DCQG / difficulty-controlled question generation project.

Before making research claims, code changes, or experiment decisions, read:

- `review-stage/PROJECT_STATUS.md`
- `refine-logs/FINAL_PROPOSAL.md` for the current method thesis
- `refine-logs/EXPERIMENT_PLAN.md` for experiment block definitions

`PROJECT_STATUS.md` is the source of truth for current pipeline status, known failures, metric definitions, and update protocol.

---

## Core Research Framing

The project is **evidence-necessity-aware difficulty-controlled question generation** on FairytaleQA.

Difficulty is defined by a two-dimensional combination of answer explicitness and necessary evidence scope:

| Difficulty | Definition |
|---|---|
| Easy | Answer directly found, 1 necessary evidence sentence |
| Medium | Answer inferred from 1 sentence, OR direct answer requiring multiple sentences |
| Hard | Answer not directly found + multiple evidence sentences + complex/multi-step reasoning |

The main research question is:

> Can a multi-task classifier trained on counterfactual-verified evidence-necessity labels provide reliable difficulty assessment and enable generation-time reranking that improves difficulty control?

Do not reintroduce the old event-hop difficulty definition (Easy=1hop, Medium=2hop, Hard=3hop). That hypothesis failed on MAVEN-ERE.

---

## Current Method

The main pipeline uses the **no-vote evidence labeling approach**:

**Step 1: LLM-assisted labeling (Qwen3-32B, no-vote pipeline)**
- Selector: CoT evidence analysis with `/think` (selects evidence sentences, labels difficulty)
- Blind verifier: checks if selected evidence alone is sufficient (without seeing the question)
- Removal verifier: leave-one-out necessity check (removes each evidence sentence, checks if answer still determinable)
- Core module: `dcqg/path/no_vote_evidence.py`

**Step 2: Human annotation (200 samples, 2 annotators)**
- Calibrate and validate LLM labels
- Cohen's kappa for inter-annotator agreement

**Step 3: Multi-task DeBERTa classifier training**
- Head 1: difficulty classification (Easy/Medium/Hard)
- Head 2: evidence sentence identification ([S0]-[S30] markers)
- Trained on no-vote labels, 5-fold CV

**Step 4: Experiment comparison**
- 4 QG methods: Direct, ICL, SelfRefine, Ours (narrative evidence graph)
- K=5 candidates per item, classifier reranks by P(target_difficulty)
- Compare: K=1 (no reranking) vs K=5+classifier vs K=5+LLM judge vs K=5+random
- Validate with human evaluation

---

## Key Files

The primary codebase is the `dcqg/` package.

**Package (`dcqg/`):**
- `dcqg/datasets/fairytaleqa_loader.py`: FairytaleQA HuggingFace loader
- `dcqg/path/no_vote_evidence.py`: no-vote evidence labeling pipeline (main labeling approach)
- `dcqg/path/answer_grounded_evidence.py`: answer-grounded evidence analysis (Stage A, used by no-vote selector)
- `dcqg/path/counterfactual_verify.py`: counterfactual verification (used by removal verifier)
- `dcqg/path/self_consistency.py`: self-consistency aggregation
- `dcqg/graph/narrative_graph.py`: narrative evidence graph construction
- `dcqg/generation/fairytale_qg.py`: 4 QG methods (Direct, ICL, SelfRefine, Ours)
- `dcqg/generation/parser.py`: JSON response parsing
- `dcqg/difficulty/definitions.py`: difficulty definitions (Easy/Medium/Hard)
- `dcqg/difficulty/classifier.py`: multi-task DeBERTa classifier
- `dcqg/difficulty/data.py`: [Sn] tokenization and dataset class
- `dcqg/difficulty/reranker.py`: classifier-based reranking
- `dcqg/evaluation/judge.py`: LLM judge, quality judge, evaluate_item
- `dcqg/utils/`: config, API client, JSONL, text helpers

**Entry points (`scripts/`):**
- `scripts/run_evidence_no_vote_pilot.py`: no-vote evidence labeling (main labeling script)
- `scripts/run_crossqg_eval.py`: CrossQG evaluation (story-matched)
- `scripts/run_k5_generation.py`: K=5 candidate generation
- `scripts/run_reranking_eval.py`: reranking evaluation
- `scripts/train_classifier.py`: classifier training
- `scripts/run_llm_judge_difficulty.py`: LLM judge difficulty evaluation
- `scripts/run_human_eval.py`: human evaluation

---

## Commands

Run commands as modules from the repository root so `dcqg/` resolves cleanly.

Evidence audit (Stage A, 3 runs):

```bash
python -m scripts.run_answer_grounded_evidence_audit --split train --implicit_limit 100
```

No-vote evidence pilot:

```bash
python -u -m scripts.run_evidence_no_vote_pilot --split train --implicit_limit 500 --sample_mode random --seed 42 --batch_size 5 --timeout 300 --model Qwen/Qwen3-32B
```

CrossQG evaluation (story-matched):

```bash
python -m scripts.run_crossqg_eval --selection_mode story_matched --candidates_per_level_per_story 1 --max_stories 106
```

Check syntax for a changed Python file:

```bash
python -m py_compile dcqg/path/counterfactual_verify.py
```

---

## Experiment Rules

- Use fixed samples when comparing methods.
- Use Direct, ICL, SelfRefine, and Ours as the 4 QG methods.
- Do not use the old MAVEN-ERE baselines (ZeroShot-TargetQG, ICL-TargetQG, PathQG-HardAware).
- Primary metrics: Macro F1 (difficulty), per-class F1, Cohen's kappa vs. human.
- Secondary metrics: evidence sentence recall/precision/F1, difficulty accuracy, Spearman rho.
- API errors and parse errors must be reported separately.
- The classifier serves dual use: evaluation instrument AND reranking signal. Claims about reranking improvement must be validated by human evaluation, not just classifier self-evaluation.

---

## Documentation Update Rule

After any change to the evidence audit, classifier, generation, or evaluation pipeline, update:

- `review-stage/PROJECT_STATUS.md`

Keep `PROJECT_STATUS.md` as the full research log. Keep this `CLAUDE.md` concise and operational.
