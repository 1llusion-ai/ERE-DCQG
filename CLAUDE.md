# DCQG Claude Instructions

This repository is the DCQG / evidence-necessity-aware difficulty-controlled question generation project.

Before making research claims, code changes, or experiment decisions, read:

- `review-stage/PROJECT_STATUS.md`
- `refine-logs/FINAL_PROPOSAL.md` for the full experiment plan

`PROJECT_STATUS.md` is the source of truth for current pipeline status, known failures, metric definitions, and update protocol.

---

## Core Research Framing

The project is **evidence-necessity-aware question generation with difficulty control**.

Difficulty is defined by a **two-dimensional combination** of answer explicitness and evidence scope (not hop count):

- **Easy**: Answer directly found in text + 1 necessary evidence sentence.
- **Medium**: Case 1 (answer not directly found + 1 necessary evidence sentence + simple inference) OR Case 2 (answer directly found + multiple necessary evidence sentences + simple synthesis).
- **Hard**: Answer not directly found + multiple necessary evidence sentences + at least one inference.

Canonical definitions live in `dcqg/difficulty/definitions.py`. All prompts inject from this single source.

The main research question is:

> Can evidence-necessity labels enable a multi-task classifier to provide reproducible, interpretable difficulty evaluation and improve difficulty control via generation-time reranking?

Do not claim that PathQG-HardAware outperforms baselines on general question quality unless current results support it.

---

## Current Priority

Four-stage experiment plan (details in `refine-logs/FINAL_PROPOSAL.md`):

1. **Evidence Audit** — No-Vote pipeline (Selector → Blind Verifier → Removal Verifier) on FairytaleQA to produce `(story, QA, evidence_sentences, difficulty_label)` labels.
2. **Classifier Training** — Multi-task DeBERTa-v3 (difficulty head + evidence head) on audit labels.
3. **QG + Reranking** — Generate K=5 candidates per method, rerank with classifier.
4. **Evaluation** — Human eval (primary), classifier on held-out fold (consistency), LLM judge (diagnostic).

Current bottleneck: No-Vote pipeline produces very few final Hard labels (Blind Verifier is very strict).

---

## Key Files

The primary codebase is the `dcqg/` package. Do not depend on `event_qg/`; it is a legacy implementation scheduled for deletion.

**Package (`dcqg/`):**
- `dcqg/difficulty/definitions.py`: canonical Easy/Medium/Hard definitions — single source of truth.
- `dcqg/difficulty/classifier.py`: MultiTaskDifficultyClassifier (DeBERTa-v3).
- `dcqg/difficulty/data.py`: training data construction for classifier.
- `dcqg/difficulty/reranker.py`: DifficultyReranker for generation-time selection.
- `dcqg/path/no_vote_evidence.py`: No-Vote evidence audit pipeline (Selector → Blind Verifier → Removal Verifier).
- `dcqg/path/fairytale_evidence_audit.py`: `_split_sentences()` utility and old FairytaleEvidenceAuditor.
- `dcqg/path/answer_grounded_evidence.py`: answer-grounded evidence planner + validator.
- `dcqg/graph/narrative_graph.py`: narrative evidence graph extraction for QG.
- `dcqg/generation/fairytale_qg.py`: 4 QG methods (Direct, ICL, SelfRefine, Ours).
- `dcqg/generation/parser.py`: JSON response parsing and generation helper.
- `dcqg/question_filter/grammar.py`: grammar quality filter.
- `dcqg/evaluation/judge.py`: LLM judge, solver, evaluate_item.
- `dcqg/evaluation/metrics.py`: fair metrics computation.
- `dcqg/evaluation/report.py`: comparison table formatting.
- `dcqg/datasets/fairytaleqa_loader.py`: FairytaleQA data loader (HuggingFace).

**Entry points (`scripts/`):**
- `scripts/run_evidence_no_vote_pilot.py`: Stage 1 — No-Vote evidence audit.
- `scripts/train_classifier.py`: Stage 2 — train multi-task classifier.
- `scripts/run_fairytale_qg_pilot.py`: Stage 3 — QG with 4 methods.
- `scripts/run_crossqg_eval.py`: Stage 3/4 — CrossQG-style evaluation.
- `scripts/run_k5_generation.py`: Stage 3 — K=5 candidate generation.
- `scripts/run_reranking_eval.py`: Stage 3/4 — reranking evaluation.
- `scripts/run_human_eval.py`: Stage 4 — human evaluation sample prep.
- `scripts/compute_table1.py` / `scripts/compute_table3.py`: Stage 4 — results tables.
- `scripts/run_llm_judge_difficulty.py`: Stage 4 — LLM judge diagnostic.

- `review-stage/PROJECT_STATUS.md`: project status and required update protocol.

---

## Commands

Run commands as modules from the repository root so `dcqg/` resolves cleanly.

### Stage 1 — Evidence Audit

```powershell
python -u -m scripts.run_evidence_no_vote_pilot --split train --output_dir outputs/runs/no_vote_pilot --implicit_limit 100 --batch_size 5 --timeout 300 --model Qwen/Qwen3-32B
```

### Stage 2 — Classifier Training

```powershell
python -m scripts.train_classifier --data outputs/runs/evidence_audit/train_dataset.jsonl --output_dir outputs/runs/classifier
```

### Stage 3 — QG + Reranking

```powershell
python -m scripts.run_fairytale_qg_pilot --candidates_path outputs/runs/candidates.jsonl --output_dir outputs/runs/qg_pilot
python -m scripts.run_k5_generation --candidates_path outputs/runs/candidates.jsonl --output_dir outputs/runs/k5
```

### Stage 4 — Evaluation

```powershell
python -m scripts.run_crossqg_eval --selection_mode story_matched --output_dir outputs/runs/crossqg
python -m scripts.run_reranking_eval --k5_dir outputs/runs/k5 --classifier_path outputs/runs/classifier
python -m scripts.run_llm_judge_difficulty --samples_path outputs/eval/test_200.jsonl
```

Check syntax for a changed Python file:

```powershell
python -m py_compile dcqg/path/no_vote_evidence.py
```

---

## Experiment Rules

- Use fixed samples when comparing methods.
- Treat Direct, ICL, and SelfRefine as baselines; Ours (narrative graph) as the proposed method.
- Do not expose target difficulty to the independent difficulty judge.
- Do not use solver answer or solver correctness as input to the difficulty judge.
- Do not use `Composite` as the main paper metric.
- Primary metrics: Difficulty consistency (human-evaluated), classifier accuracy (held-out fold), per-class F1, evidence recall/precision.
- API errors and parse errors must be reported separately. Do not count them as real model judgments.
- Do not depend on `event_qg/` or MAVEN-ERE scripts in new code.

---

## Documentation Update Rule

After any change to evidence audit, classifier training, generation, reranking, or evaluation, update:

- `review-stage/PROJECT_STATUS.md`

Update at least:

- latest data or pilot result
- solved problems
- remaining problems
- reportable conclusions

Keep `PROJECT_STATUS.md` as the full research log. Keep this `CLAUDE.md` concise and operational.
