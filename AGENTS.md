# DCQG Codex Instructions

This repository is the DCQG / event-path-constrained question generation project.

Before making research claims, code changes, or experiment decisions, read:

- `review-stage/PROJECT_STATUS.md`
- `review-stage/RESULTS_SUMMARY.md` if comparing results

`PROJECT_STATUS.md` is the source of truth for current pipeline status, known failures, metric definitions, and update protocol.

---

## Core Research Framing

The project is **target-event-grounded question generation with event-path constraints**.

Difficulty is determined by hop count (Easy=1hop, Medium=2hop, Hard=3hop). The old four-dimensional scoring (PL/RD/ES/EA) has been removed.

The main research question is:

> Can document-level event paths help generate questions whose required reasoning difficulty matches Easy / Medium / Hard targets?

Do not claim that PathQG-HardAware fully outperforms ICL or ZeroShot on general question quality unless current results support it.

---

## Trace-First Debugging Rule

Never diagnose quality problems from aggregate tables alone.

For any unexpected result, inspect trace items in this order:

```text
MAVEN raw sentence/event/offset
 -> graph node and directed edge
 -> sampled path and relation direction
 -> answer phrase extraction
 -> prefilter result
 -> LLM path judge prompt/raw/parsed
 -> QG prompt/raw/repair
 -> quality filter
 -> solver answer and judge raw response
```

If a conclusion cannot be traced to concrete examples, do not write it as a project conclusion or paper claim.

Known upstream failure modes (all fixed as of 2026-05-01):

- `answer_extraction.extract_answer_phrase_local()` is the single shared answer phrase extractor; use trace logs to verify truncation or partial extraction.
- `path_sampler.py` Medium sampling 鈥?fixed to strict `src -> mid -> tgt`.
- `difficulty_scorer.py` 鈥?deleted, replaced by hop-based scoring.
- Path length is not the same as actual inference-step difficulty.

---

## Current Priority

Upstream data chain is fixed. Next steps:

1. Run larger pilots to validate hop-based scoring works at scale.
2. Address hard degradation and path coverage issues.
3. Full experiment rerun when pilots look good.

---

## Key Files

The primary codebase is the `dcqg/` package. Do not depend on `event_qg/`; it is a legacy implementation scheduled for deletion.

**Package (`dcqg/`):**
- `dcqg/graph/event_graph.py`: builds document-level event graphs.
- `dcqg/path/sampler.py`: samples Easy / Medium / Hard paths (hop-based).
- `dcqg/path/diagnostics.py`: deterministic path diagnostics / prefilter.
- `dcqg/path/llm_filter.py`: LLM path quality judge.
- `dcqg/path/answer_extraction.py`: answer phrase extraction and final-event validity.
- `dcqg/path/direction.py`: path binding check, Hard validation.
- `dcqg/generation/generator.py`: PathQG-HardAware generation with retry.
- `dcqg/generation/prompts.py`: difficulty-aware few-shot prompts.
- `dcqg/generation/baselines.py`: baseline models (ZeroShot, ICL, SelfRefine, ablations).
- `dcqg/question_filter/pipeline.py`: post-generation quality filter pipeline.
- `dcqg/evaluation/solver.py`: Solver and Judge classes.
- `dcqg/evaluation/judge.py`: LLM judge v2, quality judge, evaluate_item.
- `dcqg/tracing/render.py`: full-chain debug trace (readable_trace.md).

**Entry points (`scripts/`):**
- `scripts/run_smoke_test.py`: end-to-end smoke test.
- `scripts/run_quality_pilot.py`: generation/filter pilot.
- `scripts/run_pipeline.py`: full pipeline orchestrator.
- `scripts/01_build_graph.py` through `scripts/05_evaluate.py`: individual stage entry points.

- `review-stage/PROJECT_STATUS.md`: project status and required update protocol.

---

## Commands

Run commands as modules from the repository root so `dcqg/` resolves cleanly.

Run a 3-item end-to-end smoke test (skip path judge and solver for speed):

```powershell
python -m scripts.run_smoke_test --limit 3 --skip_path_judge --skip_solver
```

Run the full smoke test with all stages:

```powershell
python -m scripts.run_smoke_test --limit 5
```

Run quality pilot (30 per level):

```powershell
python -m scripts.run_quality_pilot --n_per_level 30
```

Run the full pipeline:

```powershell
python -m scripts.run_pipeline --limit 50 --skip_path_judge --skip_solver
```

Individual stages:

```powershell
python -m scripts.01_build_graph --input data/raw/maven_ere/valid.jsonl --limit 10
python -m scripts.02_sample_paths --input data/raw/maven_ere/valid.jsonl --limit 100
python -m scripts.03_filter_paths --input outputs/runs/latest/paths.raw.jsonl --skip_llm_judge
python -m scripts.04_generate_questions --input outputs/runs/latest/paths.filtered.jsonl
python -m scripts.05_evaluate --input outputs/runs/latest/questions.raw.jsonl --skip_solver
```

Check syntax for a changed Python file:

```powershell
python -m py_compile dcqg/path/sampler.py
```

---

## Experiment Rules

- Use fixed samples when comparing methods.
- Treat `PathOnlyQG` and `RelationTypeQG` as ablations, not external baselines.
- Use `ZeroShotTargetQG`, `ICLTargetQG`, `SelfRefine`, and `PathQG-HardAware` as main comparisons.
- Do not use `Composite` as the main paper metric; it is internally weighted and custom.
- Prefer primary metrics from `PROJECT_STATUS.md`: Difficulty Consistency, Inference-step Consistency, Solver Accuracy by Difficulty, and Question Quality.
- API errors and parse errors must be reported separately. Do not count them as real model judgments.

---

## Documentation Update Rule

After any change to sampling, filtering, judging, generation, evaluation, or metrics, update:

- `review-stage/PROJECT_STATUS.md`

Update at least:

- latest data or pilot result
- solved problems
- remaining problems
- reportable conclusions
- trace/log status

Keep `PROJECT_STATUS.md` as the full research log. Keep this `AGENTS.md` concise and operational.
