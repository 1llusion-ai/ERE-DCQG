# DCQG Claude Instructions

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

- `event_qg/src/graph_builder.py`: builds document-level event graphs.
- `event_qg/src/path_sampler.py`: samples Easy / Medium / Hard paths (hop-based).
- `event_qg/src/path_prefilter.py`: deterministic path diagnostics / light prefilter.
- `event_qg/src/path_llm_judge.py`: LLM path quality judge.
- `event_qg/src/answer_extraction.py`: answer phrase extraction and final-event validity helpers.
- `event_qg/src/compare_hardaware.py`: PathQG-HardAware generation.
- `event_qg/src/quality_filter.py`: post-generation quality filters.
- `event_qg/src/baselines.py`: baseline models and unified evaluation.
- `event_qg/src/evaluator.py`: solver and basic evaluation.
- `event_qg/src/evaluator_v2.py`: LLM judge and quality evaluation.
- `event_qg/src/trace_utils.py`: full-chain debug trace.
- `event_qg/src/full_pipeline_smoke.py`: end-to-end smoke test.
- `event_qg/src/quality_pilot.py`: generation/filter pilot.
- `review-stage/PROJECT_STATUS.md`: project status and required update protocol.

---

## Commands

Run LLM path judge pilot with the model configured in `event_qg/.env`:

```powershell
python event_qg/src/path_llm_judge.py `
  --input event_qg/outputs/prefiltered_paths.jsonl `
  --output_dir event_qg/outputs/path_judge_pilot `
  --sample_per_level 5
```

Run gpt-4o explicitly:

```powershell
python event_qg/src/path_llm_judge.py `
  --input event_qg/outputs/prefiltered_paths.jsonl `
  --output_dir event_qg/outputs/path_judge_pilot_gpt4o `
  --sample_per_level 5 `
  --model gpt-4o
```

Check syntax for a changed Python file:

```powershell
python -m py_compile event_qg/src/path_llm_judge.py
```

Run a true 5-item end-to-end trace smoke test:

```powershell
python event_qg/src/full_pipeline_smoke.py `
  --limit 5 `
  --output_dir event_qg/outputs/full_pipeline_smoke_5
```

Use this command when the user asks for a full-chain trace. `quality_pilot.py`
is a generation/filter pilot and does not run the full path-judge + solver
evaluation chain unless it is explicitly extended later.

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

Keep `PROJECT_STATUS.md` as the full research log. Keep this `CLAUDE.md` concise and operational.
