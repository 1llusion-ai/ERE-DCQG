# DCQG Claude Instructions

This repository is the DCQG / event-path-constrained question generation project.

Before making research claims, code changes, or experiment decisions, read:

- `review-stage/PROJECT_STATUS.md`
- `review-stage/RESULTS_SUMMARY.md` if comparing results

`PROJECT_STATUS.md` is the source of truth for current pipeline status, known failures, metric definitions, and update protocol.

---

## Core Research Framing

The project is **target-event-grounded question generation with event-path constraints**.

Do not frame the method as a validated four-dimensional difficulty scoring method. The current evidence shows that difficulty labels are mostly driven by path length. Relation diversity, evidence span, and event ambiguity are auxiliary descriptors unless later experiments prove otherwise.

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
 -> difficulty score
 -> answer phrase extraction
 -> prefilter result
 -> LLM path judge prompt/raw/parsed
 -> QG prompt/raw/repair
 -> quality filter
 -> solver answer and judge raw response
```

If a conclusion cannot be traced to concrete examples, do not write it as a project conclusion or paper claim.

Known upstream failure modes:

- `compare_hardaware.extract_answer_phrase_local()` used a fixed trigger window and can produce truncated answers such as `has been described as one of the`.
- `path_sampler.py` Medium sampling has a direction-construction risk and must be checked as strict `src -> mid -> tgt`.
- `difficulty_scorer.py` reverse-edge lookup can hide invalid path directions.
- Path length is not the same as actual inference-step difficulty.

---

## Current Priority

Do not tune prompts or rerun full experiments before fixing the upstream data chain.

Current priority order:

1. Preserve MAVEN event `offset` in sampled paths.
2. Fix Medium path sampling to strict directed paths.
3. Stop silently using reverse edges in relation subtype lookup.
4. Replace fixed-window answer phrase extraction with offset/clause-based extraction or an LLM canonicalizer.
5. Add full-chain trace logs.
6. Rerun small pilots before any full 300 or full 3890 run.

---

## Key Files

- `event_qg/src/graph_builder.py`: builds document-level event graphs.
- `event_qg/src/path_sampler.py`: samples Easy / Medium / Hard paths.
- `event_qg/src/difficulty_scorer.py`: computes PL/RD/ES/EA/D labels.
- `event_qg/src/path_prefilter.py`: deterministic path diagnostics / light prefilter.
- `event_qg/src/path_llm_judge.py`: LLM path quality judge.
- `event_qg/src/compare_hardaware.py`: PathQG-HardAware generation and answer phrase enrichment.
- `event_qg/src/quality_filter.py`: post-generation quality filters.
- `event_qg/src/baselines.py`: main baselines and unified evaluation.
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
