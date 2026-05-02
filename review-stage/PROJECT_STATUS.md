# DCQG Project Status

**Last updated:** 2026-05-03  
**Status:** The codebase has migrated to the new `dcqg/` pipeline. Current evidence supports valid event-path-grounded question generation and stronger path dependency, but does not yet support a strong claim that Hard questions are independently difficulty-consistent.

**Maintenance rule:** Update this file whenever path sampling/filtering, question generation, question filtering, evaluation, baselines, or main experiment outputs change. Future diagnosis must start from trace logs and JSONL examples, not from aggregate numbers alone.

---

## 1. Current Pipeline

The project pipeline is organized around five main stages:

```text
MAVEN-ERE raw documents
  -> Event relation graph
  -> Path sampling and filtering
  -> Question generation
  -> Question filtering
  -> Solver + Judge evaluation
  -> Independent difficulty evaluation
```

Current implementation lives in the new package:

```text
dcqg/
  graph/              event graph construction
  path/               path sampling, answer extraction, diagnostics, LLM path filtering
  generation/         PathQG-HardAware, prompts, repair, baselines
  question_filter/    grammar, answer consistency, path coverage, shortcut, implicitness checks
  evaluation/         solver, judge, metrics, reports
  tracing/            full trace records and readable trace rendering
  utils/              config, API client, JSONL, text helpers
scripts/              runnable stage scripts and experiment drivers
```

Important repository rule:

- `event_qg/` is legacy and should not be imported by the new framework.
- Root `.env` is canonical.
- `.env.example` should be safe for GitHub.
- Data should be read from `data/raw/maven_ere/`.
- New experiment outputs should go under `outputs/runs/<run_name>/`.

---

## 2. Current Main Claim Status

Supported by current experiments:

```text
PathQG-HardAware improves valid path-grounded question yield and produces more path-dependent questions than target-aware baselines on the strict path pilot.
```

Not yet supported:

```text
PathQG-HardAware generates more independently difficulty-consistent Hard questions.
```

Reason:

- Internal filters show PathQG-HardAware satisfies event-path constraints better.
- Independent difficulty evaluation still predicts most current Medium/Hard PathQG questions as Easy or Medium.
- Current Hard questions are often path-dependent but do not reliably require 3+ inference steps from an independent judge's perspective.

---

## 3. Data And Path Filtering Status

Main strict path pilot output:

- `outputs/runs/path_filter_strict_pilot/`

Key files:

| File | Meaning |
|---|---|
| `paths.raw.jsonl` | raw sampled paths |
| `paths.prefiltered.jsonl` | paths after local prefilter |
| `paths.judged.all.jsonl` | all LLM-judged paths with trace fields |
| `paths.filtered.strict.jsonl` | strict main-experiment paths |
| `paths.filtered.relaxed.jsonl` | relaxed paths, including partial Hard paths |
| `paths.rejected.jsonl` | rejected paths |
| `PATH_FILTER_REPORT.md` | path filtering report |
| `MANUAL_AUDIT_SAMPLE.md` | manual audit sample |

Strict final path counts after deduplication:

| Difficulty | Strict paths |
|---|---:|
| Easy | 32 |
| Medium | 35 |
| Hard | 9 |
| Total | 76 |

Important notes:

- Strict Hard currently keeps only `can_write_path_dependent_question=yes` paths.
- Hard partial paths are saved in relaxed outputs but are not mixed into strict main experiments.
- The strict Hard pool is too small for final paper-scale claims.
- Hard scarcity reflects MAVEN-ERE path limitations plus strict path-dependency requirements.

---

## 4. Answer Phrase Extraction Status

Resolved issues:

- Fixed overly wide clause extraction.
- Added checks for dangling end words and phrases.
- Added checks for unclosed brackets/quotes.
- Added fragment starter rejection, such as bare participles or modal fragments.
- Added list-aware handling for `and` / `or` so coordinated phrases are not always cut too early.

Validation:

- Regression tests passed after the answer phrase fixes.
- Known bad phrase pattern such as `was released in VHS titled` is now marked partial and filtered.

Remaining risk:

- Answer phrase extraction is still heuristic.
- Some phrases may be syntactically complete but semantically weak.
- Manual audits should continue sampling generated answers, not only questions.

---

## 5. QG Pilot Status

Main QG pilot on all available strict paths:

- Run directory: `outputs/runs/qg_pilot_strict_100_per_level/`
- Despite the name, this used all available strict paths: 32 Easy, 35 Medium, 9 Hard.

Filter pass results:

| Difficulty | Pass | Rate |
|---|---:|---:|
| Easy | 13/32 | 41% |
| Medium | 18/35 | 51% |
| Hard | 6/9 | 67% |
| Total | 37/76 | 49% |

Hard-specific observations:

- Hard path coverage works better after structured semantic path coverage judging.
- 8/9 Hard questions passed path coverage in the pilot.
- Remaining Hard failures are mostly answer consistency or generation quality issues, not filter exceptions.

Solver result on filter-passing questions:

| Difficulty | Correct | Rate |
|---|---:|---:|
| Easy | 1/13 | 8% |
| Medium | 3/18 | 17% |
| Hard | 1/6 | 17% |
| Total | 5/37 | 14% |

Interpretation:

- Solver accuracy is low and should be treated as auxiliary for now.
- Solver correctness should not be used as the primary evidence for difficulty control.

---

## 6. Baseline Alignment Pilot

Run directory:

- `outputs/runs/baseline_alignment_pilot/`

All methods used the same 76 strict paths and the same v3 question filter.

| Method | Valid Generated | Filter Pass | Hard Pass |
|---|---:|---:|---:|
| PathQG-HardAware | 73/76 | 36/76 | 6/9 |
| ZeroShot-TargetQG | 44/76 | 11/76 | 0/9 |
| ICL-TargetQG | 42/76 | 7/76 | 0/9 |
| SelfRefine | 41/76 | 8/76 | 0/9 |

Interpretation:

- PathQG-HardAware leads on valid generation and filter pass rate.
- PathQG-HardAware is the only method producing filter-passing Hard questions in this pilot.
- This is evidence for constrained generation yield and path-groundedness.
- This is not enough by itself to claim independent difficulty controllability.

Baseline designs:

| Method | Design |
|---|---|
| PathQG-HardAware | Uses context, target answer/final event, event path, relation sequence, and difficulty-aware prompt/repair. |
| ZeroShot-TargetQG | Uses context, target answer, and target difficulty. Does not see event path or relation sequence. |
| ICL-TargetQG | Same as ZeroShot, plus few-shot examples. Does not see event path or relation sequence. |
| SelfRefine | ZeroShot generation followed by critique/revision. Does not see event path or relation sequence. |

---

## 7. Independent Difficulty Evaluation

Run directory:

- `outputs/runs/independent_difficulty_eval_pilot/`

Purpose:

- Evaluate actual generated question difficulty without using the internal filter as the difficulty metric.
- Difficulty-only judge does not see target difficulty, method name, solver answer, or solver correctness.
- Path-dependency judge separately checks whether the question depends on prior path events.

Results:

| Metric | PathQG-HardAware | ZeroShot | ICL | SelfRefine |
|---|---:|---:|---:|---:|
| Judged valid questions | 36 | 11 | 7 | 8 |
| Difficulty accuracy | 47% | 73% | 71% | 63% |
| Spearman rho | 0.56 | 0.60 | 0.56 | N/A |
| Path dependency strong | 31% | 9% | 14% | 0% |
| Answerable | 100% | 100% | 100% | 100% |

PathQG-HardAware per-target result:

| Target | N | Pred Easy | Pred Medium | Pred Hard | Accuracy |
|---|---:|---:|---:|---:|---:|
| Easy | 12 | 12 | 0 | 0 | 100% |
| Medium | 18 | 13 | 5 | 0 | 28% |
| Hard | 6 | 4 | 2 | 0 | 0% |

Interpretation:

- PathQG-HardAware has the strongest path dependency.
- However, the independent judge does not rate current PathQG-HardAware Hard questions as Hard.
- Current Medium/Hard questions often collapse to Easy/Medium under independent evaluation.
- This is the central unresolved research problem.

---

## 8. Hard Difficulty Problem

Observed Hard failure mode:

```text
After event A and event B, what happened?
```

This kind of question is path-dependent, but it explicitly reveals much of the reasoning chain. The independent judge often treats it as Easy or Medium because the question tells the reader which events to connect.

Root issue:

- Hop count is not the same as cognitive difficulty.
- Path dependency is not the same as Hard reasoning.
- A 3-hop path can still yield a Medium question if the prompt exposes the chain.
- Many MAVEN-ERE final events are recoverable from a local answer sentence.

---

## 9. Implicit Hard Prompt Experiment

Run directory:

- `outputs/runs/hard_implicit_qg_pilot/`

Goal:

- Avoid explicitly listing the full reasoning chain in Hard questions.
- Encourage implicit chain reasoning so the solver must discover intermediate events from context.

Old vs new Hard prompt result:

| Metric | Old Hard Prompt | New Implicit Hard |
|---|---:|---:|
| Generated | 9 | 9 |
| Filter Pass | 6 | 2 |
| Independent Pred Hard | 0 | 0 |
| Independent Pred Medium | 2 | 1 |
| Independent Pred Easy | 4 | 1 |
| PathDep Strong | 5 | 2 |

Manual handcrafted sanity check:

- 1/9 handcrafted Hard-style questions were judged Hard.
- This shows the judge can assign Hard when the reasoning chain is genuinely hard.

Interpretation:

- The implicit-chain idea is plausible in principle.
- Current LLM generation does not reliably satisfy the constraints.
- The implicit Hard prompt lowers filter pass rate and still does not solve independent Hard difficulty.
- The implicit Hard prompt should remain experimental and should not be treated as the default final method yet.

Implementation caution:

- The original intended constraint was at most one prior trigger explicitly mentioned.
- Current implementation/reporting may be looser in places, allowing up to two prior triggers.
- This should be audited before using the implicit prompt in final experiments.

---

## 10. Primary Metrics

The current primary metrics should be separated into two groups.

### 10.1 Internal Constraint / Yield Metrics

These measure whether the system can generate valid path-grounded questions:

| Metric | Meaning |
|---|---|
| Valid generated rate | Model returned parseable usable output. |
| Filter pass rate | Question passed grammar, answer consistency, path coverage, and shortcut checks. |
| Hard pass rate | Hard-target questions passing all internal filters. |
| Path coverage | Question references or semantically uses required prior events. |
| Path dependency | Independent judge says prior path events are required or strongly useful. |

These support constrained generation claims, not final difficulty-control claims.

### 10.2 Independent Difficulty Metrics

These are needed for the main difficulty-control claim:

| Metric | Meaning |
|---|---|
| Difficulty consistency | Independent predicted difficulty matches target Easy/Medium/Hard. |
| Inference-step consistency | Independent required steps match expected hop level. |
| Solver accuracy by difficulty | Solver correctness should decrease as difficulty increases, but only after solver calibration. |
| Question quality | Answerability, final-event consistency, fluency/relevance if needed. |

Current caveat:

- Solver accuracy is not reliable enough to be the main metric yet.
- Independent difficulty judgment needs human calibration before being final paper evidence.
- Avoid inventing unsupported composite scores unless clearly marked as diagnostic.

---

## 11. Current Bottlenecks

1. Hard difficulty controllability is not solved.
   - Independent Pred Hard is currently 0 for PathQG-HardAware pilot outputs.

2. Strict Hard pool is too small.
   - Only 9 strict Hard paths are available in the current strict pilot.

3. Single-prompt Hard generation is weak.
   - Explicit prompt exposes the chain.
   - Implicit prompt lowers pass rate and still does not yield reliable Pred Hard.

4. Baseline generation has lower valid yield.
   - This helps PathQG on constrained generation, but does not prove difficulty consistency.

5. Solver/Judge pipeline is auxiliary.
   - Useful for answerability and solver behavior, but not currently the primary difficulty-control evidence.

---

## 12. Recommended Next Experiments

Do not keep tuning a single Hard prompt blindly on the same 9 paths.

Recommended next steps:

1. Multi-candidate Hard generation plus independent difficulty selection.
   - Generate K candidates per Hard path, e.g. K=5.
   - Run question filter.
   - Run independent difficulty judge.
   - Select predicted-Hard candidates if available.
   - Report Pred Hard yield and path dependency.

2. Increase Hard path potential.
   - Sample longer paths if possible, e.g. 5+ events.
   - Prefer CAUSE, SUBEVENT, or MIXED relation composition.
   - Prefer paths spanning more supporting sentences.
   - Avoid final events answerable from a single local sentence.

3. Human calibration.
   - Label a small set for answerability, final-event consistency, path dependency, required steps, and human difficulty.
   - Use this to calibrate the independent GPT-4o-mini judge or train a later classifier.

4. Keep claims separated.
   - Claim A: valid path-grounded generation yield.
   - Claim B: independent difficulty consistency.
   - Claim A is currently supported.
   - Claim B still needs more evidence.

---

## 13. Trace Protocol

Every full pipeline run should write trace artifacts:

| Artifact | Purpose |
|---|---|
| `full_trace.jsonl` | Machine-readable trace for every item and stage. |
| `readable_trace.md` | Human-readable inspection file. |
| stage JSONL files | Inputs/outputs at each stage for replay. |
| report markdown | Aggregate tables and interpretation. |

Future diagnosis rule:

```text
Before explaining a failure mode, inspect the relevant trace rows and JSONL examples.
Do not infer causes from aggregate tables alone.
```

Trace should include:

- graph/path fields
- answer phrase and status
- prefilter result and reason
- LLM path judge raw/status fields
- generation prompt/output/repair attempts
- question filter stage outputs
- solver answer
- judge output
- final pass/fail reason

---

## 14. Known Clean Output Directories

| Directory | Use |
|---|---|
| `outputs/runs/path_filter_strict_pilot/` | Main strict/relaxed path filtering pilot. |
| `outputs/runs/qg_pilot_strict_29_v3/` | Stable small QG pilot after path coverage fix. |
| `outputs/runs/qg_pilot_strict_100_per_level/` | All available strict paths QG pilot. |
| `outputs/runs/baseline_alignment_pilot/` | Four-method baseline alignment on same 76 strict paths. |
| `outputs/runs/independent_difficulty_eval_pilot/` | Independent difficulty and path dependency evaluation. |
| `outputs/runs/hard_implicit_qg_pilot/` | Experimental implicit Hard prompt pilot. |
| `outputs/archive/legacy_event_qg_outputs/` | Archived legacy outputs from old framework. |

---

## 15. Rules For Future Reporting

When reporting results, always separate:

- valid generated rate
- filter pass rate
- independent difficulty accuracy
- path dependency
- answerability / final-event consistency
- solver accuracy

Do not report filter pass rate as difficulty accuracy.
Do not expose target difficulty to the independent difficulty judge.
Do not use solver answer or solver correctness as input to the independent difficulty judge.
Do not reintroduce the old four-dimensional difficulty scorer.
Do not depend on `event_qg/` in new code.
Do inspect trace logs before proposing fixes.

---

## 16. Current Next Action

The most promising next action is:

```text
Run multi-candidate Hard generation with independent difficulty selection, preferably on a larger and higher-potential Hard path pool.
```

Minimum experimental design:

1. Build or select a larger Hard candidate path pool.
2. Generate K questions per path.
3. Filter questions.
4. Independently judge difficulty and path dependency.
5. Select best candidates.
6. Manually audit a small subset.
7. Compare against target-aware baselines under the same selection budget.

Success criteria:

- Non-zero and meaningful Pred Hard rate.
- Strong or partial path dependency remains high.
- Answerability and final-event consistency remain acceptable.
- The result is not just higher internal filter pass.
