# DCQG Project Status

**Last updated:** 2026-05-12
**Status:** Stage 1 story-matched evaluation implemented. Smoke test passed (5 stories, 15 candidates, 60 generations). CrossQG evaluation pipeline now supports two selection modes: balanced (original) and story_matched (each story contributes equal Easy/Med/Hard). Story-matched, retry/budget, and similarity diagnostics added to report. Next: full story-matched run with 106 stories, then inspect results before Stage 2/3.

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
| `outputs/runs/evidence_necessity_audit_*/` | Evidence necessity audit outputs. |

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

The immediate next action is:

```text
1. Design evidence-role-aware QG prompts for narrative domain.
2. Use implicit/causal/motivation QA pairs as Hard generation targets.
3. Build QG pipeline that takes evidence-chain annotations and generates Hard questions.
4. Target pool: 2166 implicit train QA pairs, 15.2% verified Hard rate.
```

Commands:

```powershell
# Already completed:
python -m scripts.run_fairytale_evidence_audit --split validation --filter_implicit --limit 200 --output_dir outputs/runs/fairytale_evidence_audit_implicit_200_20260510
python -m scripts.run_fairytale_evidence_audit --split train --filter_implicit --limit 500 --output_dir outputs/runs/fairytale_evidence_audit_train_implicit_500_20260510
```

---

## 14. Hard Rescue Pilot (2026-05-06)

**Goal:** Verify that multi-strategy Hard generation can produce non-zero Pred Hard.

**Approach:**
- Expanded Hard path pool: strict (9) + relaxed (22) = 31 paths after dedup
- 5 generation strategies: hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
- K=2 candidates per path per strategy (10-path smoke test)
- Independent difficulty judge (gpt-4o-mini via AIHUBMIX)
- Updated judge prompt to emphasize chain reasoning over information counting

**Key code changes:**
- `dcqg/generation/prompts.py`: 4 new Hard prompt functions (hidden_endpoint, relation_composition, contrastive, missing_bridge) + `_extract_anchors` helper
- `dcqg/generation/generator.py`: `generate_multi_strategy()` with strategy dispatch
- `dcqg/evaluation/judge.py`: `independent_difficulty_judge()` and `independent_path_dependency_judge()` functions
- `dcqg/question_filter/grammar.py`: added "given", "between", "if", "suppose", "assuming" to allowed starters
- `dcqg/path/direction.py`: relaxed `validate_hard_question` from 2+ to 1+ prior events
- `scripts/run_hard_rescue_pilot.py`: full orchestrator script

**Smoke test results (10 paths, K=2, 100 candidates):**

| Metric | Value |
|--------|------:|
| Pred Hard (candidate-level) | 24/98 (24.5%) |
| Pred Medium | 63/98 (64.3%) |
| Pred Easy | 11/98 (11.2%) |
| Paths with >= 1 Pred Hard | 8/10 (80%) |
| Paths with Pred Hard + ans + fec + pathdep | 6/10 (60%) |
| Answerable | 98/98 (100%) |
| Final-Event Consistent | 78/98 (80%) |
| PathDep Strong | 41/98 (42%) |

**Per-strategy comparison:**

| Strategy | Pred Hard | Pred Med | Pred Easy |
|----------|----------:|---------:|----------:|
| hidden_endpoint | 45% | 55% | 0% |
| missing_bridge | 45% | 45% | 10% |
| relation_composition | 26% | 63% | 11% |
| implicit_chain | 5% | 84% | 11% |
| contrastive | 0% | 75% | 25% |

**Key findings:**
1. hidden_endpoint and missing_bridge are the most effective Hard strategies (45% Pred Hard each).
2. contrastive strategy is ineffective (0% Pred Hard).
3. The judge prompt was critical: emphasizing "reasoning steps" vs "information count" changed Pred Hard from 0% to 24.5%.
4. The old implicit_chain strategy only achieves 5% Pred Hard.
5. Filter pass rate is 0% because `--skip_llm_filters` was used in smoke test. Full run with LLM filters needed.

**Success criteria evaluation:**
- Pred Hard > 0: YES (24.5%)
- >= 20% paths produce Pred Hard + quality candidates: YES (60%)
- 3-level difficulty-control claim: VIABLE pending full-scale validation

**Next step:** Run full pilot (31 paths, K=5, LLM filters enabled) to validate at scale.

---

## 15. Full Hard Rescue Pilot (2026-05-06)

**Goal:** Validate smoke test results at scale with LLM filters enabled.

**Configuration:**
- 22 Hard paths (strict + relaxed, after dedup)
- 5 strategies x K=5 candidates = 550 total candidates
- LLM filters enabled (not skipped)
- Independent difficulty judge (gpt-4o-mini via AIHUBMIX)

**Run directory:** `outputs/runs/hard_rescue_pilot_20260506_040642/`

**Pool statistics:**

| Stage | Count |
|-------|------:|
| Selected Hard paths | 22 |
| Total candidates | 550 |
| Grammar pass | 505 |
| Generation errors | 21 |
| Filter pass | 65 |

**Candidate-level difficulty prediction:**

| Metric | All Candidates | Filter-Passing Only |
|--------|---------------:|--------------------:|
| Judged | 529 | 65 |
| Pred Hard | 124 (23.4%) | 5 (7.7%) |
| Pred Medium | 346 (65.4%) | 47 (72.3%) |
| Pred Easy | 59 (11.2%) | 13 (20.0%) |

**Path-level Pred Hard yield:**

| Metric | Count | Rate |
|--------|------:|-----:|
| Paths with >= 1 Pred Hard | 17 | 77.3% |
| Paths with Pred Hard + answerable | 17 | 77.3% |
| Paths with Pred Hard + ans + fec + pathdep | 13 | 59.1% |

**Per-strategy comparison:**

| Strategy | N judged | Pred Hard | Pred Med | Pred Easy | Ans% | FEC% | PathDep Strong% | Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|
| hidden_endpoint | 105 | 46 (44%) | 53 (50%) | 6 (6%) | 100% | 81% | 33% | 10% |
| missing_bridge | 108 | 54 (50%) | 46 (43%) | 8 (7%) | 100% | 88% | 42% | 4% |
| relation_composition | 108 | 11 (10%) | 86 (80%) | 11 (10%) | 100% | 66% | 42% | 9% |
| implicit_chain | 104 | 9 (9%) | 84 (81%) | 11 (11%) | 99% | 61% | 39% | 18% |
| contrastive | 104 | 4 (4%) | 77 (74%) | 23 (22%) | 100% | 70% | 36% | 18% |

**Filter pass rate:** 65/550 (11.8%)

**Filter fail reasons (top):**
- path_coverage=covers 1 prior events, need >= 2: 312 (57%)
- answer_consistency=no (various): 104+ (19%+)
- path_coverage=covers 0 prior events: 60 (11%)

**Quality metrics:**

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable | 528/529 (100%) | 65/65 (100%) |
| Final-Event Consistent | 387/529 (73%) | 56/65 (86%) |
| PathDep Strong | 203/529 (38%) | 26/65 (40%) |
| Single-Sent Answerable=no | 126/529 (24%) | 5/65 (8%) |

**Key findings:**
1. Pred Hard rate held at scale: 23.4% (vs 24.5% smoke test).
2. missing_bridge (50%) overtook hidden_endpoint (44%) as top strategy.
3. LLM filters reduce candidates to 11.8% pass rate — path coverage is the biggest blocker (312/550 fail on "covers 1 prior events, need >= 2").
4. Only 5 filter-passing candidates are Pred Hard (7.7%), but 3 are genuine high-quality samples.
5. Path yield (59.1%) confirms the smoke test finding (60%).
6. contrastive remains ineffective (4% Pred Hard).

**Success criteria evaluation:**
- Pred Hard > 0: YES (23.4% all, 7.7% filter-passing)
- >= 20% paths produce Pred Hard + quality candidates: YES (59.1%)
- Filter-passing Pred Hard samples exist: YES (3 samples with ans + fec + pathdep)
- 3-level difficulty-control claim: VIABLE

**2026-05-10 superseding note:** Later QG-eligible Hard rescue runs showed that this optimistic interpretation does not hold under blind quality-only evaluation. In `outputs/runs/hard_rescue_qg_eligible_20path_20260509/`, 0/112 judged candidates and 0/21 quality-pass candidates were judged Hard. The current conclusion is not to abandon difficulty-controlled QG, but to replace raw event-path hop count with necessity-verified evidence paths. See `review-stage/DIRECTION_UPDATE_2026-05-10.md`.

**Bottleneck:** path_coverage filter requires 2+ prior events referenced, but many Hard questions intentionally hide events. This is a design tension — the filter was designed for explicit-path questions, not implicit-chain questions. Consider relaxing path_coverage for Hard questions or adding a Hard-specific path coverage check.

---

## 17. Evidence Necessity Audit (2026-05-10)

**Goal:** Audit validation documents to determine if enough truly Hard evidence candidates exist, where the answer cannot be found from a single sentence alone.

**Motivation:** The event-hop hypothesis failed because 3-hop event paths still produced questions where the answer was locally identifiable. The new hypothesis: Hard = 3+ necessary evidence steps required to identify the answer. Before redesigning QG prompts, we need to verify that such candidates exist in the data.

**Implementation:**
- Core module: `dcqg/path/evidence_necessity.py`
- Audit script: `scripts/run_evidence_necessity_audit.py`
- Uses LLM (JUDGE_MODEL) to assess evidence necessity for each event candidate
- Single LLM call per document (batches all candidates)
- Output: JSONL candidates + markdown audit report

**New difficulty definitions (evidence-based):**

| Level | Evidence requirement |
|---|---|
| Easy | Answer sentence alone is sufficient. |
| Medium | Requires anchor sentence + answer sentence (2 evidence steps). |
| Hard | Requires anchor + bridge/disambiguator + answer; answer sentence alone is insufficient. |

**Output fields per candidate:**
- `answer_sentence_id`, `anchor_sentence_ids`, `bridge_sentence_ids`
- `evidence_span`, `num_required_sentences`
- `answer_locality`: single_sentence / two_sentence / multi_sentence
- `reasoning_operation`: bridge / contrast / temporal_order / causal_chain / disambiguation / comparison
- `answer_sentence_alone_sufficient`: yes / partial / no
- `evidence_necessity`: weak / partial / strong
- `evidence_necessity_reason`
- `evidence_difficulty`: Easy / Medium / Hard (derived from above)

**Smoke test result (2 documents, 3 candidates each):**

| Metric | Value |
|---|---:|
| Documents | 2 |
| Total candidates | 6 |
| Easy | 0 |
| Medium | 2 |
| Hard | 4 |
| Hard (alone_sufficient=no) | 4 |
| Assessment status OK | 6/6 |

**Output directory:** `outputs/runs/evidence_necessity_audit_YYYYMMDD/`

**Success criteria:**
- Hard evidence candidates >= 100: proceed to evidence-role-aware QG prompt design.
- Hard evidence candidates < 100: need to change data construction or answer types.

**Important constraints:**
- Do not use predicted difficulty as a filtering threshold.
- Difficulty judge is for audit/evaluation only, not for main data filtering.
- Do not treat graph hop count as Hard label directly.
- Event graph can propose candidate evidence chains, but must pass evidence necessity verification.

---

## 18. FairytaleQA Evidence Audit (2026-05-10)

**Goal:** Test whether FairytaleQA (narrative QA) produces more verified Hard evidence candidates than MAVEN-ERE (event-based QA).

**Motivation:** MAVEN-ERE evidence audit showed 0/112 blind Hard candidates because answer sentences are locally identifiable. Hypothesis: narrative QA with character motivation, causal chains, and multi-sentence inference may be a better domain for difficulty-controlled QG.

**Implementation:**
- Branch: `fairytale-evidence-audit`
- Loader: `dcqg/datasets/fairytaleqa_loader.py` (HuggingFace CSV via `hf_hub_download`)
- Core module: `dcqg/path/fairytale_evidence_audit.py`
- Audit script: `scripts/run_fairytale_evidence_audit.py`

**FairytaleQA-specific assessment fields:**
- `answer_sentence_alone_sufficient`: yes / partial / no
- `section_evidence_sufficient`: yes / partial / no
- `full_context_needed`: yes / partial / no
- `required_evidence_sentences`, `bridge_sentence_ids`, `num_required_sentences`
- `reasoning_operation`: explicit_lookup / temporal_order / causal_chain / motivation / character_state / disambiguation / summary_inference / contrast
- `bridge_removal_effect`: none / harder / ambiguous / unanswerable
- `necessity_type`: background_context / answer_identification / disambiguation / causal_bridge / temporal_bridge / motivation_bridge / summary_synthesis

**Hard classification (conservative):** Requires ALL of:
- answer_sentence_alone_sufficient = no
- num_required_sentences >= 3
- bridge_removal_effect in {ambiguous, unanswerable}
- necessity_type in {answer_identification, disambiguation, causal_bridge, temporal_bridge, motivation_bridge, summary_synthesis}

**200-sample result (validation split):**

| Metric | Value |
|---|---:|
| QA pairs loaded | 200 |
| Candidates assessed | 200 |
| Easy | 156 (78.0%) |
| Medium | 36 (18.0%) |
| Hard | 8 (4.0%) |
| Parse failures | 0 |
| Contradictions fixed | 15 |
| Invalid sentence IDs | 0 |
| Hard validation violations | 0 |
| Assessment OK | 200/200 |

**Hard cross-tab patterns (200-sample):**

| Dimension | Hard rate | Hard count / Total |
|---|---:|---:|
| implicit | 17.5% | 7/40 |
| explicit | 0.6% | 1/160 |
| summary | 9.5% | 2/21 |
| local | 3.4% | 6/179 |
| causal relationship | 6.7% | 3/45 |
| action | 4.6% | 3/65 |
| feeling | 4.0% | 1/25 |

**Hard by reasoning operation:**
- causal_chain: 4
- summary_inference: 3
- disambiguation: 1

**Hard by necessity type:**
- causal_bridge: 4
- summary_synthesis: 3
- disambiguation: 1

**Comparison against MAVEN-ERE:**
- MAVEN-ERE: 0% verified Hard rate (0/112 blind Hard)
- FairytaleQA: 4.0% verified Hard rate (8/200)
- Narrative QA is more promising for difficulty-controlled QG
- `implicit` questions are 29x more likely to be Hard than `explicit`

**Output directory:** `outputs/runs/fairytale_evidence_audit_YYYYMMDD/`

**Success criteria:**
- Verified Hard > 0: PASS (8 found)
- Verified Hard >= 10: NEAR MISS (8/10, implicit questions fill the gap)
- Hard rate >= 5%: NEAR MISS (4.0%, implicit subset is 17.5%)

**Targeted audit (implicit filter):**

| Run | Split | Pool | Audited | Hard | Hard rate |
|---|---|---:|---:|---:|---:|
| validation implicit | val | 281 | 200 | 29 | 14.5% |
| train implicit | train | 2166 | 500 | 76 | 15.2% |

**Hard by attribute (train implicit 500):**
- causal relationship: 41 (54%)
- action: 13 (17%)
- feeling: 12 (16%)
- prediction: 7 (9%)
- character: 2 (3%)
- outcome resolution: 1 (1%)

**Hard by question prefix (train implicit 500):**
- why did: 34 (45%)
- how did: 16 (21%)
- what will: 5 (7%)
- how will: 4 (5%)
- why was: 6 (8%)

**Hard by necessity type (train implicit 500):**
- motivation_bridge: 34 (45%)
- causal_bridge: 18 (24%)
- summary_synthesis: 17 (22%)
- disambiguation: 7 (9%)

**Success criteria (targeted):**
- Validation implicit Hard rate >= 10%: PASS (14.5%)
- Train implicit Hard count >= 30: PASS (76)
- Hard examples mostly causal/motivation/summary: PASS (91% are motivation_bridge + causal_bridge + summary_synthesis)
- Hard candidate pool: 2166 implicit train QA pairs available for QG
- Next: design evidence-role-aware QG prompts targeting implicit/causal/motivation types.

---

## 19. Narrative Evidence Graph Audit (2026-05-10)

**Goal:** Validate that narrative evidence graphs can be reliably extracted from FairytaleQA Hard candidates, to serve as scaffolding for difficulty-controlled QG.

**Implementation:**
- Core module: `dcqg/graph/narrative_graph.py`
- Audit script: `scripts/run_narrative_graph_audit.py`
- Graph schema: 12 node types, 10 edge relations, 4 evidence roles
- Validation: 9 structural checks + post-processing (auto-label bridge, retry on empty)

**Graph schema:**

Node types: action, state, emotion, goal, motivation, belief, outcome, consequence, problem, attempt, resolution, description

Edge relations: temporal_before, causes, motivates, explains, results_in, enables, prevents, contrasts_with, same_character, supports_inference

Evidence roles: anchor, bridge, answer, context

**Smoke test (20 Hard samples, train implicit):**

| Metric | Target | Actual | Status |
|---|---|---:|---|
| parse_ok | >= 95% | 100% (20/20) | PASS |
| graph_valid | >= 80% | 95% (19/20) | PASS |
| Required evidence coverage | >= 90% | 100% (65/65) | PASS |
| Bridge sentence coverage | >= 90% | 84.8% (28/33) | NEAR MISS |
| Answer node coverage | - | 95% (19/20) | PASS |
| Avg nodes | >= 3 | 3.2 | PASS |
| Avg edges | >= 2 | 2.2 | PASS |

**Graph structure:**
- Most common node type: action (34%), state (28%), description (14%)
- Most common edge relation: causes (33%), temporal_before (31%), results_in (16%)
- Edge necessity: 93% strong, 7% partial

**Failure analysis:**
- 1/20 invalid: LLM failed to assign answer-role node (all nodes labeled anchor/bridge/context)
- Bridge coverage gap: some bridge sentences have answer-role nodes (dual-purpose), validation relaxed to accept this

**Recommendation:** Schema is stable enough for QG pilot. Next step: use narrative graphs as scaffolding for evidence-role-aware Hard question generation.

---

## 20. Story-Matched CrossQG Evaluation (Stage 1, 2026-05-12)

**Goal:** Make evaluation fair by ensuring each selected story contributes the same number of Easy/Medium/Hard candidates, preventing story/context confounds in difficulty comparisons.

**Implementation:**
- Script: `scripts/run_crossqg_eval.py`
- Added `--selection_mode` arg: `balanced` (original) or `story_matched` (new)
- Added `--candidates_per_level_per_story` arg (default 1)
- Added `--max_stories` arg (optional)
- New function: `_select_story_matched_candidates()`

**Story-matched candidate selection:**
1. Group candidates by `story_name` and `evidence_difficulty`
2. Keep only stories with >= 1 Easy, >= 1 Medium, >= 1 Hard
3. For each eligible story, select exactly 1 Easy, 1 Medium, 1 Hard (configurable)
4. Selection is deterministic under seed, with preference scoring within level
5. Candidate pool: 228 stories → 106 eligible → ~318 candidates (106 × 3)

**Candidate preference within story:**
- Easy: prefer `answer_sentence_alone_sufficient=yes` and `num_required_sentences=1`
- Medium: prefer `num_required_sentences=2`
- Hard: prefer `necessity_type` in {motivation_bridge, causal_bridge, summary_synthesis} and non-temporal reasoning

**New diagnostics added to report:**
- Section 13: Story-Matched Diagnostics (story-level accuracy, win/tie/loss, story Spearman, per-story failures)
- Section 14: Retry & Budget Diagnostics (attempts, repair usage, self-check failure rate, retry/failure reasons)
- Section 15: Similarity Diagnostics (lexical similarity between levels, evidence overlap, difficulty collapse counts)
- Section 11: Prompt Audit (difficulty judge is blind - confirmed target difficulty NOT in judge prompts)
- Section 8: Success Criteria (story-matched specific: stories >= 100, equal per-story counts, identical denominator)

**Retry policy:**
- Allowed triggers: JSON parse failure, empty output, degenerate output, invalid format, no question mark, answer leakage, grammar failure, Ours graph_policy self-check failure
- Forbidden triggers: `predicted_difficulty != target_difficulty` from final difficulty judge, low difficulty accuracy, HRP-v2 failure
- If all retries fail: keep row, count as generation failure in end-to-end metrics

**Similarity diagnostics (diagnostic only, no filtering):**
- Easy-Medium, Medium-Hard, Easy-Hard character 4-gram Jaccard similarity
- Evidence sentence overlap among judge-used evidence
- Difficulty collapse counts (all 3 questions predicted as same difficulty)
- Collapse-to-Medium count

**Smoke test results (5 stories, 15 candidates, 60 generations):**
- Output: `outputs/runs/fairytale_qg_crossqg_eval_20260512_story_matched_smoke/`
- All 5 stories have equal Easy/Med/Hard count: PASS
- All methods have identical denominator (15 each): PASS
- Story-matched diagnostics present: PASS
- Similarity diagnostics present: PASS
- Retry/budget diagnostics present: PASS
- Difficulty judge blind audit: PASS (target difficulty NOT in prompts)
- Ours graph_policy fields populated: PASS

**Expected full run (106 stories):**
- 106 stories × 3 levels = 318 candidates
- 318 × 4 methods = 1272 generations
- Output directory: `outputs/runs/fairytale_qg_crossqg_eval_20260512_story_matched_v1/`

**Next:** Run full story-matched evaluation (106 stories). Stop after Stage 1 full report. Do not implement Stage 2/3 until results are inspected.
