# DCQG Experiment Design Review (Difficulty-Controlled QG Focus)

**Date**: 2026-05-06
**Reviewer**: GPT-5.5 (xhigh reasoning via Codex MCP)
**Review Type**: Multi-round focused review (3 rounds)
**Constraint**: Research direction MUST remain difficulty-controlled question generation

---

## Round 1: Full Experiment Design Audit

### Bottom Line
The current plan is NOT sufficient for a difficulty-controlled QG claim. It currently supports a weaker claim: path constraints increase path dependency, but not that hop count controls perceived difficulty. The pilot's "Hard = 0% judged Hard" result is fatal unless fixed or the claim is narrowed.

---

### Q1: Experiment Design Sufficiency

Missing minimum evidence:
1. **Human-validated difficulty by level.** LLM judge alone is not enough.
2. **Monotonicity test:** Easy < Medium < Hard in human/solver difficulty after controlling for answerability.
3. **Causal path ablations:** same document/target event, different path lengths.
4. **Doc-matched comparison:** compare 1/2/3-hop questions from the same documents.
5. **Pre/post filter reporting:** otherwise the filter may be doing the "difficulty control."
6. **Hard-path feasibility analysis:** 9 strict Hard paths is too few.

Minimum viable experiment set:
- Generate from matched 1/2/3-hop paths
- Include path-guided vs no-path vs wrong-path/random-path
- Evaluate with human audit + at least two independent LLM judges
- Report answerability-gated difficulty
- Show monotonic increase in difficulty and path dependency with hop count

---

### Q2: Baseline Fairness

| Baseline / Ablation | Status | Why |
|---|---|---|
| Context-only, no path | MUST | true no-structure baseline |
| Path + context | MUST | full method |
| Path-only, no context | MUST | shows path alone is insufficient / hallucination risk |
| Relation-type only | MUST | tests whether ordered path matters beyond relation labels |
| Random valid path same document | MUST | tests whether any path helps |
| Wrong-difficulty path | MUST | causal test: does 3-hop path make harder Q than 1-hop under same target label? |
| Shuffled path order | NICE | tests ordered structure |
| Same endpoints, different path length | NICE but strong | best causal evidence if available |
| CoT baseline | NICE | reviewer may ask whether reasoning prompt alone creates difficulty |
| Retrieval-augmented baseline | NICE | only if documents are long |

---

### Q3: Metric Definitions

**Answerability:** Binary. A question is answerable if a competent annotator/model can identify a unique answer from the document without external knowledge.

**Target difficulty hit rate:**
Among answerable, fluent, non-leaking questions:
```
HitRate_d = (1/N_d) * sum_i 1[d_hat_i == d_i]
```
Also report within-one accuracy and confusion matrix.

**Solver accuracy:** Use only on answerable questions. Lower solver accuracy means harder ONLY if answerability and clarity are held constant. Report:
- SolverAcc_easy > SolverAcc_medium > SolverAcc_hard
- Confidence intervals
- Separate "wrong because unanswerable/ambiguous" from "wrong despite answerable"

**Path dependency (0-3 scale):**
- 0 = no path needed
- 1 = weak: path helps but shortcut exists
- 2 = strong: answer requires at least two path relations
- 3 = exact: requires all specified relations in order
Report strong+exact dependency rate.

**Shortcut rate:** Percentage of questions answerable without the intended path, using lexical overlap, sentence-local clue, entity prior, or answer leakage.

**Difficulty alignment:** Combine target hit + monotonicity: per-level hit rate, macro accuracy, ordinal correlation, monotonic solver trend.

---

### Q4: Proving Event Paths Are Difficulty Signals

Key experiments:
1. **Matched-hop experiment:** same document, 1/2/3-hop paths, show judged difficulty increases with hop count
2. **Wrong-path intervention:** Easy label + 3-hop path vs Hard label + 1-hop path. If judged difficulty follows path more than label, strong evidence
3. **Random-path control:** random same-document path should reduce answerability/path dependency
4. **Relation-only vs ordered path:** if relation-only performs similarly, ordered path is not the real signal
5. **Regression analysis:** predict judged difficulty from hop count while controlling for confounds

---

### Q5: LLM Judge Circularity

Minimum cross-judge setup:
- Generator: Qwen2.5-7B
- Judges: one Qwen-family judge, one non-Qwen judge, one human subset
- Solver: non-Qwen model if possible
- 300-600 judged items
- Report agreement: judge-human accuracy, Cohen/Fleiss kappa

If judges disagree, human labels decide the main difficulty result.

---

### Q6: Must-Do vs Nice-to-Have

**MUST-DO:**
- Human difficulty audit
- Answerability-gated difficulty metrics
- Confusion matrix per level
- Matched 1/2/3-hop experiment
- Context-only baseline
- Random-path baseline
- Wrong-difficulty/path-label conflict test
- Relation-type-only ablation
- Pre/post filter statistics with reject reasons
- Cross-family judge
- Statistical significance / confidence intervals

**NICE-TO-DO:**
- CoT baseline
- Retrieval baseline
- Second generator size: Qwen2.5-14B
- Another 7B model
- Second dataset
- Distilled judge
- Same-endpoint different-length paths
- Shuffled path-order ablation
- Extensive human evaluation beyond 150-300 items

---

### Q7: Experiment Packages

**Minimum submittable package:**
- 600 generated questions: 3 levels x 200, balanced after filtering
- Methods: Direct/context-only, ICL, Self-refine, Full PathQG, RelationType, RandomPath, WrongDifficultyPath
- Human audit: 150 items, stratified by method/level
- LLM judges: Qwen2.5-32B + one non-Qwen judge
- Solver: one non-Qwen model
- Report: answerability, difficulty confusion matrix, solver accuracy, path dependency, shortcut rate, filter rejection

Supports claim only if:
- PathQG has higher macro difficulty hit than baselines
- Hard has nonzero and preferably clearly higher Hard judgment than baselines
- Easy > Medium > Hard solver accuracy among answerable items
- Path dependency increases without unacceptable answerability loss

**Enhanced package:**
- 1,200-1,800 generated questions
- Two generators: Qwen2.5-7B + another 7B model
- Qwen2.5-14B extension
- Second event-relation dataset
- Same-document matched path triples
- Shuffled-path and same-endpoint ablations
- Human audit 300-450 items
- Regression analysis over path/document/question features

---

### Q8: Claims-to-Evidence Matrix

| Outcome | Allowed Claim | Forbidden Claim |
|---|---|---|
| Hop count monotonic in human + solver eval | Event-path hop count is an effective difficulty-control signal | None, if answerability is stable |
| PathQG improves path dependency but not difficulty | Path constraints improve structural grounding | Path controls difficulty |
| 1-hop/2-hop work, 3-hop fails | Event paths support coarse Easy/Medium control | Full Easy/Medium/Hard control |
| Hard questions mostly unanswerable | Method creates complex but unreliable questions | Method generates hard answerable questions |
| Difficulty follows prompt label, not path | Prompting controls difficulty more than paths | Event paths are the main signal |
| Difficulty follows path in label-conflict test | Strong evidence that paths causally influence difficulty | Claiming this without human validation |
| RelationType equals Full PathQG | Relation semantics help difficulty/structure | Ordered event paths are necessary |
| RandomPath performs similarly to Full | Path signal is not specific | Event graph structure is effective |
| Cross-judge disagreement high | Difficulty evaluation is unstable; use human labels | Strong automatic evaluation claim |
| Human labels show Hard never Hard | Current hypothesis is not supported | Any 3-level difficulty-control claim |

---

## Round 2: Hard Generation Strategies + Path-Label Conflict Design

### 5 Concrete Hard Generation Strategies

#### Strategy 1: Hidden-Path Endpoint Question
Use the path during generation, but forbid mentioning intermediate events. The question gives only a weak anchor and asks for the endpoint.

Template:
```
Generate one HARD question whose answer is E4.
Constraints:
- Do NOT mention E2 or E3.
- Do NOT mention the trigger words of E4.
- The question must require using the chain E1 -> E2 -> E3 -> E4.
- Do NOT ask "what happened after..." or list the path events.
```

Bad: "After the protest and the arrest, what happened?"
Better: "Which later action resulted from the earlier confrontation described in the report?"

#### Strategy 2: Relation-Composition Question
Ask about the composed relation created by the path, not the sequence.

Template:
```
The question should ask about E4, but frame it through the combined meaning of the path.
- Prefer causal, enabling, preventing, motivating, or consequence wording.
- If relations are only temporal, ask about the event that resolves the situation.
```

Example: "Which event was the eventual consequence of the initial attack?"

#### Strategy 3: Contrastive Path Question
Generate a plausible distractor event, then ask a question where only the true path endpoint fits.

Template:
```
Write one question whose answer is E4, not D (distractor).
The question must make D look plausible unless the solver follows the event path.
```

Example: "Which response was connected to the earlier security incident, rather than to the separate diplomatic meeting?"

#### Strategy 4: Missing-Bridge Question
Ask for an intermediate bridge event that explains why two distant events are related.

Example: "What intervening development explains how the initial investigation became linked to the later resignation?"

#### Strategy 5: Generate-Then-Rank Hard Selection
Generate 5-10 candidates per path using different hard modes, then select with a judge that penalizes path mention and shortcuts.

For Hard: generate 8, keep top 1.

---

### Path-Label Conflict Experiment Design

**3x3 factorial design:**

| Prompt Label | 1-hop Path | 2-hop Path | 3-hop Path |
|---|---|---|---|
| Easy | normal | conflict | conflict |
| Medium | conflict | normal | conflict |
| Hard | conflict | conflict | normal |

Minimum: 50 items per cell = 450 questions. Better: 100 per cell = 900 questions.

**2x2 reduced version (if compute tight):**

| Prompt Label | 1-hop Path | 3-hop Path |
|---|---|---|
| Easy | normal | conflict |
| Hard | conflict | normal |

Minimum: 100 per cell = 400 questions.

**Metrics (answerable questions only):**
- Judged difficulty distribution
- Target-label hit rate
- Path-hop hit rate
- Ordinal difficulty score
- Solver accuracy by condition
- Path dependency, shortcut rate, answerability

**Main analysis:**
```
JudgedDifficulty ~ PromptLabel + PathHop + PromptLabel:PathHop + controls
```

**Strong evidence FOR hypothesis:**
1. Same prompt label: judged difficulty increases as path hop increases
2. In conflict cells: path beats label (Easy+3hop harder than Hard+1hop)
3. PathHop has significant positive effect in regression
4. Solver accuracy decreases with path hop while answerability stable

**Strong evidence AGAINST:**
- Judged difficulty follows prompt label, not hop count
- Easy+3hop remains Easy
- Hard+1hop is judged harder than Easy+3hop
- 3-hop only lowers answerability, not valid difficulty

**Matched-hop experiment:**
Best: same document, same start event, all 1/2/3-hop paths
Acceptable: same document, same start event, two of three hops + mixed-effects regression
Fallback: same document, matched start events with similar properties

Report coverage honestly: number of documents with full triples, partial triples, usable paths per level.

---

## Round 3: Minimum Viable Story + Paper Outline

### Minimum Viable Difficulty-Control Story

The absolute floor:

| Requirement | Absolute Floor |
|---|---:|
| Answerability after filtering | >= 70% |
| Easy judged Easy | >= 70% |
| Medium judged Medium-or-Hard | >= 50% |
| Hard judged Hard | >= 20-25% |
| Hard judged Easy | <= 50% |
| PathHop coefficient in conflict experiment | positive, significant, beta >= 0.25-0.30, p < 0.05 |
| Easy+3hop harder than Easy+1hop | significant |
| Hard+3hop harder than Hard+1hop | significant |
| PathQG path dependency vs no-path baseline | at least +15-20 points |
| Shortcut rate for Hard | below 50%, ideally below 40% |

**Minimally acceptable difficulty distribution:**

| Target | Judged Easy | Judged Medium | Judged Hard |
|---|---:|---:|---:|
| Easy / 1-hop | 70-80% | 15-25% | 0-5% |
| Medium / 2-hop | 35-50% | 40-50% | 5-15% |
| Hard / 3-hop | 35-50% | 25-40% | 20-30% |

If Easy/Medium works but Hard mostly fails, survive with:
> "Event paths support coarse-grained difficulty control, with reliable Easy/Medium separation and partial Hard control."

If Hard hit remains 0-10% even after improved prompting: make a 2-level paper (Easy vs Non-Easy).

---

### Paper Outline (Difficulty-Control Framing)

**Title:** Event-Path Hop Count as a Structural Signal for Difficulty-Controlled Question Generation

**1. Introduction (0.8 pages)**
Current QG systems control style/topic but not difficulty. Introduce event paths as document-internal structural signals that regulate required reasoning depth. Difficulty should be controlled by required reasoning structure, not prompt labels.

Fig 1: document event graph with 1/2/3-hop paths and generated Easy/Medium/Hard questions.

**2. Task Definition (0.6 pages)**
Define difficulty-controlled event-centric QG. Difficulty = required event-relation reasoning depth. Evaluation uses independent perceived difficulty and solver behavior.

Tab 1: formal definitions of Easy/Medium/Hard, answerability, path dependency, shortcut.

**3. Method: Event-Path-Guided QG (1.2 pages)**
Graph construction, path sampling, filtering, generation prompts (including improved Hard strategies: hidden-path, contrastive, bridge, generate-rank), post-generation filters.

Fig 2: pipeline diagram.

**4. Experimental Design (0.8 pages)**
Dataset, generators, baselines, ablations, evaluation protocol. Emphasize: design separates prompt-label effects from path-hop effects.

Tab 2: methods and what each tests.

**5. Main Results (1.0 pages)**
Answerability-gated difficulty, solver accuracy, path dependency, shortcut rate. PathQG improves structural dependency and shifts difficulty upward.

Tab 3: main comparison. Tab 4: difficulty confusion matrix.

**6. Causal Evidence: Path-Label Conflict (1.0 pages)**
THE most important section. Show 3x3 or 2x2 conflict experiment. When label and path disagree, judged difficulty follows path hop.

Tab 5: conflict matrix. Tab 6: ordinal regression.

**7. Hard Difficulty Analysis (1.0 pages)**
Frame as error analysis that led to stronger design. "A pilot version produced path-dependent but cognitively shallow Hard questions. This exposed the distinction between path exposure and path reasoning. We redesigned Hard generation to hide intermediate events and select against shortcuts."

Tab 7: before/after Hard strategy comparison.

**8. Human Evaluation and Judge Reliability (0.6 pages)**
Human audit, cross-family LLM agreement, judge-human agreement.

Tab 8: agreement table.

**9. Discussion and Limitations (0.5 pages)**
3-hop paths are sparse, Hard control weaker than Easy/Medium, hop count is imperfect proxy. Event paths are effective but not sufficient alone; robust Hard control requires path-aware generation + shortcut filtering.

**10. Conclusion (0.2 pages)**
> "Event-path hop count provides a useful structural signal for difficulty-controlled QG. It reliably separates easier from more complex questions and, with shortcut-aware generation, enables partial Hard control."

---

### Key Framing for Hard Failure

Do NOT frame as: "Hard failed."
Frame as: "Naive Hard generation failed, revealing that difficulty is not path length alone. Our final method shows that path length becomes effective when the generated question hides rather than reveals the reasoning path."

This turns the failure into a methodological contribution: **difficulty control requires both structural path depth and anti-shortcut realization**.

---

## Reviewer Sharp Recommendations

1. Do NOT submit with the current Hard result (0% judged Hard). Fix it first.
2. The next experiment should be the path-label conflict + matched-hop evaluation; it directly tests whether hop count is doing causal work.
3. The paper can survive weak absolute scores, but it cannot survive a central difficulty-control method whose Hard outputs are never judged Hard.
4. If after improved Hard generation the Hard hit remains 0-10%, make a 2-level Easy vs Non-Easy paper, not a 3-level paper.
5. Human evaluation is not optional. Even 150 items with one annotator is critical.
6. Cross-family judge is not optional. Same-family Qwen-only judging will be flagged by reviewers.
