# Hard Rescue Pilot Report

**Date:** 2026-05-06 03:37
**Paths:** 10 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
**K candidates per path per strategy:** 2
**Total candidates generated:** 100
**API calls:** generation=185, filter=95, judge=190

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 10 |
| Total candidates | 100 |
| Grammar pass | 90 |
| Generation errors | 5 |
| Filter pass | 0 |

## 2. Candidate-Level Difficulty Prediction

### All Candidates (including filter-failed)

| Metric | Value |
|--------|------:|
| Judged candidates | 95 |
| Pred Hard | 0 (0.0%) |
| Pred Medium | 76 (80.0%) |
| Pred Easy | 19 (20.0%) |

### Filter-Passing Candidates Only

| Metric | Value |
|--------|------:|
| Judged candidates | 0 |
| Pred Hard | 0 |
| Pred Medium | 0 |
| Pred Easy | 0 |

## 3. Path-Level Pred Hard Yield

| Metric | Count | Rate |
|--------|------:|-----:|
| Total unique paths | 10 | — |
| Paths with >= 1 Pred Hard | 0 | 0.0% |
| Paths with Pred Hard + answerable | 0 | 0.0% |
| Paths with Pred Hard + ans + fec + pathdep | 0 | 0.0% |

## 4. Per-Strategy Comparison

| Strategy | N judged | Pred Hard | Pred Med | Pred Easy | Ans% | FEC% | PathDep Strong% | Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|
| hidden_endpoint | 20 | 0 (0%) | 18 (90%) | 2 (10%) | 100% | 60% | 55% | 0% |
| relation_composition | 18 | 0 (0%) | 14 (78%) | 4 (22%) | 100% | 78% | 50% | 0% |
| contrastive | 19 | 0 (0%) | 15 (79%) | 4 (21%) | 100% | 89% | 42% | 0% |
| missing_bridge | 18 | 0 (0%) | 14 (78%) | 4 (22%) | 100% | 89% | 44% | 0% |
| implicit_chain | 20 | 0 (0%) | 15 (75%) | 5 (25%) | 100% | 75% | 30% | 0% |

## 5. Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no: skipped (early exit) | 95 |
| path_coverage=skipped (early exit) | 95 |
| answer_phrase=skipped (early exit) | 5 |
| grammar=base: no question mark | 2 |
| grammar=broken_grammar: What the destruction | 1 |
| grammar=base: word repetition: what | 1 |
| grammar=base: bad start: by | 1 |

## 6. Quality Metrics Summary

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable (yes/partial) | 95/95 (100%) | — |
| Final-Event Consistent | 74/95 (78%) | — |
| PathDep Strong | 42/95 (44%) | — |
| PathDep Strong+Partial | 42/95 (44%) | — |
| Single-Sent Answerable=no | 0/95 (0%) | — |

## 7. Best Samples (Pred Hard + Answerable + PathDep)

*No Pred Hard + answerable + path-dependent candidates found.*

## 8. Worst Samples (Filter Failed or Pred Easy)

### Worst #1 [implicit_chain]

- **Question:** After the western-most Allied corps surrounded and isolated Bayonne, and the remaining two Allied corps pushed Soult's army back to Orthez, what happened to the French soldiers during their retreat?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact from one sentence in the context, specifically about the French soldiers' retreat.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #2 [hidden_endpoint]

- **Question:** What was the ultimate consequence for Aleksander Rogaliński after the uprising began?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact about the outcome of the engagement, which is clearly stated in one sentence.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #3 [relation_composition]

- **Question:** What was the consequence of the Russian commander's attempt to negotiate and take the Polish forces into captivity, and how did this lead to his death?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying the consequence of the Russian commander's attempt to negotiate, which is directly stated in one sentence of the context.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #4 [relation_composition]

- **Question:** What was the fate of the Russian commander and the outcome for the Russian Murom Regiment following the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact about the fate of the Russian commander, which is explicitly stated in one sentence of the context.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #5 [contrastive]

- **Question:** What was the outcome for the Russian commander after the Poles refused to negotiate during the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact in one sentence that directly states the outcome for the Russian commander after the Poles refused to negotiate.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

## 9. Success Criteria Evaluation

- **Pred Hard rate (candidate-level):** 0.0%
- **Path yield (Pred Hard + ans + fec + pathdep):** 0.0%

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**
Switch to Easy vs Non-Easy or path-groundedness paper.