# Hard Rescue Pilot Report

**Date:** 2026-05-06 02:59
**Paths:** 3 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
**K candidates per path per strategy:** 2
**Total candidates generated:** 30
**API calls:** generation=54, filter=30, judge=60

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 3 |
| Total candidates | 30 |
| Grammar pass | 27 |
| Generation errors | 0 |
| Filter pass | 0 |

## 2. Candidate-Level Difficulty Prediction

### All Candidates (including filter-failed)

| Metric | Value |
|--------|------:|
| Judged candidates | 30 |
| Pred Hard | 0 (0.0%) |
| Pred Medium | 29 (96.7%) |
| Pred Easy | 1 (3.3%) |

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
| Total unique paths | 3 | — |
| Paths with >= 1 Pred Hard | 0 | 0.0% |
| Paths with Pred Hard + answerable | 0 | 0.0% |
| Paths with Pred Hard + ans + fec + pathdep | 0 | 0.0% |

## 4. Per-Strategy Comparison

| Strategy | N judged | Pred Hard | Pred Med | Pred Easy | Ans% | FEC% | PathDep Strong% | Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|
| hidden_endpoint | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 100% | 33% | 0% |
| relation_composition | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 33% | 67% | 0% |
| contrastive | 6 | 0 (0%) | 5 (83%) | 1 (17%) | 100% | 83% | 50% | 0% |
| missing_bridge | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 67% | 50% | 0% |
| implicit_chain | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 33% | 67% | 0% |

## 5. Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no: skipped (early exit) | 30 |
| path_coverage=skipped (early exit) | 30 |
| answer_phrase=skipped (early exit) | 3 |
| grammar=too_long_hard: 47 words | 1 |
| grammar=too_long_hard: 46 words | 1 |
| grammar=base: no question mark | 1 |

## 6. Quality Metrics Summary

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable (yes/partial) | 30/30 (100%) | — |
| Final-Event Consistent | 19/30 (63%) | — |
| PathDep Strong | 16/30 (53%) | — |
| PathDep Strong+Partial | 16/30 (53%) | — |
| Single-Sent Answerable=no | 0/30 (0%) | — |

## 7. Best Samples (Pred Hard + Answerable + PathDep)

*No Pred Hard + answerable + path-dependent candidates found.*

## 8. Worst Samples (Filter Failed or Pred Easy)

### Worst #1 [contrastive]

- **Question:** After the western-most Allied corps surrounded and isolated Bayonne, and the remaining two Allied corps pushed Soult's army back to Orthez, what happened to the French soldiers during their retreat?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact from one sentence in the context, specifically about the withdrawal of the French soldiers.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #2 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman Empire after it stopped the Russian advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The solver needs to connect the stopping of the Russian advance at Silistra with the subsequent signing of an agreement on 30 March, requiring them to understand the implications of the events.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #3 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The solver needs to connect the stopping of the Russian advance at Silistra with the subsequent actions of the allies, specifically the signing of an agreement, which requires understanding the sequence of events.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #4 [relation_composition]

- **Question:** What was the consequence for Russia following its troops stopping the advance at Silistra and the subsequent Ottoman defensive campaign?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** no
- **Filter pass:** False
- **Judge reason:** The question requires the solver to connect the stopping of the Russian advance at Silistra with the subsequent actions of France and Britain, which leads to the signing of an agreement. This involves understanding the implications of the events rather than just finding a single fact.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #5 [relation_composition]

- **Question:** What was the consequence for Russia following its troops stopping in the Balkans in July 1853?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** no
- **Filter pass:** False
- **Judge reason:** The question requires the solver to connect the stopping of Russian troops in the Balkans with the subsequent actions of France and Britain, leading to the signing of an agreement, which involves understanding the sequence of events.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

## 9. Success Criteria Evaluation

- **Pred Hard rate (candidate-level):** 0.0%
- **Path yield (Pred Hard + ans + fec + pathdep):** 0.0%

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**
Switch to Easy vs Non-Easy or path-groundedness paper.