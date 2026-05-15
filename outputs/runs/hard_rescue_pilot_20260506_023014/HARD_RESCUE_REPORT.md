# Hard Rescue Pilot Report

**Date:** 2026-05-06 02:40
**Paths:** 3 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
**K candidates per path per strategy:** 2
**Total candidates generated:** 30
**API calls:** generation=58, filter=28, judge=56

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 3 |
| Total candidates | 30 |
| Grammar pass | 26 |
| Generation errors | 2 |
| Filter pass | 0 |

## 2. Candidate-Level Difficulty Prediction

### All Candidates (including filter-failed)

| Metric | Value |
|--------|------:|
| Judged candidates | 28 |
| Pred Hard | 0 (0.0%) |
| Pred Medium | 26 (92.9%) |
| Pred Easy | 2 (7.1%) |

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
| hidden_endpoint | 6 | 0 (0%) | 5 (83%) | 1 (17%) | 100% | 67% | 50% | 0% |
| relation_composition | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 100% | 100% | 0% |
| contrastive | 4 | 0 (0%) | 4 (100%) | 0 (0%) | 100% | 75% | 50% | 0% |
| missing_bridge | 6 | 0 (0%) | 5 (83%) | 1 (17%) | 100% | 83% | 83% | 0% |
| implicit_chain | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 33% | 83% | 0% |

## 5. Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no: skipped (early exit) | 28 |
| path_coverage=skipped (early exit) | 28 |
| answer_phrase=skipped (early exit) | 2 |
| grammar=too_long_hard: 48 words | 1 |
| grammar=base: no question mark | 1 |

## 6. Quality Metrics Summary

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable (yes/partial) | 28/28 (100%) | — |
| Final-Event Consistent | 20/28 (71%) | — |
| PathDep Strong | 21/28 (75%) | — |
| PathDep Strong+Partial | 21/28 (75%) | — |
| Single-Sent Answerable=no | 0/28 (0%) | — |

## 7. Best Samples (Pred Hard + Answerable + PathDep)

*No Pred Hard + answerable + path-dependent candidates found.*

## 8. Worst Samples (Filter Failed or Pred Easy)

### Worst #1 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the French army following their defeat at Orthez, where they were forced to retreat after their center and left flank were overcome by the Allied forces?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact from one sentence in the context, specifically about the withdrawal of the French army.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #2 [missing_bridge]

- **Question:** Between the western-most Allied corps surrounding and isolating Bayonne and the withdrawal eventually ending in a scramble for safety, what was the significant event that connected these two points?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question asks for a significant event that connects two points, and the answer can be found in a single sentence that directly relates to the withdrawal being conducted.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #3 [hidden_endpoint]

- **Question:** What were the key military and political events that led to the Russian fleet's destruction at Sinop and the subsequent arrival of French and British forces in Varna, and how did these events contribu
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to connect the destruction of the Turkish reinforcement attempt by the Russian fleet at Sinop and the arrival of French and British forces in Varna, which involves understanding the sequence of events and their implications, thus needing two distinct pieces of information.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #4 [hidden_endpoint]

- **Question:** What were the key military and diplomatic actions that led to the Russians abandoning Silistra and the subsequent easing of tensions between France, Britain, and Russia?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** no
- **Filter pass:** False
- **Judge reason:** The question requires the solver to connect the military actions leading to the Russians abandoning Silistra and the diplomatic context, which involves understanding multiple events and their implications.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #5 [relation_composition]

- **Question:** What outcome resulted from the chain of events that began with Russian troops stopping their advance at Silistra and involved a Russian fleet destroying a Turkish reinforcement attempt?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The solver needs to connect the stopping of the Russian advance at Silistra and the destruction of the Turkish reinforcement attempt to understand the outcome, which requires linking two distinct pieces of information.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

## 9. Success Criteria Evaluation

- **Pred Hard rate (candidate-level):** 0.0%
- **Path yield (Pred Hard + ans + fec + pathdep):** 0.0%

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**
Switch to Easy vs Non-Easy or path-groundedness paper.