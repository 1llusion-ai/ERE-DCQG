# Hard Rescue Pilot Report

**Date:** 2026-05-06 02:20
**Paths:** 3 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
**K candidates per path per strategy:** 2
**Total candidates generated:** 30
**API calls:** generation=58, filter=29, judge=58

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 3 |
| Total candidates | 30 |
| Grammar pass | 19 |
| Generation errors | 1 |
| Filter pass | 0 |

## 2. Candidate-Level Difficulty Prediction

### All Candidates (including filter-failed)

| Metric | Value |
|--------|------:|
| Judged candidates | 29 |
| Pred Hard | 0 (0.0%) |
| Pred Medium | 27 (93.1%) |
| Pred Easy | 2 (6.9%) |

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
| hidden_endpoint | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 50% | 83% | 0% |
| relation_composition | 6 | 0 (0%) | 5 (83%) | 1 (17%) | 100% | 50% | 83% | 0% |
| contrastive | 5 | 0 (0%) | 5 (100%) | 0 (0%) | 100% | 100% | 20% | 0% |
| missing_bridge | 6 | 0 (0%) | 5 (83%) | 1 (17%) | 100% | 83% | 50% | 0% |
| implicit_chain | 6 | 0 (0%) | 6 (100%) | 0 (0%) | 100% | 33% | 100% | 0% |

## 5. Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no: skipped (early exit) | 29 |
| path_coverage=skipped (early exit) | 29 |
| answer_phrase=skipped (early exit) | 10 |
| grammar=base: bad start: given | 5 |
| grammar=base: bad start: between | 5 |

## 6. Quality Metrics Summary

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable (yes/partial) | 29/29 (100%) | — |
| Final-Event Consistent | 18/29 (62%) | — |
| PathDep Strong | 20/29 (69%) | — |
| PathDep Strong+Partial | 20/29 (69%) | — |
| Single-Sent Answerable=no | 0/29 (0%) | — |

## 7. Best Samples (Pred Hard + Answerable + PathDep)

*No Pred Hard + answerable + path-dependent candidates found.*

## 8. Worst Samples (Filter Failed or Pred Easy)

### Worst #1 [missing_bridge]

- **Question:** Between Russian troops occupying the Danubian Principalities in the Balkans in July 1853 and the Treaty of Paris forbidding Russia from basing warships in the Black Sea, what significant event occurre
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question directly asks for the significant event that occurred after the Russian occupation and before the Treaty of Paris, which can be answered by identifying the final event in the context.
- **Filter reason:** grammar=base: bad start: between; answer_phrase=skipped (early exit); answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #2 [relation_composition]

- **Question:** After the western-most Allied corps surrounded and isolated Bayonne, and the remaining two Allied corps pushed the French back to Orthez, what happened to the French soldiers during their withdrawal?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by identifying a single fact from one sentence in the context, specifically about the withdrawal of the French soldiers.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #3 [hidden_endpoint]

- **Question:** What led to the public outcry in Britain and France, and how did the Treaty of Paris conclude the conflict?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** no
- **Filter pass:** False
- **Judge reason:** The question requires the solver to connect the public outcry in Britain and France with the conclusion of the conflict through the Treaty of Paris, which involves understanding the context of the war and the signing of the treaty, thus needing to connect two pieces of information.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #4 [hidden_endpoint]

- **Question:** What significant event occurred after the Ottoman forces were stopped at Silistra and the Russian fleet destroyed the Turkish attempt to reinforce Kars, leading to the arrival of French and British fo
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The solver needs to connect the stopping of the Ottoman forces at Silistra and the destruction of the Turkish reinforcement attempt to understand that these events led to the arrival of French and British forces, which is a two-step connection.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #5 [relation_composition]

- **Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Russian fleet destroying the Turkish attempt to reinforce Kars, leading to the arrival of French and British forces in 
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The solver needs to connect the stopping of the Ottoman forces at Silistra and the destruction of the Turkish reinforcement attempt to understand the arrival of French and British forces, which requires linking two distinct pieces of information.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

## 9. Success Criteria Evaluation

- **Pred Hard rate (candidate-level):** 0.0%
- **Path yield (Pred Hard + ans + fec + pathdep):** 0.0%

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**
Switch to Easy vs Non-Easy or path-groundedness paper.