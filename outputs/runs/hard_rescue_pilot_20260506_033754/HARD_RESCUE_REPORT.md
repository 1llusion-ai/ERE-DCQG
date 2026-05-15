# Hard Rescue Pilot Report

**Date:** 2026-05-06 04:05
**Paths:** 10 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
**K candidates per path per strategy:** 2
**Total candidates generated:** 100
**API calls:** generation=169, filter=98, judge=196

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 10 |
| Total candidates | 100 |
| Grammar pass | 93 |
| Generation errors | 2 |
| Filter pass | 0 |

## 2. Candidate-Level Difficulty Prediction

### All Candidates (including filter-failed)

| Metric | Value |
|--------|------:|
| Judged candidates | 98 |
| Pred Hard | 24 (24.5%) |
| Pred Medium | 63 (64.3%) |
| Pred Easy | 11 (11.2%) |

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
| Paths with >= 1 Pred Hard | 8 | 80.0% |
| Paths with Pred Hard + answerable | 8 | 80.0% |
| Paths with Pred Hard + ans + fec + pathdep | 6 | 60.0% |

## 4. Per-Strategy Comparison

| Strategy | N judged | Pred Hard | Pred Med | Pred Easy | Ans% | FEC% | PathDep Strong% | Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|
| hidden_endpoint | 20 | 9 (45%) | 11 (55%) | 0 (0%) | 100% | 85% | 55% | 0% |
| relation_composition | 19 | 5 (26%) | 12 (63%) | 2 (11%) | 100% | 79% | 42% | 0% |
| contrastive | 20 | 0 (0%) | 15 (75%) | 5 (25%) | 100% | 70% | 40% | 0% |
| missing_bridge | 20 | 9 (45%) | 9 (45%) | 2 (10%) | 100% | 90% | 30% | 0% |
| implicit_chain | 19 | 1 (5%) | 16 (84%) | 2 (11%) | 100% | 74% | 42% | 0% |

## 5. Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no: skipped (early exit) | 98 |
| path_coverage=skipped (early exit) | 98 |
| answer_phrase=skipped (early exit) | 5 |
| grammar=base: no question mark | 3 |
| grammar=too_long_hard: 60 words | 1 |
| grammar=base: word repetition: consequence | 1 |

## 6. Quality Metrics Summary

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable (yes/partial) | 98/98 (100%) | — |
| Final-Event Consistent | 78/98 (80%) | — |
| PathDep Strong | 41/98 (42%) | — |
| PathDep Strong+Partial | 41/98 (42%) | — |
| Single-Sent Answerable=no | 24/98 (24%) | — |

## 7. Best Samples (Pred Hard + Answerable + PathDep)

### Best #1 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman forces after the Russian advance was stopped at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace the sequence of events from the stopping of the Russian advance at Silistra to the signing of the agreement, which involves understanding multiple interconnected events and their consequences.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Best #2 [relation_composition]

- **Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Turkish reinforcement being destroyed at Sinop, leading to France and Britain's decision to move their forces to Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace multiple events: the stopping of the Ottoman forces, the destruction of the Turkish reinforcement, and the subsequent decision by France and Britain, which leads to the final event of signing on. This involves connecting several pieces of information across different sentences.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Best #3 [relation_composition]

- **Question:** What was the consequence for Russia following its troops stopping in the Balkans in July 1853?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace a chain of events from the initial stopping of Russian troops to the eventual signing of an agreement, which involves multiple steps and connections between different pieces of information.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Best #4 [implicit_chain]

- **Question:** What was the consequence of the Russian fleet's destruction of the Turkish reinforcement attempt at Sinop on the Ottoman forces' ability to defend Kars?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace the impact of the Russian fleet's action on the Ottoman forces, which involves understanding the sequence of events leading to the signing of an agreement, thus necessitating multiple reasoning steps.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Best #5 [relation_composition]

- **Question:** How did the Russian occupation of the Danubian Principalities and subsequent Ottoman defensive actions lead to the Russian fleet's destruction at Sinop, and what were the consequences for Russia's nav
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace multiple events: the Russian occupation leading to Ottoman actions, the destruction at Sinop, and the resulting treaty implications for Russia's naval presence, which involves connecting several pieces of information across different sentences.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

## 8. Worst Samples (Filter Failed or Pred Easy)

### Worst #1 [contrastive]

- **Question:** What was the outcome of the incident at the Myyrmanni shopping mall on October 11, 2002, in Vantaa, Finland, where the mall was crowded with 1,000–2,000 people, including many children, after a bomb e
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The answer can be directly found in the sentence that states the incident was closed in January 2003 without any indictments, requiring no additional reasoning or connections.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #2 [contrastive]

- **Question:** What was the outcome of the incident at the Myyrmanni shopping mall on October 11, 2002, in Vantaa, Finland, where the area was exceptionally crowded with 1,000–2,000 people, including many children, 
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that states the incident was closed in January 2003 without any indictments, requiring only one step to extract the information.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #3 [relation_composition]

- **Question:** What was the ultimate outcome for the Russian commander and his unit following the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that states the outcome for the Russian commander, requiring no additional reasoning or connections.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #4 [contrastive]

- **Question:** What was the consequence for the Russian commander when the Polish unit under Aleksander Rogaliński charged during the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question directly asks for the consequence of the Polish charge, which is explicitly stated in one sentence about the Russian commander's fate.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

### Worst #5 [contrastive]

- **Question:** What was the outcome for the Russian commander when his regiment faced the Polish uprising and refused to negotiate?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question directly asks about the outcome for the Russian commander, which can be found in a single sentence that states he was killed after the fight.
- **Filter reason:** answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)

## 9. Success Criteria Evaluation

- **Pred Hard rate (candidate-level):** 24.5%
- **Path yield (Pred Hard + ans + fec + pathdep):** 60.0%

**RESULT: SUCCESS. >= 20% paths produce Pred Hard + quality candidates.**
3-level difficulty-control claim is viable with this evidence.