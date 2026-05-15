# Hard Rescue Pilot Report

**Date:** 2026-05-06 07:30
**Paths:** 22 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition, contrastive, missing_bridge, implicit_chain
**K candidates per path per strategy:** 5
**Total candidates generated:** 550
**API calls:** generation=928, filter=529, judge=1058

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 22 |
| Total candidates | 550 |
| Grammar pass | 505 |
| Generation errors | 21 |
| Filter pass | 65 |

## 2. Candidate-Level Difficulty Prediction

### All Candidates (including filter-failed)

| Metric | Value |
|--------|------:|
| Judged candidates | 529 |
| Pred Hard | 124 (23.4%) |
| Pred Medium | 346 (65.4%) |
| Pred Easy | 59 (11.2%) |

### Filter-Passing Candidates Only

| Metric | Value |
|--------|------:|
| Judged candidates | 65 |
| Pred Hard | 5 (7.7%) |
| Pred Medium | 47 (72.3%) |
| Pred Easy | 13 (20.0%) |

## 3. Path-Level Pred Hard Yield

| Metric | Count | Rate |
|--------|------:|-----:|
| Total unique paths | 22 | — |
| Paths with >= 1 Pred Hard | 17 | 77.3% |
| Paths with Pred Hard + answerable | 17 | 77.3% |
| Paths with Pred Hard + ans + fec + pathdep | 13 | 59.1% |

## 4. Per-Strategy Comparison

| Strategy | N judged | Pred Hard | Pred Med | Pred Easy | Ans% | FEC% | PathDep Strong% | Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|
| hidden_endpoint | 105 | 46 (44%) | 53 (50%) | 6 (6%) | 100% | 81% | 33% | 10% |
| relation_composition | 108 | 11 (10%) | 86 (80%) | 11 (10%) | 100% | 66% | 42% | 9% |
| contrastive | 104 | 4 (4%) | 77 (74%) | 23 (22%) | 100% | 70% | 36% | 18% |
| missing_bridge | 108 | 54 (50%) | 46 (43%) | 8 (7%) | 100% | 88% | 42% | 4% |
| implicit_chain | 104 | 9 (9%) | 84 (81%) | 11 (11%) | 99% | 61% | 39% | 18% |

## 5. Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| path_coverage=covers 1 prior events, need >= 2 [FAIL] | 312 |
| answer_consistency=no: extracted from text: asks=no ans=no cons=no | 104 |
| path_coverage=covers 0 prior events, need >= 2 [FAIL] | 60 |
| answer_consistency=no: extracted from text: asks=no ans=yes cons=no | 36 |
| answer_phrase=skipped (early exit) | 24 |
| answer_consistency=no: skipped (early exit) | 24 |
| path_coverage=skipped (early exit) | 24 |
| hard_implicit=3 prior triggers explicitly in question (max 2 allowed) | 24 |
| answer_consistency=no: brief | 18 |
| answer_consistency=no: extracted from text: asks=yes ans=no cons=no | 14 |
| grammar=base: no question mark | 9 |
| grammar=too_long_hard: 46 words | 3 |
| answer_consistency=no: extracted from text: asks=no ans=yes cons=yes | 3 |
| answer_consistency=no: The question asks about agricultural impacts in Bangladesh, but the target event and context provided are about the impact on St. Martin's Island. There is no information in the context about agricultural losses in Bangladesh. | 3 |
| grammar=too_long_hard: 62 words | 2 |
| answer_consistency=no: The question does not match the target event and the context does not provide information about the immediate consequences for The Myyrmanni after it became crowded. | 2 |
| answer_consistency=no: The question does not match the target event trigger and the provided context does not contain information about the ultimate consequence for the King David Hotel after warnings were sent. | 2 |
| grammar=base: word repetition: defense | 1 |
| answer_consistency=no: extracted from text: asks=yes ans=yes cons=no | 1 |
| answer_consistency=no: The question asks about the ultimate consequence of the French army's withdrawal after being surrounded and isolated from Bayonne, which is not directly related to the target final event described in the target answer. | 1 |
| answer_consistency=no: The question does not match the target event trigger and the context does not provide information about public outcry or an inquiry leading to the Dutch campaign. | 1 |
| answer_consistency=no: The question asks about the ultimate consequence for Malacca after the Dutch began their campaign, which is not the same as the target event about agreeing not to seek territories or wage war with the Malay kingdoms. | 1 |
| answer_consistency=no: The question does not specifically ask about the target final event (the closure of the investigation), and the context does not provide information about the public reaction to the incident or their awareness of the presence of children. | 1 |
| answer_consistency=no: The question asks about the ultimate outcome for Aleksander Rogaliński and his unit, which is addressed in the context. However, it does not specifically ask about the target final event 'killed'. | 1 |
| grammar=too_long_hard: 54 words | 1 |
| answer_consistency=no: The question asks about key developments leading to the successful conclusion of the operation, which is not the same as the target event of NATO adapting to the post-Cold War era. | 1 |
| grammar=base: word repetition: day | 1 |
| grammar=base: word repetition: what | 1 |
| grammar=base: word repetition: consequence | 1 |
| answer_consistency=no: The question diverges from the provided context and introduces elements not mentioned, such as public outcry and an official inquiry, which are not part of the given information. | 1 |
| answer_consistency=no: The question does not align with the target final event. The target event is about the number of people killed, while the question asks about the consequences of warnings sent after the event. | 1 |
| answer_consistency=no: The question does not specifically ask about the target final event (91 people killed) but rather about the consequences for the King David Hotel after warnings were sent, which is not addressed in the context. | 1 |
| answer_consistency=no: The question does not match the target event trigger and the provided context does not contain information about the ultimate outcome for The King David Hotel after warnings were sent. | 1 |
| answer_consistency=no: The question is about the impact of Irgun's warnings on the hotel's preparedness, which is not related to the number of people killed or the target event. | 1 |
| answer_consistency=no: The question does not match the target event and the context does not provide information about Christians in Ottoman vassal states after the war in the Balkans. | 1 |
| grammar=base: word repetition: public | 1 |
| answer_consistency=no: The question does not ask about the target final event, which is related to granting Christians official equality. The context and question are about military actions in the Crimean War, not about religious rights. | 1 |
| answer_consistency=no: The question does not ask about the target final event, which is related to granting Christians official equality. The context and question focus on military events and do not mention any granting of rights to Christians. | 1 |
| answer_consistency=no: The question does not match the target event and the context provided does not contain information about modern military technology used in the Crimean War for better communication and coordination. | 1 |
| answer_consistency=no: The question does not directly ask about the target final event (J. Randy Taraborrelli's comment) and the provided context does not contain enough information to answer the full question accurately. | 1 |
| grammar=base: bad start: who, | 1 |
| answer_consistency=no: The question is about the inspiration for the tour's title, not about the incorporation of multimedia components. | 1 |
| answer_consistency=no: The question is about the inspiration for the tour's title and does not directly ask about the incorporation of multimedia components, which is the target final event. | 1 |
| answer_consistency=no: The question focuses on the short-term military consequences of the Battle of Balaclava and Inkerman, while the target event is about the long-term social and institutional changes in Russia following the war. | 1 |
| answer_consistency=no: The question does not specifically ask about the target final event (injuries) and introduces elements (warnings and their consequences) not directly supported by the provided context. | 1 |
| grammar=base: word repetition: population | 1 |
| grammar=too_long_hard: 53 words | 1 |
| answer_consistency=no: The question asks about the ultimate consequence for Silkwood, Queensland after the cyclone, which can be answered from the context, and a correct answer would identify the same final event as the target answer. | 1 |
| answer_consistency=no: The question asks about the ultimate consequence for Silkwood, Queensland after the cyclone, which can be answered from the context, and a correct answer would identify the same final event as the target answer meaning. | 1 |
| answer_consistency=no: The question focuses on the public outcry and the evacuation, not the damage to St. Martin's Island. | 1 |
| grammar=repeat_question_mark | 1 |

## 6. Quality Metrics Summary

| Metric | All Judged | Filter-Passing |
|--------|----------:|---------------:|
| Answerable (yes/partial) | 528/529 (100%) | 65/65 (100%) |
| Final-Event Consistent | 387/529 (73%) | 56/65 (86%) |
| PathDep Strong | 203/529 (38%) | 26/65 (40%) |
| PathDep Strong+Partial | 203/529 (38%) | 26/65 (40%) |
| Single-Sent Answerable=no | 126/529 (24%) | 5/65 (8%) |

## 7. Best Samples (Pred Hard + Answerable + PathDep)

### Best #1 [hidden_endpoint]

- **Question:** What were the consequences of the Russian occupation of the Danubian Principalities and the subsequent Ottoman defensive campaign that stopped the Russian advance at Silistra, leading to the Ottoman r
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** True
- **Judge reason:** The question requires the solver to trace a chain of events: the Russian occupation leads to the Ottoman defensive campaign, which then connects to the destruction of the reinforcement attempt, ultimately leading to the consequence of forbidding Russia from basing warships in the Black Sea. This involves multiple steps of reasoning across different sentences.
- **Filter reason:** all checks passed

### Best #2 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Russian Murom Regiment after the uprising began and they approached a local manor where the Poles had their quarters?
- **Answer:** ordered a charge of the Russians
- **Event path:** uprising -> approached -> refused -> ordered
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** True
- **Judge reason:** The question requires the solver to trace the sequence of events from the uprising to the Russian approach, the refusal to negotiate, and finally to the order to charge, which involves multiple reasoning steps.
- **Filter reason:** all checks passed

### Best #3 [missing_bridge]

- **Question:** What was the ultimate consequence for the defenders of Sihang Warehouse after they held out against the Japanese forces during the Battle of Shanghai in October and November 1937?
- **Answer:** provided an exaggerated number to girl guide Yang Huimin to announce to the public
- **Event path:** held -> War -> opening -> announce
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** True
- **Judge reason:** The question requires the solver to trace the chain of events from the defense of the warehouse to the morale boost for the Chinese army, then to the exaggerated number provided by the commander, which involves multiple steps of reasoning.
- **Filter reason:** all checks passed

### Best #4 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace a chain of events from the Ottoman stopping the Russian advance at Silistra to the signing of an agreement, which involves multiple steps and connections between different pieces of information.
- **Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 1 prior events, need >= 2 [FAIL]

### Best #5 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Pred difficulty:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question requires the solver to trace a chain of events from the Ottoman forces stopping the Russian advance at Silistra to the signing of an agreement, which involves multiple steps and connections between different pieces of information in the context.
- **Filter reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]

## 8. Worst Samples (Filter Failed or Pred Easy)

### Worst #1 [contrastive]

- **Question:** How did the initial state of the French army's retreat compare to the later chaos, given that they were initially isolated and then overpowered before their departure became a disordered rush?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that describes the initial state of the French army's retreat, which states it was conducted in good order.
- **Filter reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]

### Worst #2 [missing_bridge]

- **Question:** What was the outcome of the Allied corps' eastward drive after they had pushed the French back from several river lines and isolated Bayonne?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Pred difficulty:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that describes the outcome of the French withdrawal, which is clearly stated in one sentence.
- **Filter reason:** answer_consistency=no: extracted from text: asks=no ans=no cons=no; path_coverage=covers 1 prior events, need >= 2 [FAIL]

### Worst #3 [relation_composition]

- **Question:** What was the ultimate outcome for the Russian commander and his unit following the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that states the outcome for the Russian commander, requiring only one step to extract the information.
- **Filter reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]

### Worst #4 [relation_composition]

- **Question:** What was the fate of the Russian commander and the outcome for the Russian Murom Regiment following the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that states the fate of the Russian commander, which is clearly presented in the context.
- **Filter reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]

### Worst #5 [relation_composition]

- **Question:** What was the fate of the Russian commander and the outcome for the Russian Murom Regiment following the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Pred difficulty:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes
- **Filter pass:** False
- **Judge reason:** The question can be answered by directly referencing the sentence that states the fate of the Russian commander, which is clearly presented in the context.
- **Filter reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]

## 9. Success Criteria Evaluation

- **Pred Hard rate (candidate-level):** 23.4%
- **Path yield (Pred Hard + ans + fec + pathdep):** 59.1%

**RESULT: SUCCESS. >= 20% paths produce Pred Hard + quality candidates.**
3-level difficulty-control claim is viable with this evidence.