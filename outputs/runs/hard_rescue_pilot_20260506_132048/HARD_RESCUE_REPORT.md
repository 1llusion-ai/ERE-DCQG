# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 13:36
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, missing_bridge, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 30
**API calls:** generation=54, filter=28, judge=56

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 30 |
| Grammar pass | 28 |
| Generation errors | 2 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 28 |
| Blind Pred Hard | 3 (10.7%) |
| Blind Pred Medium | 15 (53.6%) |
| Blind Pred Easy | 10 (35.7%) |

## 3. Blind Difficulty Judge — New Filter-Passing Candidates

| Metric | Value |
|--------|------:|
| Filter-passing candidates | 0 |
| Blind Pred Hard | 0 |
| Blind Pred Medium | 0 |
| Blind Pred Easy | 0 |

## 4. Path-Level Blind Pred Hard Yield

| Metric | Count | Rate |
|--------|------:|-----:|
| Total unique paths | 5 | — |
| Paths with >= 1 Blind Pred Hard | 3 | 60.0% |
| Paths with Blind Hard + answerable | 2 | 40.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 1 | 20.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 9 | 1 (11%) | 6 (67%) | 2 (22%) | 100% | 89% | 44% | 0% |
| missing_bridge | 10 | 0 (0%) | 6 (60%) | 4 (40%) | 100% | 100% | 30% | 0% |
| relation_composition | 9 | 2 (22%) | 3 (33%) | 4 (44%) | 89% | 89% | 44% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| path_dependency=none | 17 |
| blind_pred=Medium | 15 |
| single_sentence_answerable=partial | 13 |
| answer_consistency=no | 12 |
| blind_pred=Easy | 10 |
| single_sentence_answerable=yes | 10 |
| answer_consistency=judge_error | 4 |
| blind_fec=no | 2 |
| hard_implicit=3 prior triggers explicitly in question (max 2 allowed) | 1 |
| blind_answerable=no | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 27/28 (96%) | — |
| Blind Final-Event Consistent | 26/28 (93%) | — |
| PathDep Strong | 11/28 (39%) | — |
| Single-Sent Answerable=no | 5/28 (18%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 3 | 60.0% |
| Top-1 Answerable + FEC | 3 | 60.0% |
| Top-1 PathDep Strong | 4 | 80.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | hidden_endpoint | Hard | yes | no | strong | no | N | What were the key military actions that led to the Russian f |
| 2 | relation_composition | Hard | yes | yes | strong | no | N | Why did the Russian occupation of the Danubian Principalitie |
| 3 | missing_bridge | Medium | yes | yes | strong | partial | N | What was the ultimate consequence for the Anglo-Portuguese A |
| 4 | relation_composition | Hard | no | no | none | no | N | What public outcry and inquiry led to the Dutch starting the |
| 5 | missing_bridge | Medium | yes | yes | strong | no | N | What was the ultimate consequence for The Myyrmanni after it |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 28/28
- **blind_context_contains_answer_sentence=false:** 0/28

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What were the key military actions that led to the Russian fleet's destruction at Sinop, and how did this event influence the arrival of French and Br
**Blind Pred:** Hard

```
You are an expert difficulty evaluator for reading comprehension questions.

## Context
[S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman suzerainty, then began to cross the Danube.
[S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
[S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russian fleet at Sinop.
[S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
[S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
[S15] Aside from a minor skirmish at Köstence (today Constanța), there was little for the allies to do.
[S25] France and Britain welcomed this development, as the conflict was growing unpopular at home.
[S26] The Treaty of Paris, signed on 30 March 1856, ended the war.
[S27] It forbade Russia from basing warships in the Black Sea.

## Question
"What were the key military actions that led to the Russian fleet's destruction at Sinop, and how did this event influence the arrival of French and British forces in Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question
```

### Sample 2

**Question:** What significant event occurred after the Ottoman forces stopped the Russian advance at Silistra and the Turkish reinforcement attempt was destroyed b
**Blind Pred:** Medium

```
You are an expert difficulty evaluator for reading comprehension questions.

## Context
[S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman suzerainty, then began to cross the Danube.
[S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
[S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russian fleet at Sinop.
[S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
[S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
[S15] Aside from a minor skirmish at Köstence (today Constanța), there was little for the allies to do.
[S25] France and Britain welcomed this development, as the conflict was growing unpopular at home.
[S26] The Treaty of Paris, signed on 30 March 1856, ended the war.
[S27] It forbade Russia from basing warships in the Black Sea.

## Question
"What significant event occurred after the Ottoman forces stopped the Russian advance at Silistra and the Turkish reinforcement attempt was destroyed by a Russian fleet at Sinop, leading to the arrival of French and British forces in Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver d
```

### Sample 3

**Question:** What was the ultimate consequence for the Russian troops after they stopped their advance at Silistra?
**Blind Pred:** Medium

```
You are an expert difficulty evaluator for reading comprehension questions.

## Context
[S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman suzerainty, then began to cross the Danube.
[S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
[S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russian fleet at Sinop.
[S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
[S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
[S15] Aside from a minor skirmish at Köstence (today Constanța), there was little for the allies to do.
[S25] France and Britain welcomed this development, as the conflict was growing unpopular at home.
[S26] The Treaty of Paris, signed on 30 March 1856, ended the war.
[S27] It forbade Russia from basing warships in the Black Sea.

## Question
"What was the ultimate consequence for the Russian troops after they stopped their advance at Silistra?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [hidden_endpoint]

- **Question:** What were the key military actions that led to the Russian fleet's destruction at Sinop, and how did this event influence the arrival of French and British forces in Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to trace multiple events: the siege at Kars, the destruction of the Turkish reinforcement attempt at Sinop, and the subsequent arrival of French and British forces in Varna, which involves connecting several pieces of information across different sentences.
- **Filter Reason:** blind_fec=no

### Best #2 [relation_composition]

- **Question:** Why did the Russian occupation of the Danubian Principalities and the destruction of the Ottoman fleet at Sinop lead to the reinforcement of Ottoman forces by France and Britain, ultimately resulting 
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect multiple events: the Russian occupation leading to Ottoman reinforcement by France and Britain, and the subsequent abandonment of Silistra by the Russians, which involves understanding the sequence and implications of these events.
- **Filter Reason:** answer_consistency=no

## 12. Worst Samples

### Worst #1 [missing_bridge]

- **Question:** What was the ultimate consequence for Russia after it stopped advancing in the Balkans in July 1853?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence [S27], which states the consequence for Russia after the war.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [relation_composition]

- **Question:** After the Russian troops stopped advancing at Silistra and the Ottoman attempt to reinforce Kars was destroyed by the Russian fleet at Sinop, what was the long-term impact on Russia's naval presence i
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence about the Treaty of Paris, which states the long-term impact on Russia's naval presence in the Black Sea.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the French army's withdrawal after they were surrounded and isolated from Bayonne and forced to retreat from Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the withdrawal of the French army, which states that it was initially conducted in good order.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #4 [relation_composition]

- **Question:** What was the consequence of the French army's initial good order during their withdrawal after being overcome at Orthez, and how did this affect the safety of the French soldiers?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial good order of the withdrawal and its consequences.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [hidden_endpoint]

- **Question:** What was the ultimate resolution for the Dutch after their initial campaign began in the early 17th century to destroy Portuguese power in the East, particularly concerning their relationship with the
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch regarding their relationship with the Malay kingdoms.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 60.0%
- **Top-1 per path: Answerable + FEC rate:** 60.0%
- **Top-1 per path: PathDep Strong rate:** 80.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 10.7%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 60.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/5**

**RESULT: Hard potential exists, but no publishable Hard candidates passed the new filter.**
Hard rescue candidate generation shows potential; final Hard quality remains unvalidated.