# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 15:43
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=41, filter=19, judge=38

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 18 |
| Generation errors | 1 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 19 |
| Blind Pred Hard | 3 (15.8%) |
| Blind Pred Medium | 8 (42.1%) |
| Blind Pred Easy | 8 (42.1%) |

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
| Paths with >= 1 Blind Pred Hard | 2 | 40.0% |
| Paths with Blind Hard + answerable | 1 | 20.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 9 | 1 (11%) | 5 (56%) | 3 (33%) | 89% | 67% | 89% | 0% |
| relation_composition | 10 | 2 (20%) | 3 (30%) | 5 (50%) | 90% | 70% | 90% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no | 8 |
| blind_pred=Easy | 8 |
| single_sentence_answerable=yes | 8 |
| blind_pred=Medium | 8 |
| blind_fec=no | 6 |
| answer_consistency=judge_error | 5 |
| path_dependency=none | 2 |
| blind_answerable=no | 2 |
| grammar=base: no question mark | 1 |
| answer_phrase=skipped (early exit) | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 17/19 (89%) | — |
| Blind Final-Event Consistent | 13/19 (68%) | — |
| PathDep Strong | 17/19 (89%) | — |
| Single-Sent Answerable=no | 4/19 (21%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 2 | 40.0% |
| Top-1 Answerable + FEC | 3 | 60.0% |
| Top-1 PathDep Strong | 5 | 100.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Easy | yes | yes | strong | yes | N | What formal resolution resulted from Russian troops stopping |
| 2 | relation_composition | Hard | no | no | strong | no | N | What July forces rushed to reinforce Silistra despite July 1 |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | What was the consequence of the French center and left flank |
| 4 | hidden_endpoint | Hard | no | no | strong | no | N | What public outcry and inquiry led to the Dutch launching sm |
| 5 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What was the outcome of the especially crowded Myyrmanni sho |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 19/19
- **blind_context_contains_answer_sentence=false:** 0/19

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What formal resolution ended the military engagement after the Russian forces stopped advancing at Silistra and the Ottoman forces successfully defend
**Blind Pred:** Easy

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
"What formal resolution ended the military engagement after the Russian forces stopped advancing at Silistra and the Ottoman forces successfully defended the town?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. F
```

### Sample 2

**Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived at Varna?
**Blind Pred:** Easy

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
"What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived at Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across
```

### Sample 3

**Question:** What formal resolution resulted from Russian troops stopping their advance in the Balkans in July 1853?
**Blind Pred:** Easy

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
"What formal resolution resulted from Russian troops stopping their advance in the Balkans in July 1853?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace throug
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** What public outcry followed the Russian fleet's destruction of the Ottoman garrison at Sinop, and how did this event prompt France and Britain to rush forces to Gallipoli to prevent an Ottoman collaps
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to understand the public outcry following the destruction of the Ottoman garrison, connect it to the subsequent actions of France and Britain, and then relate these events to the Treaty of Paris, which is not directly mentioned in the context as a response to the outcry.
- **Filter Reason:** answer_consistency=judge_error; blind_fec=no

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the military engagement after the Russian forces stopped advancing at Silistra and the Ottoman forces successfully defended the town?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the military engagement, which is explicitly stated in sentence [26]. The answer can be found in that single sentence without needing to connect multiple pieces of information.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes

### Worst #2 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [relation_composition]

- **Question:** What formal resolution resulted from Russian troops stopping their advance in the Balkans in July 1853?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that mentions the Treaty of Paris, which is the formal resolution related to the events described.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes

### Worst #4 [relation_composition]

- **Question:** What formal resolution resulted from Russian troops stopping their advance in the Balkans in July 1853?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that mentions the Treaty of Paris, which is the formal resolution related to the events described.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [hidden_endpoint]

- **Question:** What specific agreement resulted from the combined Dutch-Johor effort that effectively destroyed the last bastion of Portuguese power in the Malay archipelago after the Dutch began their campaign and 
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the agreement mentioned in sentence S8, which states the Dutch agreed not to seek territories or wage war with the Malay kingdoms.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 40.0%
- **Top-1 per path: Answerable + FEC rate:** 60.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 15.8%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 60.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/5**

**RESULT: Hard potential exists, but no publishable Hard candidates passed the new filter.**
Hard rescue candidate generation shows potential; final Hard quality remains unvalidated.