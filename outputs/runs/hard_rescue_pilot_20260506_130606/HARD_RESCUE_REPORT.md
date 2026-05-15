# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 13:07
**Paths:** 2 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, missing_bridge, relation_composition
**K candidates per path per strategy:** 1
**Total candidates generated:** 6
**API calls:** generation=9, filter=6, judge=12

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 2 |
| Total candidates | 6 |
| Grammar pass | 6 |
| Generation errors | 0 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 6 |
| Blind Pred Hard | 1 (16.7%) |
| Blind Pred Medium | 5 (83.3%) |
| Blind Pred Easy | 0 (0.0%) |

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
| Total unique paths | 2 | — |
| Paths with >= 1 Blind Pred Hard | 1 | 50.0% |
| Paths with Blind Hard + answerable | 1 | 50.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 1 | 50.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 2 | 0 (0%) | 2 (100%) | 0 (0%) | 100% | 100% | 50% | 0% |
| missing_bridge | 2 | 0 (0%) | 2 (100%) | 0 (0%) | 100% | 100% | 50% | 0% |
| relation_composition | 2 | 1 (50%) | 1 (50%) | 0 (0%) | 100% | 100% | 100% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no | 5 |
| blind_pred=Medium | 5 |
| single_sentence_answerable=partial | 4 |
| path_dependency=none | 2 |
| answer_consistency=judge_error | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 6/6 (100%) | — |
| Blind Final-Event Consistent | 6/6 (100%) | — |
| PathDep Strong | 4/6 (67%) | — |
| Single-Sent Answerable=no | 2/6 (33%) | — |

## 8. Selected Top-1 Per Path

**2 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 2 | — |
| Top-1 Blind Pred Hard | 1 | 50.0% |
| Top-1 Answerable + FEC | 2 | 100.0% |
| Top-1 PathDep Strong | 2 | 100.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Medium | yes | yes | strong | partial | N | What was the consequence for Russia following its troops sto |
| 2 | relation_composition | Hard | yes | yes | strong | no | N | How did the Russian occupation of the Danubian Principalitie |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 6/6
- **blind_context_contains_answer_sentence=false:** 0/6

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
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
"What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace t
```

### Sample 2

**Question:** What was the ultimate consequence for the Danubian Principalities after Russian troops stopped their advance?
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
"What was the ultimate consequence for the Danubian Principalities after Russian troops stopped their advance?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace 
```

### Sample 3

**Question:** What was the consequence for Russia following its troops stopping in the Balkans in July 1853?
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
"What was the consequence for Russia following its troops stopping in the Balkans in July 1853?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** How did the Russian occupation of the Danubian Principalities and the subsequent Ottoman defensive campaign at Silistra set the stage for the Russian fleet's destruction at Sinop and the subsequent ru
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect multiple events: the Russian occupation leading to the Ottoman defensive campaign, the siege at Kars, the destruction of the Turkish reinforcement by the Russian fleet, and the subsequent rush of French and British forces. This involves tracing a complex chain of events across several sentences.
- **Filter Reason:** answer_consistency=no

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the stopping of the Russian advance at Silistra with the subsequent events leading to the Treaty of Paris, which signifies the ultimate consequence for the Ottoman forces.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium; single_sentence_answerable=partial; path_dependency=none

### Worst #2 [missing_bridge]

- **Question:** What was the ultimate consequence for the Danubian Principalities after Russian troops stopped their advance?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the occupation of the Danubian Principalities by Russian troops and the subsequent events leading to the Treaty of Paris to understand the ultimate consequence.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium; single_sentence_answerable=partial

### Worst #3 [relation_composition]

- **Question:** What was the consequence for Russia following its troops stopping in the Balkans in July 1853?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver must first identify the initial event of Russian troops stopping in the Balkans and then connect it to the subsequent consequence of the Treaty of Paris, which requires linking two separate pieces of information.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium; single_sentence_answerable=partial

### Worst #4 [hidden_endpoint]

- **Question:** How did the Ottoman defensive efforts at Silistra and the subsequent Russian defeat at Sinop influence the British and French decision to rush forces to Gallipoli, and what was the ultimate impact on 
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the Ottoman defensive efforts at Silistra and the Russian defeat at Sinop to understand why France and Britain rushed forces to Gallipoli, and then link that to the Treaty of Paris which forbade Russia from basing warships in the Black Sea.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium

### Worst #5 [missing_bridge]

- **Question:** What was the ultimate consequence for Russia after its troops stopped at Silistra in the Balkans in July 1853?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the stopping of Russian troops at Silistra to the eventual outcome of the Treaty of Paris, which requires linking two separate pieces of information.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Medium; single_sentence_answerable=partial; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 50.0%
- **Top-1 per path: Answerable + FEC rate:** 100.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **All candidates: Blind Pred Hard rate:** 16.7%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [PASS] Top-1 Answerable + FEC >= 80%
- [PASS] Top-1 PathDep Strong >= 50%
- [PASS] Blind judge: non-zero Hard

**Criteria met: 4/4**

**RESULT: SUCCESS. All criteria met. 3-level difficulty-control claim is viable.**