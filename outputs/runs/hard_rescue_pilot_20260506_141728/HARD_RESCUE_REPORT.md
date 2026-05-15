# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 14:27
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=48, filter=19, judge=38

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 19 |
| Generation errors | 1 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 19 |
| Blind Pred Hard | 0 (0.0%) |
| Blind Pred Medium | 13 (68.4%) |
| Blind Pred Easy | 6 (31.6%) |

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
| Paths with >= 1 Blind Pred Hard | 0 | 0.0% |
| Paths with Blind Hard + answerable | 0 | 0.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 9 | 0 (0%) | 6 (67%) | 3 (33%) | 100% | 89% | 33% | 0% |
| relation_composition | 10 | 0 (0%) | 7 (70%) | 3 (30%) | 100% | 100% | 30% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| path_dependency=none | 13 |
| blind_pred=Medium | 13 |
| answer_consistency=no | 11 |
| single_sentence_answerable=partial | 11 |
| blind_pred=Easy | 6 |
| single_sentence_answerable=yes | 6 |
| answer_consistency=judge_error | 5 |
| blind_fec=no | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 19/19 (100%) | — |
| Blind Final-Event Consistent | 18/19 (95%) | — |
| PathDep Strong | 6/19 (32%) | — |
| Single-Sent Answerable=no | 2/19 (11%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 0 | 0.0% |
| Top-1 Answerable + FEC | 5 | 100.0% |
| Top-1 PathDep Strong | 3 | 60.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Medium | yes | yes | none | no | N | What were the consequences of the Russian fleet's destructio |
| 2 | relation_composition | Medium | yes | yes | none | no | N | What restriction resulted from Russian troops stopping their |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | What action resulted from the Anglo-Portuguese Army's Moving |
| 4 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What were the initial steps taken by the Dutch to reduce Por |
| 5 | relation_composition | Medium | yes | yes | strong | partial | N | What led to the public outcry following the especially crowd |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 19/19
- **blind_context_contains_answer_sentence=false:** 0/19

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived in Varna?
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
"What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived in Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the
```

### Sample 2

**Question:** What were the consequences of the Russian fleet's destruction at Sinop and the French and British forces arriving at Varna in June 1854?
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
"What were the consequences of the Russian fleet's destruction at Sinop and the French and British forces arriving at Varna in June 1854?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information a
```

### Sample 3

**Question:** What formal resolution ended the conflict after the Ottoman forces successfully defended Silistra and the Russian attempt to reinforce the garrison at
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
"What formal resolution ended the conflict after the Ottoman forces successfully defended Silistra and the Russian attempt to reinforce the garrison at Kars was destroyed?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question a
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived in Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [relation_composition]

- **Question:** What formal resolution ended the conflict after the Ottoman forces successfully defended Silistra and the Russian attempt to reinforce the garrison at Kars was destroyed?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [hidden_endpoint]

- **Question:** After the western-most Allied corps surrounded and isolated Bayonne, and the remaining two Allied corps pushed Soult's army back to Orthez, how did the French retreat from Orthez end up?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes how the French retreat ended.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #4 [hidden_endpoint]

- **Question:** What specific agreement resulted from the Dutch's efforts to remove Portuguese influence in the Malay archipelago after their combined campaign with Johor in 1606?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the specific agreement mentioned in sentence S8, which states the Dutch agreed not to seek territories or wage war with the Malay kingdoms.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [relation_composition]

- **Question:** What restriction/outcome/action resulted from the Dutch's began in the campaign to destroy Portuguese power in the East?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the restriction agreed upon by the Dutch.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 0.0%
- **Top-1 per path: Answerable + FEC rate:** 100.0%
- **Top-1 per path: PathDep Strong rate:** 60.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 0.0%

- [FAIL] Top-1 Blind Pred Hard = 0.0% (need >= 20%)
- [PASS] Top-1 Answerable + FEC >= 80%
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] All candidates: Blind Pred Hard = 0

**Criteria met: 2/5**

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**