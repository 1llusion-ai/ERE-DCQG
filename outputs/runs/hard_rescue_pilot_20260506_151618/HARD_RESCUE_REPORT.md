# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 15:33
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=91, filter=18, judge=36

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 16 |
| Generation errors | 2 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 18 |
| Blind Pred Hard | 0 (0.0%) |
| Blind Pred Medium | 16 (88.9%) |
| Blind Pred Easy | 2 (11.1%) |

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
| hidden_endpoint | 9 | 0 (0%) | 8 (89%) | 1 (11%) | 100% | 100% | 89% | 0% |
| relation_composition | 9 | 0 (0%) | 8 (89%) | 1 (11%) | 100% | 89% | 78% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 16 |
| answer_consistency=no | 10 |
| path_dependency=none | 3 |
| hard_implicit=3 prior triggers explicitly in question (max 2 allowed) | 2 |
| blind_pred=Easy | 2 |
| single_sentence_answerable=yes | 2 |
| answer_phrase=skipped (early exit) | 2 |
| blind_fec=no | 1 |
| grammar=base: no question mark | 1 |
| grammar=base: word repetition: effort | 1 |
| answer_consistency=judge_error | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 18/18 (100%) | — |
| Blind Final-Event Consistent | 17/18 (94%) | — |
| PathDep Strong | 15/18 (83%) | — |
| Single-Sent Answerable=no | 3/18 (17%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 0 | 0.0% |
| Top-1 Answerable + FEC | 4 | 80.0% |
| Top-1 PathDep Strong | 5 | 100.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Medium | yes | no | strong | no | N | What led to the French and British forces moving to Varna, a |
| 2 | hidden_endpoint | Medium | yes | yes | strong | no | N | After the Ottomans destroyed the Turkish attempt to reinforc |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | How did the orderly French retreat from Bayonne turn eventua |
| 4 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What specific agreement resulted from the combined Dutch-Joh |
| 5 | hidden_endpoint | Medium | yes | yes | strong | partial | N | Why was the Myyrmanni shopping mall especially crowded on th |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 18/18
- **blind_context_contains_answer_sentence=false:** 0/18

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Turkish reinforcement being destroyed at Sinop before the French and 
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
"What was the consequence of the Ottoman forces being stopped at Silistra and the Turkish reinforcement being destroyed at Sinop before the French and British forces arrived at Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what th
```

### Sample 2

**Question:** What significant event led to the public outcry, and how did this relate to the inquiry into the conflict's unpopularity?
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
"What significant event led to the public outcry, and how did this relate to the inquiry into the conflict's unpopularity?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the conte
```

### Sample 3

**Question:** What led to the French and British forces moving to Varna, and how did this action relate to the Russian withdrawal from Silistra?
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
"What led to the French and British forces moving to Varna, and how did this action relate to the Russian withdrawal from Silistra?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across 
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [relation_composition]

- **Question:** What formal resolution ended the conflict after the French and British forces arrived at Varna and the Russians had already abandoned Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [hidden_endpoint]

- **Question:** What resulted from the combined Dutch-Joh effort effort after the siege of Malacac in 1646?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch after the siege.
- **Filter Reason:** grammar=base: word repetition: effort; answer_phrase=skipped (early exit); answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes

### Worst #3 [hidden_endpoint]

- **Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Turkish reinforcement being destroyed at Sinop before the French and British forces arrived at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the stopping of Ottoman forces at Silistra and the destruction of Turkish reinforcements at Sinop to the subsequent arrival of French and British forces, leading to the signing of the Treaty of Paris. This requires linking two events before arriving at the final consequence.
- **Filter Reason:** answer_consistency=no; hard_implicit=3 prior triggers explicitly in question (max 2 allowed); blind_pred=Medium

### Worst #4 [hidden_endpoint]

- **Question:** What significant event led to the public outcry, and how did this relate to the inquiry into the conflict's unpopularity?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect the unpopularity of the conflict with the signing of the Treaty of Paris, which involves understanding the context of public sentiment and the treaty's significance.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium; path_dependency=none

### Worst #5 [relation_composition]

- **Question:** What led to the French and British forces moving to Varna, and how did this action relate to the Russian withdrawal from Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect the movement of French and British forces to Varna with the Russian withdrawal from Silistra, which involves understanding the context of both events and their relationship, thus requiring two steps.
- **Filter Reason:** blind_pred=Medium; blind_fec=no

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 0.0%
- **Top-1 per path: Answerable + FEC rate:** 80.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 0.0%

- [FAIL] Top-1 Blind Pred Hard = 0.0% (need >= 20%)
- [PASS] Top-1 Answerable + FEC >= 80%
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] All candidates: Blind Pred Hard = 0

**Criteria met: 2/5**

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**