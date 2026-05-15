# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 14:14
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=60, filter=15, judge=30

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 13 |
| Generation errors | 5 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 15 |
| Blind Pred Hard | 3 (20.0%) |
| Blind Pred Medium | 8 (53.3%) |
| Blind Pred Easy | 4 (26.7%) |

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
| Paths with Blind Hard + answerable | 2 | 40.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 7 | 2 (29%) | 3 (43%) | 2 (29%) | 100% | 43% | 57% | 0% |
| relation_composition | 8 | 1 (12%) | 5 (62%) | 2 (25%) | 100% | 88% | 38% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no | 9 |
| blind_pred=Medium | 8 |
| single_sentence_answerable=partial | 8 |
| path_dependency=none | 8 |
| blind_fec=no | 5 |
| blind_pred=Easy | 4 |
| single_sentence_answerable=yes | 4 |
| grammar=base: no question mark | 2 |
| answer_phrase=skipped (early exit) | 2 |
| answer_consistency=judge_error | 2 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 15/15 (100%) | — |
| Blind Final-Event Consistent | 10/15 (67%) | — |
| PathDep Strong | 7/15 (47%) | — |
| Single-Sent Answerable=no | 3/15 (20%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 2 | 40.0% |
| Top-1 Answerable + FEC | 3 | 60.0% |
| Top-1 PathDep Strong | 4 | 80.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What significant event occurred after the Russians were stop |
| 2 | relation_composition | Hard | yes | no | none | no | N | What public outcry and inquiry arose after the Russian fleet |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | How did the isolated Bayonne and subsequent Allied eastward  |
| 4 | hidden_endpoint | Hard | yes | no | strong | no | N | What public outcry and inquiry led to the Dutch beginning th |
| 5 | relation_composition | Medium | yes | yes | strong | partial | N | What was the outcome of the incident at the crowded Myyrmann |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 15/15
- **blind_context_contains_answer_sentence=false:** 0/15

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What outcry followed in response due to the Ottoman defeat in Silistra and the subsequent Russian reinforcement of their pérdida pérdida pérdida pérdi
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
"What outcry followed in response due to the Ottoman defeat in Silistra and the subsequent Russian reinforcement of their pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida loss pérdida in pérdida i
```

### Sample 2

**Question:** What significant event occurred after the Russians were stopped at Silistra and the Ottoman forces faced a siege at Kars, leading to a major naval con
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
"What significant event occurred after the Russians were stopped at Silistra and the Ottoman forces faced a siege at Kars, leading to a major naval confrontation that changed the course of the war?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Unde
```

### Sample 3

**Question:** What formal resolution ended the war after the French and British forces arrived in Varna?
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
"What formal resolution ended the war after the French and British forces arrived in Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of 
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [hidden_endpoint]

- **Question:** What public outcry and inquiry led to the Dutch beginning their campaign to destroy Portuguese power in the East, resulting in the siege of Malacca in 1606?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to trace the Dutch campaign's motivations, the siege of Malacca, and the agreement with Johor, which involves multiple interconnected events and facts across different sentences.
- **Filter Reason:** answer_consistency=no; blind_fec=no

## 12. Worst Samples

### Worst #1 [relation_composition]

- **Question:** What formal resolution ended the war after the French and British forces arrived in Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the war, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [hidden_endpoint]

- **Question:** What was the outcome of the combined Dutch-Johor effort against the Portuguese in Malacca after the naval battle of Cape Rachado in 1606, and how did the Dutch and Johor agree to proceed following the
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch after their victory.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [relation_composition]

- **Question:** What restriction or outcome resulted from the agreement between the Dutch and Johor after the Dutch started launching small incursions and skirmishes against the Portuguese?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch with Johor.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #4 [hidden_endpoint]

- **Question:** Why was the inquiry into the incident at the Myyrmanni shopping mall, where the crowd was exceptionally large and many were injured, and which resulted in the bombing by a single suspect, ultimately c
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S6, which states that the inquiry was closed without indictments as Gerdt was the sole suspect.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #5 [hidden_endpoint]

- **Question:** What outcry followed in response due to the Ottoman defeat in Silistra and the subsequent Russian reinforcement of their pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida pérdida
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** no | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect the Ottoman defeat at Silistra with the subsequent actions of France and Britain, which involves understanding the context of the war and the resulting Treaty of Paris. This requires linking two pieces of information, making it a medium difficulty question.
- **Filter Reason:** grammar=base: no question mark; answer_phrase=skipped (early exit); answer_consistency=no; blind_pred=Medium; blind_fec=no; single_sentence_answerable=partial; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 40.0%
- **Top-1 per path: Answerable + FEC rate:** 60.0%
- **Top-1 per path: PathDep Strong rate:** 80.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 20.0%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 60.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/5**

**RESULT: Hard potential exists, but no publishable Hard candidates passed the new filter.**
Hard rescue candidate generation shows potential; final Hard quality remains unvalidated.