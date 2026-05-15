# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 15:00
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=52, filter=19, judge=38

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
| Blind Pred Hard | 2 (10.5%) |
| Blind Pred Medium | 8 (42.1%) |
| Blind Pred Easy | 9 (47.4%) |

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
| hidden_endpoint | 9 | 2 (22%) | 3 (33%) | 4 (44%) | 100% | 67% | 44% | 0% |
| relation_composition | 10 | 0 (0%) | 5 (50%) | 5 (50%) | 100% | 90% | 30% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| path_dependency=none | 12 |
| blind_pred=Easy | 9 |
| single_sentence_answerable=yes | 9 |
| blind_pred=Medium | 8 |
| answer_consistency=no | 7 |
| answer_consistency=judge_error | 4 |
| blind_fec=no | 4 |
| grammar=base: no question mark | 1 |
| answer_phrase=skipped (early exit) | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 19/19 (100%) | — |
| Blind Final-Event Consistent | 15/19 (79%) | — |
| PathDep Strong | 7/19 (37%) | — |
| Single-Sent Answerable=no | 3/19 (16%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 2 | 40.0% |
| Top-1 Answerable + FEC | 3 | 60.0% |
| Top-1 PathDep Strong | 2 | 40.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Easy | yes | yes | none | yes | N | What formal resolution ended the conflict after the Russians |
| 2 | hidden_endpoint | Hard | yes | no | none | no | N | The destruction of the Turkish fleet by by the Russian fleet |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | What action resulted from the Anglo-Portuguese Army's Moving |
| 4 | hidden_endpoint | Hard | yes | no | none | no | N | What public outcry and inquiry led to the Dutch beginning th |
| 5 | relation_composition | Medium | yes | yes | strong | partial | N | What restriction or outcome resulted from the especially cro |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 19/19
- **blind_context_contains_answer_sentence=false:** 0/19

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What formal resolution ended the conflict after the Russian forces stopped their advance at Silistra and the Ottoman forces successfully defended agai
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
"What formal resolution ended the conflict after the Russian forces stopped their advance at Silistra and the Ottoman forces successfully defended against the Russian siege at Kars?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the 
```

### Sample 2

**Question:** What formal resolution ended the conflict after the Russian forces stopped at Silistra and the Ottoman attempt to reinforce Kars was destroyed by the 
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
"What formal resolution ended the conflict after the Russian forces stopped at Silistra and the Ottoman attempt to reinforce Kars was destroyed by the Russian fleet?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2.
```

### Sample 3

**Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna in June 1854?
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
"What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna in June 1854?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant informati
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russian forces stopped their advance at Silistra and the Ottoman forces successfully defended against the Russian siege at Kars?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russian forces stopped at Silistra and the Ottoman attempt to reinforce Kars was destroyed by the Russian fleet?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The answer can be extracted from that single sentence without needing to connect multiple pieces of information.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [relation_composition]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna in June 1854?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #4 [relation_composition]

- **Question:** What formal outcryon ended the on and Britain rushed forces to Gallipinous in June 1854 and arriving on for the Russians to abandon Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the date of the Treaty of Paris, which is explicitly stated in sentence [26]. The solver can find the answer in one step without needing to connect multiple pieces of information.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [hidden_endpoint]

- **Question:** After the Russian troops were stopped at Silistra and the Ottoman forces destroyed a Turkish attempt to reinforce Kars, what significant restriction was imposed on Russia following the conclusion of t
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the restriction imposed on Russia after the war, which is clearly stated in one sentence.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 40.0%
- **Top-1 per path: Answerable + FEC rate:** 60.0%
- **Top-1 per path: PathDep Strong rate:** 40.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 10.5%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 60.0% (need >= 80%)
- [FAIL] Top-1 PathDep Strong = 40.0% (need >= 50%)
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 2/5**

**RESULT: Hard potential exists, but no publishable Hard candidates passed the new filter.**
Hard rescue candidate generation shows potential; final Hard quality remains unvalidated.