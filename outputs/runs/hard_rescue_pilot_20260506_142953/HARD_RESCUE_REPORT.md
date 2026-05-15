# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 14:44
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=66, filter=18, judge=36

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 17 |
| Generation errors | 2 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 18 |
| Blind Pred Hard | 3 (16.7%) |
| Blind Pred Medium | 9 (50.0%) |
| Blind Pred Easy | 6 (33.3%) |

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
| hidden_endpoint | 9 | 3 (33%) | 5 (56%) | 1 (11%) | 89% | 67% | 67% | 0% |
| relation_composition | 9 | 0 (0%) | 4 (44%) | 5 (56%) | 100% | 100% | 33% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no | 9 |
| path_dependency=none | 9 |
| blind_pred=Medium | 9 |
| single_sentence_answerable=partial | 8 |
| blind_pred=Easy | 6 |
| single_sentence_answerable=yes | 6 |
| blind_fec=no | 3 |
| answer_consistency=judge_error | 3 |
| hard_implicit=3 prior triggers explicitly in question (max 2 allowed) | 2 |
| grammar=broken_grammar: What the Russian | 1 |
| answer_phrase=skipped (early exit) | 1 |
| blind_answerable=no | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 17/18 (94%) | — |
| Blind Final-Event Consistent | 15/18 (83%) | — |
| PathDep Strong | 9/18 (50%) | — |
| Single-Sent Answerable=no | 4/18 (22%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 2 | 40.0% |
| Top-1 Answerable + FEC | 3 | 60.0% |
| Top-1 PathDep Strong | 3 | 60.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | hidden_endpoint | Hard | no | no | none | no | N | What the Russian fleet's at destroyed the Ottoman reinforcem |
| 2 | hidden_endpoint | Hard | yes | no | strong | no | N | What led to the Russian fleet destroying the Turkish attempt |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | What action resulted from the Allied army's Moving east and  |
| 4 | relation_composition | Easy | yes | yes | none | yes | N | What series of actions did the Dutch undertake to gain contr |
| 5 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What was the outcome of the especially crowded Myyrmanni sho |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 18/18
- **blind_context_contains_answer_sentence=false:** 0/18

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What the Russian fleet's at destroyed the Ottoman reinforcements at the Battle of Sin Varna in June 18 1854?
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
"What the Russian fleet's at destroyed the Ottoman reinforcements at the Battle of Sin Varna in June 18 1854?"

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

**Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Russian fleet destroying the Turkish attempt to reinforce Kars, leadi
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
"What was the consequence of the Ottoman forces being stopped at Silistra and the Russian fleet destroying the Turkish attempt to reinforce Kars, leading to the arrival of French and British forces in Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They mus
```

### Sample 3

**Question:** What formal resolution ended the conflict following the Russian abandonment of Silistra and the allied forces' arrival at Varna?
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
"What formal resolution ended the conflict following the Russian abandonment of Silistra and the allied forces' arrival at Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across th
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [hidden_endpoint]

- **Question:** What led to the Russian fleet destroying the Turkish attempt to reinforce the garrison at Sinop, and how did this event prompt France and Britain to rush forces to Gallipoli, subsequently moving north
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to trace multiple events: the Turkish attempt to reinforce the garrison, the destruction by the Russian fleet, and the subsequent actions of France and Britain, which involves connecting several sentences and understanding the sequence of events.
- **Filter Reason:** answer_consistency=no; blind_fec=no

## 12. Worst Samples

### Worst #1 [relation_composition]

- **Question:** What formal resolution ended the conflict following the Russian abandonment of Silistra and the allied forces' arrival at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the Treaty of Paris ended the war, which is a straightforward extraction of information.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [hidden_endpoint]

- **Question:** How did the Dutch's prolonged campaign and the combined Dutch-Johor effort lead to the effective destruction of Portuguese power in Malacca, and what were the key steps involved in this process?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch regarding their actions with the Malay kingdoms.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [relation_composition]

- **Question:** What series of actions did the Dutch undertake to gain control of Malacca, and how did their agreement with Johor influence the outcome?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch with Johor.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #4 [relation_composition]

- **Question:** What agreement did the Dutch make with Johor after effectively destroying the last bastion of Portuguese power in Malacca, following their initial naval engagement and the subsequent siege in 1606?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the information in sentence S8, which states the agreement made by the Dutch with Johor.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [relation_composition]

- **Question:** What was the outcome of the public outcry following the especially crowded Myyrmanni bombing in Myyrmäki, and how did the inquiry into the incident conclude?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S6, which states the conclusion of the inquiry into the incident.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 40.0%
- **Top-1 per path: Answerable + FEC rate:** 60.0%
- **Top-1 per path: PathDep Strong rate:** 60.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 16.7%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 60.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/5**

**RESULT: Hard potential exists, but no publishable Hard candidates passed the new filter.**
Hard rescue candidate generation shows potential; final Hard quality remains unvalidated.