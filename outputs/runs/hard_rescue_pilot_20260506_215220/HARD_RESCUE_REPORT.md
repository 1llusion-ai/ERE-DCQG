# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 22:21
**Paths:** 8 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 4
**Total candidates generated:** 64
**API calls:** generation=146, filter=29, judge=87

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 8 |
| Total candidates | 64 |
| Grammar pass | 28 |
| Generation errors | 35 |
| Drift check failures | 55 |
| Drift repaired | 31 |
| Too direct answer-type cue | 10 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |
| Unsupported answer type (skipped) | 2 |

## 1b. Answer Type Distribution

| Hard Answer Type | Template Family | N Candidates | Blind Hard | Blind Hard% | FEC yes/partial | FEC% | Strict Pass |
|-----------------|----------------|---:|----------:|----------:|----------------|-----:|------------:|
| agreement_resolution | agreement | 7 | 0 | 0% | 7 | 100% | 0 |
| casualty_damage | casualty | 8 | 0 | 0% | 8 | 100% | 0 |
| investigation_outcome | investigation | 8 | 1 | 12% | 6 | 75% | 0 |
| restriction_policy | restriction | 6 | 0 | 0% | 4 | 67% | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 29 |
| Blind Pred Hard | 1 (3.4%) |
| Blind Pred Medium | 12 (41.4%) |
| Blind Pred Easy | 16 (55.2%) |

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
| Total unique paths | 4 | — |
| Paths with >= 1 Blind Pred Hard | 1 | 25.0% |
| Paths with Blind Hard + answerable | 0 | 0.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 13 | 0 (0%) | 5 (38%) | 8 (62%) | 100% | 92% | 92% | 0% | 0% |
| relation_composition | 16 | 1 (6%) | 7 (44%) | 8 (50%) | 94% | 81% | 94% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Easy | 16 |
| single_sentence_answerable=yes | 16 |
| blind_pred=Medium | 12 |
| single_sentence_answerable=partial | 10 |
| answer_consistency=no | 9 |
| answer_consistency=judge_error | 7 |
| alignment_asks=no | 4 |
| alignment_natural=no | 4 |
| target_drift=yes | 4 |
| blind_fec=no | 4 |
| path_dependency=none | 2 |
| grammar=base: word repetition: what | 1 |
| answer_phrase=skipped (early exit) | 1 |
| blind_answerable=no | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 28/29 (97%) | — | — |
| Blind Final-Event Consistent | 25/29 (86%) | — | — |
| PathDep Strong | 27/29 (93%) | — | — |
| Single-Sent Answerable=no | 3/29 (10%) | — | — |
| Alignment asks=yes/partial | 25/29 (86%) | — | — |

## 8. Selected Top-1 Per Path

**4 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 4 | — |
| Top-1 Blind Pred Hard | 1 | 25.0% |
| Top-1 Answerable + FEC | 2 | 50.0% |
| Top-1 PathDep Strong | 4 | 100.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Answer Type | Template | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|------------|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | relation_composition | restriction_policy | restriction | Medium | yes | no | strong | no | N | N | What led to the French and British forces rushing  |
| 2 | hidden_endpoint | agreement_resolution | agreement | Medium | yes | yes | strong | partial | N | N | What was the consequence of the Dutch launching sm |
| 3 | relation_composition | investigation_outcome | investigation | Hard | no | no | strong | no | N | N | Why was the public outcry over the Myyrmanni bombi |
| 4 | hidden_endpoint | casualty_damage | casualty | Medium | yes | yes | strong | partial | N | N | What were the Russian commander's intentions and h |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 21/29
- **blind_context_contains_answer_sentence=false:** 8/29

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** How did the Russian occupation of the Danubian Principalities and the Battle of Sinop impact the independence of Wallachia and Moldavia?
**Blind Pred:** Medium

```
You are an expert difficulty evaluator for reading comprehension questions.

## Context
[S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman suzerainty, then began to cross the Danube.
[S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
[S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russian fleet at Sinop.
[S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
[S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
[S26] The Treaty of Paris, signed on 30 March 1856, ended the war.
[S27] It forbade Russia from basing warships in the Black Sea.
[S28] The Ottoman vassal states of Wallachia and Moldavia became largely independent.

## Question
"How did the Russian occupation of the Danubian Principalities and the Battle of Sinop impact the independence of Wallachia and Moldavia?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer


```

### Sample 2

**Question:** What significant change occurred in Russian military presence after the Ottoman forces successfully halted the Russian advance at Silistra and the Rus
**Blind Pred:** Medium

```
You are an expert difficulty evaluator for reading comprehension questions.

## Context
[S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman suzerainty, then began to cross the Danube.
[S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
[S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russian fleet at Sinop.
[S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
[S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
[S26] The Treaty of Paris, signed on 30 March 1856, ended the war.
[S27] It forbade Russia from basing warships in the Black Sea.
[S28] The Ottoman vassal states of Wallachia and Moldavia became largely independent.

## Question
"What significant change occurred in Russian military presence after the Ottoman forces successfully halted the Russian advance at Silistra and the Russian fleet was defeated at Sinop?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace throug
```

### Sample 3

**Question:** What event led to the French and British forces rushing to Gallipoli, and how did this action influence the subsequent Russian withdrawal from Silistr
**Blind Pred:** Medium

```
You are an expert difficulty evaluator for reading comprehension questions.

## Context
[S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman suzerainty, then began to cross the Danube.
[S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
[S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russian fleet at Sinop.
[S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
[S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
[S26] The Treaty of Paris, signed on 30 March 1856, ended the war.
[S27] It forbade Russia from basing warships in the Black Sea.
[S28] The Ottoman vassal states of Wallachia and Moldavia became largely independent.

## Question
"What event led to the French and British forces rushing to Gallipoli, and how did this action influence the subsequent Russian withdrawal from Silistra?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to re
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the Dutch East India Company's began and defined the Dutch stance towards the Malay kingdoms?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the Dutch agreed not to seek territories or wage war with the Malay kingdoms.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none; answer_consistency=judge_error

### Worst #2 [hidden_endpoint]

- **Question:** What formal resolution ended the Dutch and Johor conflict over Portuguese territories in the Malay archipelago after the Dutch began their campaign to destroy Portuguese power and launched several inc
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch regarding territories and warfare.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=judge_error

### Worst #3 [relation_composition]

- **Question:** What formal resolution ended the Dutch East India Company's began and defined the terms of their future interactions with Malay kingdoms?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the terms of the agreement made by the Dutch.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=judge_error

### Worst #4 [relation_composition]

- **Question:** What formal resolution ended the Dutch and Johor alliance against the Portuguese in Malacca after the Dutch began their campaign to destroy Portuguese power and launched small incursions and skirmishe
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch regarding their actions in relation to the Malay kingdoms.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #5 [relation_composition]

- **Question:** What terms did the Dutch agree with Johor after the Dutch began their campaign to destroy Portuguese power in the East and launched small incursions against the Portuguese in Malacca?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be found directly in sentence S8, which states the terms agreed upon by the Dutch with Johor.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 25.0%
- **Top-1 per path: Answerable + FEC rate:** 50.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 3.4%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 50.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/6**

**RESULT: Blind Pred Hard candidates exist, but none pass relaxed Hard filter.**