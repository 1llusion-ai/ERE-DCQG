# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 17:55
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 3
**Total candidates generated:** 30
**API calls:** generation=64, filter=24, judge=72

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 30 |
| Grammar pass | 24 |
| Generation errors | 6 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 24 |
| Blind Pred Hard | 4 (16.7%) |
| Blind Pred Medium | 15 (62.5%) |
| Blind Pred Easy | 5 (20.8%) |

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
| Paths with >= 1 Blind Pred Hard | 3 | 75.0% |
| Paths with Blind Hard + answerable | 3 | 75.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 12 | 1 (8%) | 7 (58%) | 4 (33%) | 100% | 83% | 100% | 0% | 0% |
| relation_composition | 12 | 3 (25%) | 8 (67%) | 1 (8%) | 100% | 50% | 92% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 15 |
| answer_consistency=no | 15 |
| single_sentence_answerable=partial | 14 |
| alignment_asks=no | 12 |
| alignment_natural=no | 12 |
| target_drift=yes | 12 |
| blind_fec=no | 8 |
| blind_pred=Easy | 5 |
| single_sentence_answerable=yes | 5 |
| answer_consistency=judge_error | 4 |
| path_dependency=none | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 24/24 (100%) | — | — |
| Blind Final-Event Consistent | 16/24 (67%) | — | — |
| PathDep Strong | 23/24 (96%) | — | — |
| Single-Sent Answerable=no | 5/24 (21%) | — | — |
| Alignment asks=yes/partial | 12/24 (50%) | — | — |

## 8. Selected Top-1 Per Path

**4 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 4 | — |
| Top-1 Blind Pred Hard | 3 | 75.0% |
| Top-1 Answerable + FEC | 1 | 25.0% |
| Top-1 PathDep Strong | 4 | 100.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | relation_composition | Hard | yes | no | strong | no | N | N | What led to the French and British forces rushing to Gallipo |
| 2 | relation_composition | Medium | yes | yes | strong | partial | N | N | What resulted from the Anglo-Portuguese Army's Moving east i |
| 3 | hidden_endpoint | Hard | yes | no | strong | no | N | N | What public outcry and inquiry led to the Dutch beginning th |
| 4 | relation_composition | Hard | yes | no | strong | no | N | N | What restriction resulted from the crowded Myyrmanni shoppin |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 24/24
- **blind_context_contains_answer_sentence=false:** 0/24

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** How did the destruction of the Turkish attempt to reinforce the garrison at Sinop and the subsequent rush of French and British forces to Gallipoli im
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
"How did the destruction of the Turkish attempt to reinforce the garrison at Sinop and the subsequent rush of French and British forces to Gallipoli impact the outcome of the war in the Balkans?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Tr
```

### Sample 2

**Question:** How did the Ottoman defensive efforts at Silistra and the Russian defeat at Sinop influence the French and British forces to rush to Varna, and what w
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
"How did the Ottoman defensive efforts at Silistra and the Russian defeat at Sinop influence the French and British forces to rush to Varna, and what was the consequence of their arrival for the Russian forces at Silistra?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant informat
```

### Sample 3

**Question:** After the Russian fleet destroyed the Turkish attempt to reinforce the garrison at Sinop, what action did France and Britain take to prevent an Ottoma
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
"After the Russian fleet destroyed the Turkish attempt to reinforce the garrison at Sinop, what action did France and Britain take to prevent an Ottoman collapse?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/fa
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** What restriction resulted from the crowded Myyrmanni shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction resulting from the crowded mall, which requires understanding the bombing event, the investigation outcome, and the implications of Gerdt being the sole suspect. This involves multiple steps of reasoning across different sentences.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #2 [relation_composition]

- **Question:** What restriction resulted from the crowded Myyrmanni shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction resulting from the crowded mall, which requires understanding the bombing incident, the investigation, and the lack of indictments, necessitating multiple reasoning steps to connect these events.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #3 [relation_composition]

- **Question:** What led to the French and British forces rushing to Gallipoli, and how did the Ottoman defense at Silistra and the Russian naval destruction at Sinop influence this decision?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect multiple events: the Ottoman defense at Silistra, the Russian naval destruction at Sinop, and the subsequent actions of France and Britain, which involves tracing a complex chain of reasoning across several sentences.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #4 [hidden_endpoint]

- **Question:** What public outcry and inquiry led to the Dutch beginning their campaign to launch small incursions against the Portuguese, ultimately resulting in the siege of Malacca in 1606?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to trace multiple events: understanding the public outcry and inquiry, linking it to the Dutch campaign, and then connecting it to the siege of Malacca, which involves multiple sentences and reasoning steps.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** After the French were isolated and their center and left flank overcome, how did the initial state of their retreat appear from Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes the initial state of the French retreat.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #2 [hidden_endpoint]

- **Question:** What resulted from the French army's initial retreat after their center and left flank were overcome during the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial withdrawal of the French army, which is clearly stated in one sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=judge_error

### Worst #3 [hidden_endpoint]

- **Question:** What agreement resulted from the combined Dutch-Johor effort that led to the destruction of the last Portuguese stronghold in the Malay archipelago after the Dutch began their campaign and launched se
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch regarding their actions in the Malay kingdoms.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #4 [hidden_endpoint]

- **Question:** What agreement did the Dutch reach with Johor after launching their campaign to destroy Portuguese power in the East and effectively taking control of Malacca in 1640?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch with Johor.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #5 [relation_composition]

- **Question:** What restriction resulted from the Dutch's began in the campaign to destroy Portuguese power in the East?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the restriction agreed upon by the Dutch.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none; answer_consistency=no

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 75.0%
- **Top-1 per path: Answerable + FEC rate:** 25.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 16.7%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 25.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/6**

**RESULT: Blind Pred Hard candidates exist, but none pass relaxed Hard filter.**