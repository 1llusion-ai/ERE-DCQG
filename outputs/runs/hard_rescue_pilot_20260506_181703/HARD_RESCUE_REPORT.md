# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 18:30
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 3
**Total candidates generated:** 30
**API calls:** generation=73, filter=21, judge=63

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 30 |
| Grammar pass | 19 |
| Generation errors | 9 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 21 |
| Blind Pred Hard | 3 (14.3%) |
| Blind Pred Medium | 13 (61.9%) |
| Blind Pred Easy | 5 (23.8%) |

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
| Paths with Blind Hard + answerable | 2 | 50.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 10 | 1 (10%) | 7 (70%) | 2 (20%) | 100% | 70% | 90% | 0% | 0% |
| relation_composition | 11 | 2 (18%) | 6 (55%) | 3 (27%) | 91% | 73% | 82% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 13 |
| single_sentence_answerable=partial | 12 |
| answer_consistency=no | 10 |
| alignment_asks=no | 8 |
| alignment_natural=no | 8 |
| target_drift=yes | 8 |
| blind_fec=no | 6 |
| blind_pred=Easy | 5 |
| single_sentence_answerable=yes | 5 |
| answer_consistency=judge_error | 3 |
| path_dependency=none | 3 |
| grammar=base: no question mark | 2 |
| answer_phrase=skipped (early exit) | 2 |
| hard_implicit=3 prior triggers explicitly in question (max 2 allowed) | 1 |
| blind_answerable=no | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 20/21 (95%) | — | — |
| Blind Final-Event Consistent | 15/21 (71%) | — | — |
| PathDep Strong | 18/21 (86%) | — | — |
| Single-Sent Answerable=no | 4/21 (19%) | — | — |
| Alignment asks=yes/partial | 13/21 (62%) | — | — |

## 8. Selected Top-1 Per Path

**4 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 4 | — |
| Top-1 Blind Pred Hard | 3 | 75.0% |
| Top-1 Answerable + FEC | 1 | 25.0% |
| Top-1 PathDep Strong | 3 | 75.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | hidden_endpoint | Hard | yes | no | strong | no | N | N | What outcry arose when learning the Russian advance rejectio |
| 2 | relation_composition | Medium | yes | yes | strong | partial | N | N | What resulted from the Anglo-Portuguese Army's Moving east i |
| 3 | relation_composition | Hard | no | no | none | no | N | N | What public outcry and inquiry led to the Dutch launching sm |
| 4 | relation_composition | Hard | yes | no | strong | no | N | N | What restriction resulted from The Myyrmanni's crowded shopp |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 21/21
- **blind_context_contains_answer_sentence=false:** 0/21

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What outcry arose when learning the Russian advance rejection of their theirbading a fleet to reinforce the g on the Danube.
**Blind Pred:** Hard

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
"What outcry arose when learning the Russian advance rejection of their theirbading a fleet to reinforce the g on the Danube."

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key ques
```

### Sample 2

**Question:** After the Russians occupied the Danubian Principalities and the Ottoman fleet's attempt to reinforce Kars was destroyed by the Russians, what action d
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
"After the Russians occupied the Danubian Principalities and the Ottoman fleet's attempt to reinforce Kars was destroyed by the Russians, what action did France and Britain take in response to the potential Ottoman collapse?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant inform
```

### Sample 3

**Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Russian fleet destroying the Turkish reinforcement at Sinop, leading 
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
"What was the consequence of the Ottoman forces being stopped at Silistra and the Russian fleet destroying the Turkish reinforcement at Sinop, leading to France and Britain rushing forces to Gallipoli?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the contex
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction resulting from the crowded shopping mall, which requires understanding the context of the bombing, the investigation, and the lack of indictments. This involves multiple steps to connect the crowd's presence to the investigation's outcome.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #2 [hidden_endpoint]

- **Question:** What outcry arose when learning the Russian advance rejection of their theirbading a fleet to reinforce the g on the Danube.
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question is complex and requires the solver to understand the context of the Russian advance, the Ottoman response, and the implications of the Treaty of Paris, which involves multiple steps of reasoning across different sentences.
- **Filter Reason:** grammar=base: no question mark; answer_phrase=skipped (early exit); blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

## 12. Worst Samples

### Worst #1 [relation_composition]

- **Question:** In July 11 1 1 8 83,, Russian troops troops forces occupied the Danubian  on  n repercian  on  nian  on  nian  on  n  nian  n
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the Treaty of Paris forbade Russia from basing warships in the Black Sea, which is a straightforward extraction of information.
- **Filter Reason:** grammar=base: no question mark; answer_phrase=skipped (early exit); blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Worst #2 [hidden_endpoint]

- **Question:** What resulted from the French army's initial retreat after their center and left flank were overcome during the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial withdrawal of the French army, which is clearly stated in one sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [hidden_endpoint]

- **Question:** What agreement resulted from the combined Dutch-Johor effort that effectively destroyed the last bastion of Portuguese power in the Malay archipelago after the Dutch began their campaign and launched 
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the agreement mentioned in sentence S8, which states the terms agreed upon by the Dutch and Johor.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #4 [relation_composition]

- **Question:** What restriction/outcome/action resulted from the Dutch's began in the campaign to destroy Portuguese power in the East?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the agreement made by the Dutch.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=judge_error

### Worst #5 [relation_composition]

- **Question:** What restriction/outcome/action resulted from the Dutch's began in the campaign to destroy Portuguese power in the East?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence S8, which states the restriction agreed upon by the Dutch.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 75.0%
- **Top-1 per path: Answerable + FEC rate:** 25.0%
- **Top-1 per path: PathDep Strong rate:** 75.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 14.3%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 25.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/6**

**RESULT: Blind Pred Hard candidates exist, but none pass relaxed Hard filter.**