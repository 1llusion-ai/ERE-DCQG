# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 18:55
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 3
**Total candidates generated:** 30
**API calls:** generation=84, filter=21, judge=63

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 30 |
| Grammar pass | 21 |
| Generation errors | 9 |
| Drift check failures | 22 |
| Drift repaired | 17 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 21 |
| Blind Pred Hard | 3 (14.3%) |
| Blind Pred Medium | 9 (42.9%) |
| Blind Pred Easy | 9 (42.9%) |

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
| Paths with >= 1 Blind Pred Hard | 2 | 50.0% |
| Paths with Blind Hard + answerable | 2 | 50.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 11 | 1 (9%) | 5 (45%) | 5 (45%) | 100% | 82% | 73% | 0% | 0% |
| relation_composition | 10 | 2 (20%) | 4 (40%) | 4 (40%) | 100% | 70% | 100% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no | 9 |
| blind_pred=Easy | 9 |
| single_sentence_answerable=yes | 9 |
| blind_pred=Medium | 9 |
| single_sentence_answerable=partial | 8 |
| blind_fec=no | 5 |
| alignment_asks=no | 5 |
| alignment_natural=no | 5 |
| target_drift=yes | 5 |
| answer_consistency=judge_error | 4 |
| path_dependency=none | 3 |
| trigger_leakage | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 21/21 (100%) | — | — |
| Blind Final-Event Consistent | 16/21 (76%) | — | — |
| PathDep Strong | 18/21 (86%) | — | — |
| Single-Sent Answerable=no | 4/21 (19%) | — | — |
| Alignment asks=yes/partial | 16/21 (76%) | — | — |

## 8. Selected Top-1 Per Path

**4 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 4 | — |
| Top-1 Blind Pred Hard | 2 | 50.0% |
| Top-1 Answerable + FEC | 2 | 50.0% |
| Top-1 PathDep Strong | 4 | 100.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | hidden_endpoint | Hard | yes | no | strong | no | N | N | How did the destruction of the Turkish attempt to reinforce  |
| 2 | relation_composition | Medium | yes | yes | strong | partial | N | N | What resulted from the Anglo-Portuguese Army's Moving east i |
| 3 | hidden_endpoint | Medium | yes | yes | strong | partial | N | N | What were the initial skirmishes and the significant siege t |
| 4 | relation_composition | Hard | yes | no | strong | no | N | N | What restriction resulted from The Myyrmanni's crowded shopp |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 21/21
- **blind_context_contains_answer_sentence=false:** 0/21

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** How did the destruction of the Turkish attempt to reinforce Kars impact the subsequent rush of French and British forces to Gallipoli and Varna?
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
"How did the destruction of the Turkish attempt to reinforce Kars impact the subsequent rush of French and British forces to Gallipoli and Varna?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the 
```

### Sample 2

**Question:** What limitation did the Treaty of Paris place on Russia following the conflict in the Balkans?
**Blind Pred:** Easy

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
"What limitation did the Treaty of Paris place on Russia following the conflict in the Balkans?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key question: HOW MANY sequential reas
```

### Sample 3

**Question:** How did the Ottoman defensive actions at Silistra and the destruction of the Turkish fleet at Sinop lead to French and British forces rushing to Galli
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
"How did the Ottoman defensive actions at Silistra and the destruction of the Turkish fleet at Sinop lead to French and British forces rushing to Gallipoli?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction resulting from the crowded shopping mall, which requires understanding the bombing incident, the investigation, and the lack of indictments, necessitating multiple reasoning steps to connect these events.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #2 [hidden_endpoint]

- **Question:** How did the destruction of the Turkish attempt to reinforce Kars impact the subsequent rush of French and British forces to Gallipoli and Varna?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to understand the impact of the destruction of the Turkish attempt to reinforce Kars, which involves tracing the sequence of events leading to the rush of French and British forces and the eventual outcome of the war, making it a complex reasoning task.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #3 [relation_composition]

- **Question:** How did the Russian fleet's destruction of the Turkish attempt to reinforce Kars and the subsequent Ottoman defense at Silistra influence French and British forces to rush to Gallipoli?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect multiple events: the destruction of the Turkish reinforcement attempt, the Ottoman defense at Silistra, and the subsequent actions of France and Britain, which involves tracing a complex chain of reasoning across several sentences.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=judge_error

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What limitation did the Treaty of Paris place on Russia following the conflict in the Balkans?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence [27], which states the limitation imposed by the Treaty of Paris on Russia.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [relation_composition]

- **Question:** What limitation was imposed on Russia following the Ottoman attempt to reinforce Kars being destroyed by the Russian fleet?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for a limitation imposed on Russia, which is explicitly stated in a single sentence (S27) following the context of the events leading to the Treaty of Paris.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #3 [hidden_endpoint]

- **Question:** What resulted from the French army's retreat after their center and left flank were overcome during the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the withdrawal of the French army, which is clearly stated in the context.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #4 [hidden_endpoint]

- **Question:** What resulted from the French army's initial attempt to retreat after their center and left flank were overcome at the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial withdrawal of the French army, which is clearly stated in one sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [relation_composition]

- **Question:** After the western-most Allied corps surrounded and isolated Bayonne, and the remaining two Allied corps pushed Soult's army back to Orthez, what happened to the French soldiers during their retreat?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes the French soldiers' retreat.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=judge_error

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 50.0%
- **Top-1 per path: Answerable + FEC rate:** 50.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 14.3%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 50.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/6**

**RESULT: Blind Pred Hard candidates exist, but none pass relaxed Hard filter.**