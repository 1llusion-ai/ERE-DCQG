# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 16:44
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=50, filter=20, judge=60

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 20 |
| Generation errors | 0 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 20 |
| Blind Pred Hard | 3 (15.0%) |
| Blind Pred Medium | 8 (40.0%) |
| Blind Pred Easy | 9 (45.0%) |

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
| Paths with >= 1 Blind Pred Hard | 3 | 60.0% |
| Paths with Blind Hard + answerable | 3 | 60.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 1 | 20.0% |
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 10 | 1 (10%) | 5 (50%) | 4 (40%) | 100% | 90% | 90% | 0% | 0% |
| relation_composition | 10 | 2 (20%) | 3 (30%) | 5 (50%) | 100% | 60% | 90% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| answer_consistency=no | 11 |
| blind_pred=Easy | 9 |
| single_sentence_answerable=yes | 9 |
| blind_pred=Medium | 8 |
| single_sentence_answerable=partial | 8 |
| alignment_asks=no | 8 |
| alignment_natural=no | 8 |
| target_drift=yes | 8 |
| blind_fec=no | 5 |
| answer_consistency=judge_error | 4 |
| path_dependency=none | 2 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 20/20 (100%) | — | — |
| Blind Final-Event Consistent | 15/20 (75%) | — | — |
| PathDep Strong | 18/20 (90%) | — | — |
| Single-Sent Answerable=no | 3/20 (15%) | — | — |
| Alignment asks=yes/partial | 12/20 (60%) | — | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 3 | 60.0% |
| Top-1 Answerable + FEC | 2 | 40.0% |
| Top-1 PathDep Strong | 5 | 100.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | relation_composition | Hard | yes | no | strong | no | N | N | What significant actions led to the Russian destruction of t |
| 2 | hidden_endpoint | Hard | yes | yes | strong | no | N | N | How did the Ottoman forces' strong defensive campaign at Sil |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | N | What resulted from the Anglo-Portuguese Army's Moving east i |
| 4 | hidden_endpoint | Medium | yes | no | strong | partial | N | N | What were the initial actions taken by the Dutch against the |
| 5 | relation_composition | Hard | yes | no | strong | no | N | N | What restriction resulted from The Myyrmanni's crowded shopp |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 20/20
- **blind_context_contains_answer_sentence=false:** 0/20

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What was the consequence of the Russian fleet destroying the Turkish reinforcement attempt at Sinop, and how did France and Britain respond?
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
"What was the consequence of the Russian fleet destroying the Turkish reinforcement attempt at Sinop, and how did France and Britain respond?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant informati
```

### Sample 2

**Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived in Varna?
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
"What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived in Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across
```

### Sample 3

**Question:** What led to the French and British forces arriving at Varna, and how did this affect the Russian withdrawal from Silistra?
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
"What led to the French and British forces arriving at Varna, and how did this affect the Russian withdrawal from Silistra?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the cont
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks about a restriction resulting from the crowded shopping mall, which requires understanding the bombing event, the investigation, and the lack of indictments, necessitating multiple reasoning steps to connect these facts.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #2 [relation_composition]

- **Question:** What significant actions led to the Russian destruction of the Ottoman garrison at Sinop, and how did this contribute to the growing public outcry in France and Britain?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to trace multiple events: the siege at Kars, the destruction of the Turkish reinforcement at Sinop, and the subsequent public reaction in France and Britain, which involves connecting several pieces of information across different sentences.
- **Filter Reason:** blind_fec=no; alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

### Best #3 [hidden_endpoint]

- **Question:** How did the Ottoman forces' strong defensive campaign at Silistra and the subsequent destruction of the Turkish reinforcement fleet at Sinop influence the French and British forces' decision to rush t
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** no
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect multiple events: the Ottoman defensive campaign at Silistra, the destruction of the Turkish fleet at Sinop, and the subsequent actions of the French and British forces, culminating in the Russian response. This involves tracing a complex chain of events rather than extracting a single fact.
- **Filter Reason:** alignment_asks=no; alignment_natural=no; target_drift=yes; answer_consistency=no

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived in Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [relation_composition]

- **Question:** What restriction did France and Britain impose on Russia after their forces rushed to Gallipoli and Varna, following the Russian occupation and the Ottoman defensive efforts?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence [S27], which states the restriction imposed on Russia.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #3 [relation_composition]

- **Question:** What restriction resulted from Russia's stopped advance in the Balkans in July 1853?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The answer can be directly found in sentence [S27], which states the restriction imposed on Russia as a result of the war.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=judge_error

### Worst #4 [hidden_endpoint]

- **Question:** What resulted from the French army's initial attempt to retreat after their center and left flank were overcome during the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial withdrawal of the French army, which is clearly stated in the context.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #5 [hidden_endpoint]

- **Question:** What was the consequence of the French center and left flank being overcome, and how did the withdrawal of the French army initially proceed? 
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the information in a single sentence about the withdrawal of the French army.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 60.0%
- **Top-1 per path: Answerable + FEC rate:** 40.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 15.0%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [FAIL] Top-1 Answerable + FEC = 40.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 3/6**

**RESULT: Blind Pred Hard candidates exist, but none pass relaxed Hard filter.**