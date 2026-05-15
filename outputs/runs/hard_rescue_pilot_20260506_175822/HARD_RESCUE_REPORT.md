# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 18:15
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 3
**Total candidates generated:** 30
**API calls:** generation=89, filter=20, judge=60

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 30 |
| Grammar pass | 19 |
| Generation errors | 10 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 20 |
| Blind Pred Hard | 0 (0.0%) |
| Blind Pred Medium | 11 (55.0%) |
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
| Total unique paths | 4 | — |
| Paths with >= 1 Blind Pred Hard | 0 | 0.0% |
| Paths with Blind Hard + answerable | 0 | 0.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 9 | 0 (0%) | 6 (67%) | 3 (33%) | 100% | 89% | 78% | 0% | 0% |
| relation_composition | 11 | 0 (0%) | 5 (45%) | 6 (55%) | 100% | 64% | 82% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 11 |
| single_sentence_answerable=partial | 10 |
| answer_consistency=no | 10 |
| blind_pred=Easy | 9 |
| single_sentence_answerable=yes | 9 |
| blind_fec=no | 5 |
| alignment_natural=no | 5 |
| alignment_asks=no | 4 |
| target_drift=yes | 4 |
| path_dependency=none | 4 |
| grammar=base: word repetition: impact | 1 |
| answer_phrase=skipped (early exit) | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 20/20 (100%) | — | — |
| Blind Final-Event Consistent | 15/20 (75%) | — | — |
| PathDep Strong | 16/20 (80%) | — | — |
| Single-Sent Answerable=no | 1/20 (5%) | — | — |
| Alignment asks=yes/partial | 16/20 (80%) | — | — |

## 8. Selected Top-1 Per Path

**4 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 4 | — |
| Top-1 Blind Pred Hard | 0 | 0.0% |
| Top-1 Answerable + FEC | 3 | 75.0% |
| Top-1 PathDep Strong | 3 | 75.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | relation_composition | Medium | yes | no | strong | no | N | N | What led France and Britain to rush forces to Gallipoli afte |
| 2 | relation_composition | Medium | yes | yes | strong | partial | N | N | What were the consequences of the French center and left fla |
| 3 | hidden_endpoint | Medium | yes | yes | none | partial | N | N | What agreement resulted from the combined Dutch-Johor effort |
| 4 | hidden_endpoint | Medium | yes | yes | strong | partial | N | N | What [specific result] followed the especially crowded Myyrm |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 20/20
- **blind_context_contains_answer_sentence=false:** 0/20

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** After the Ottomans halted the Russian advance at Silistra and the Russian fleet destroyed the Turkish supply fleet at Sinop, what significant action d
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
"After the Ottomans halted the Russian advance at Silistra and the Russian fleet destroyed the Turkish supply fleet at Sinop, what significant action did France and Britain take that led to the cessation of Russian military advances in the Balkans?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
```

### Sample 2

**Question:** What led to the Russian fleet being forced to abandon their advance on Silistra, and what was the ultimate consequence for Russia's naval presence in 
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
"What led to the Russian fleet being forced to abandon their advance on Silistra, and what was the ultimate consequence for Russia's naval presence in the Black Sea?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events
```

### Sample 3

**Question:** After the Ottomans halted the Russian advance at Silistra and British and French forces rushed to Gallipoli, what significant restriction did the Trea
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
"After the Ottomans halted the Russian advance at Silistra and British and French forces rushed to Gallipoli, what significant restriction did the Treaty of Paris impose on Russia regarding their military actions in the region?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant inf
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [relation_composition]

- **Question:** After the Ottomans halted the Russian advance at Silistra and British and French forces rushed to Gallipoli, what significant restriction did the Treaty of Paris impose on Russia regarding their milit
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the restriction imposed by the Treaty of Paris on Russia.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #2 [hidden_endpoint]

- **Question:** What resulted from the French army's initial retreat after their center and left flank were overcome during the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial withdrawal of the French army, which is clearly stated in one sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #3 [hidden_endpoint]

- **Question:** After the French army was isolated and their center and left flank were overcome, how did the initial state of their retreat appear? Did it start in good order or did it quickly turn into a scramble f
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing sentence S2, which states that the withdrawal was initially conducted in good order.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #4 [hidden_endpoint]

- **Question:** After the French were isolated and driven back from several river lines, how did their initial state of retreat appear? 
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial state of the French withdrawal, requiring only one step to extract the information.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [relation_composition]

- **Question:** What was the consequence of the French center and left flank being overcome, and how did the withdrawal from Orthez initially proceed? 
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial conduct of the withdrawal, which is clearly stated in the context.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 0.0%
- **Top-1 per path: Answerable + FEC rate:** 75.0%
- **Top-1 per path: PathDep Strong rate:** 75.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 0.0%

- [FAIL] Top-1 Blind Pred Hard = 0.0% (need >= 20%)
- [FAIL] Top-1 Answerable + FEC = 75.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] All candidates: Blind Pred Hard = 0

**Criteria met: 1/6**

**RESULT: Pred Hard = 0. No Hard-difficulty evidence.**