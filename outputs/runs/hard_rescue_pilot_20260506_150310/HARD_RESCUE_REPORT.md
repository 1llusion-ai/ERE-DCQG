# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 15:15
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=54, filter=17, judge=34

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 17 |
| Generation errors | 3 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 17 |
| Blind Pred Hard | 0 (0.0%) |
| Blind Pred Medium | 9 (52.9%) |
| Blind Pred Easy | 8 (47.1%) |

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
| hidden_endpoint | 8 | 0 (0%) | 4 (50%) | 4 (50%) | 100% | 88% | 88% | 0% |
| relation_composition | 9 | 0 (0%) | 5 (56%) | 4 (44%) | 100% | 78% | 89% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 9 |
| blind_pred=Easy | 8 |
| single_sentence_answerable=yes | 8 |
| answer_consistency=no | 7 |
| blind_fec=no | 3 |
| answer_consistency=judge_error | 3 |
| path_dependency=none | 2 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 17/17 (100%) | — |
| Blind Final-Event Consistent | 14/17 (82%) | — |
| PathDep Strong | 15/17 (88%) | — |
| Single-Sent Answerable=no | 1/17 (6%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 0 | 0.0% |
| Top-1 Answerable + FEC | 5 | 100.0% |
| Top-1 PathDep Strong | 5 | 100.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | hidden_endpoint | Easy | yes | yes | strong | yes | N | What formal resolution ended the conflict after the Russians |
| 2 | relation_composition | Medium | yes | yes | strong | no | N | What restriction resulted from Russian troops stopping their |
| 3 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What [specific result type] followed the Anglo-Portuguese Ar |
| 4 | hidden_endpoint | Easy | yes | yes | strong | yes | N | What significant actions did the Dutch take prior to agreein |
| 5 | relation_composition | Medium | yes | yes | strong | partial | N | What outcome resulted from the especially crowded Myyrmanni  |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 17/17
- **blind_context_contains_answer_sentence=false:** 0/17

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived just in time?
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
"What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived just in time?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information ac
```

### Sample 2

**Question:** What formal resolution ended the ongoing peace following the arrival of French and British forces at Varna in June 1854?
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
"What formal resolution ended the ongoing peace following the arrival of French and British forces at Varna in June 1854?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the contex
```

### Sample 3

**Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna?
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
"What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and the French and British forces arrived just in time?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which can be found in a single sentence (S26) that states the Treaty of Paris ended the war.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes

### Worst #2 [hidden_endpoint]

- **Question:** What formal resolution ended the ongoing peace following the arrival of French and British forces at Varna in June 1854?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that mentions the Treaty of Paris, which provides the formal resolution ending the war.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #3 [relation_composition]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #4 [hidden_endpoint]

- **Question:** After the Russian troops were stopped at Silistra and the Ottoman forces destroyed the Turkish attempt to reinforce Kars, what significant restriction was imposed on Russia following the Black Sea cam
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the restriction imposed on Russia, which is clearly mentioned in the context.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #5 [relation_composition]

- **Question:** What event occurred after the French center and left flank were overcome, and how did it affect the withdrawal of the French army?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the withdrawal of the French army after their center and left flank were overcome.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 0.0%
- **Top-1 per path: Answerable + FEC rate:** 100.0%
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