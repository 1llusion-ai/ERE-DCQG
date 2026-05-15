# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 16:09
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
| New Hard filter pass | 1 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 19 |
| Blind Pred Hard | 2 (10.5%) |
| Blind Pred Medium | 9 (47.4%) |
| Blind Pred Easy | 8 (42.1%) |

## 3. Blind Difficulty Judge — New Filter-Passing Candidates

| Metric | Value |
|--------|------:|
| Filter-passing candidates | 1 |
| Blind Pred Hard | 1 (100.0%) |
| Blind Pred Medium | 0 (0.0%) |
| Blind Pred Easy | 0 (0.0%) |

## 4. Path-Level Blind Pred Hard Yield

| Metric | Count | Rate |
|--------|------:|-----:|
| Total unique paths | 5 | — |
| Paths with >= 1 Blind Pred Hard | 2 | 40.0% |
| Paths with Blind Hard + answerable | 2 | 40.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 1 | 20.0% |
| Paths with >= 1 new-filter-pass | 1 | 20.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 9 | 1 (11%) | 3 (33%) | 5 (56%) | 100% | 89% | 89% | 10% |
| relation_composition | 10 | 1 (10%) | 6 (60%) | 3 (30%) | 100% | 60% | 70% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 9 |
| blind_pred=Easy | 8 |
| single_sentence_answerable=yes | 8 |
| answer_consistency=no | 6 |
| answer_consistency=judge_error | 6 |
| path_dependency=none | 4 |
| blind_fec=no | 4 |
| grammar=base: no question mark | 1 |
| answer_phrase=skipped (early exit) | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 19/19 (100%) | 1/1 (100%) |
| Blind Final-Event Consistent | 14/19 (74%) | 1/1 (100%) |
| PathDep Strong | 15/19 (79%) | 1/1 (100%) |
| Single-Sent Answerable=no | 3/19 (16%) | 1/1 (100%) |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 2 | 40.0% |
| Top-1 Answerable + FEC | 4 | 80.0% |
| Top-1 PathDep Strong | 4 | 80.0% |
| Top-1 New Filter Pass | 1 | 20.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Medium | yes | yes | none | partial | N | What formal resolution resulted from the Russian troops' sto |
| 2 | hidden_endpoint | Hard | yes | yes | strong | no | Y | How did the Ottoman's strong defensive campaign at Silistra  |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | What was the consequence of the French center and left flank |
| 4 | hidden_endpoint | Easy | yes | yes | strong | yes | N | What agreement did Johor and the Dutch reach after the Dutch |
| 5 | relation_composition | Hard | yes | no | strong | no | N | What restriction resulted from The Myyrmanni's crowded shopp |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 19/19
- **blind_context_contains_answer_sentence=false:** 0/19

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

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

### Sample 2

**Question:** What formal agreement ended the conflict after the Russian troops stopped advancing at Silistra and the Ottoman forces destroyed a Turkish attempt to 
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
"What formal agreement ended the conflict after the Russian troops stopped advancing at Silistra and the Ottoman forces destroyed a Turkish attempt to reinforce Kars?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2
```

### Sample 3

**Question:** What formal resolution resulted from the Russian troops' stopped advance in the Balkans in July 1853?
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
"What formal resolution resulted from the Russian troops' stopped advance in the Balkans in July 1853?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through 
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [hidden_endpoint]

- **Question:** How did the Ottoman's strong defensive campaign at Silistra and the subsequent destruction of their reinforcements at Sinop influence France and Britain to rush their forces to Gallipoli, and what was
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** no
- **New Hard Filter Pass:** True
- **Blind Judge Reason:** The question requires the solver to connect multiple events: the Ottoman defensive campaign at Silistra, the destruction of reinforcements at Sinop, and the subsequent actions of France and Britain, culminating in the Treaty of Paris. This involves tracing a complex chain of reasoning across several sentences.
- **Filter Reason:** all checks passed

### Best #2 [relation_composition]

- **Question:** What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction related to the crowded shopping mall, but the answer involves understanding the investigation's outcome, which requires connecting multiple pieces of information about the bombing, the suspect, and the investigation's closure.
- **Filter Reason:** answer_consistency=judge_error

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russians abandoned Silistra and French and British forces arrived at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [hidden_endpoint]

- **Question:** What formal agreement ended the conflict after the Russian troops stopped advancing at Silistra and the Ottoman forces destroyed a Turkish attempt to reinforce Kars?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal agreement that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #3 [relation_composition]

- **Question:** What formal resolution ended the conflict after the Russian fleet destroyed the Ottoman attempt to reinforce Kars, and France and Britain arrived just in time to witness the Russians abandon Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for the formal resolution that ended the conflict, which is explicitly stated in sentence [26]. The solver can find the answer by reading that single sentence.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Easy; single_sentence_answerable=yes

### Worst #4 [hidden_endpoint]

- **Question:** After the French were isolated and their center and left flank overcome, how did their initial state of retreat appear from Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes the initial state of the French retreat.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #5 [hidden_endpoint]

- **Question:** What was the consequence of the French center and left flank being overcome, and how did the withdrawal initially proceed? 
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial conduct of the withdrawal, which is clearly stated in the context.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Easy; single_sentence_answerable=yes

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 40.0%
- **Top-1 per path: Answerable + FEC rate:** 80.0%
- **Top-1 per path: PathDep Strong rate:** 80.0%
- **Top-1 per path: New Hard Filter Pass rate:** 20.0%
- **All candidates: Blind Pred Hard rate:** 10.5%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [PASS] Top-1 Answerable + FEC >= 80%
- [PASS] Top-1 PathDep Strong >= 50%
- [PASS] Top-1 New Hard Filter Pass >= 20%
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 5/5**

**RESULT: SUCCESS. All criteria met. Hard rescue fully validated.**