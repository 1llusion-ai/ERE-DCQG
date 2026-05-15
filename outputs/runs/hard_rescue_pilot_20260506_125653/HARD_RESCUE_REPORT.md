# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 12:59
**Paths:** 2 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, missing_bridge, relation_composition
**K candidates per path per strategy:** 1
**Total candidates generated:** 6
**API calls:** generation=11, filter=6, judge=12

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 2 |
| Total candidates | 6 |
| Grammar pass | 6 |
| Generation errors | 0 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 6 |
| Blind Pred Hard | 0 (0.0%) |
| Blind Pred Medium | 6 (100.0%) |
| Blind Pred Easy | 0 (0.0%) |

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
| Total unique paths | 2 | — |
| Paths with >= 1 Blind Pred Hard | 0 | 0.0% |
| Paths with Blind Hard + answerable | 0 | 0.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 2 | 0 (0%) | 2 (100%) | 0 (0%) | 100% | 100% | 100% | 0% |
| missing_bridge | 2 | 0 (0%) | 2 (100%) | 0 (0%) | 100% | 100% | 0% | 0% |
| relation_composition | 2 | 0 (0%) | 2 (100%) | 0 (0%) | 100% | 50% | 100% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 6 |
| answer_consistency=no | 3 |
| answer_consistency=judge_error | 2 |
| single_sentence_answerable=partial | 2 |
| path_dependency=none | 2 |
| blind_fec=no | 1 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 6/6 (100%) | — |
| Blind Final-Event Consistent | 5/6 (83%) | — |
| PathDep Strong | 4/6 (67%) | — |
| Single-Sent Answerable=no | 4/6 (67%) | — |

## 8. Selected Top-1 Per Path

**2 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 2 | — |
| Top-1 Blind Pred Hard | 0 | 0.0% |
| Top-1 Answerable + FEC | 1 | 50.0% |
| Top-1 PathDep Strong | 2 | 100.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Medium | yes | no | strong | no | N | What was the consequence of the Russian fleet destroying the |
| 2 | relation_composition | Medium | yes | yes | strong | no | N | What long-term effect did the Ottoman success in stopping th |

## 9. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
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

## Question
"What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key question: HOW MANY sequential reasoning steps must the solver make to answer this question, given ONLY the context above?

Reply as a single JSON object:
{
  "predicted_
```

### Sample 2

**Question:** What was the ultimate consequence for the Russian troops after they stopped their advance at Silistra?
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

## Question
"What was the ultimate consequence for the Russian troops after they stopped their advance at Silistra?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key question: HOW MANY sequential reasoning steps must the solver make to answer this question, given ONLY the context above?

Reply as a single JSON object:
{
  "predicted_diffic
```

### Sample 3

**Question:** What was the consequence of the Russian fleet destroying the Turkish attempt to reinforce Kars, and how did this event influence the actions of France
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

## Question
"What was the consequence of the Russian fleet destroying the Turkish attempt to reinforce Kars, and how did this event influence the actions of France and Britain?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

The key question: HOW MANY sequential reasoning steps must the solver make to answer this question, given ONLY the context
```

## 10. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 11. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What was the ultimate consequence for the Ottoman forces after they stopped the Russian advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the stopping of the Russian advance at Silistra with the subsequent actions of the allies and the eventual signing of a treaty, which requires linking two pieces of information.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Medium; single_sentence_answerable=partial

### Worst #2 [missing_bridge]

- **Question:** What was the ultimate consequence for the Russian troops after they stopped their advance at Silistra?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** partial | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect the stopping of the Russian advance at Silistra with the subsequent actions of the allies and the eventual signing of a treaty, which involves linking multiple pieces of information across different sentences.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Medium; path_dependency=none

### Worst #3 [relation_composition]

- **Question:** What was the consequence of the Russian fleet destroying the Turkish attempt to reinforce Kars, and how did this event influence the actions of France and Britain?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect the destruction of the Turkish attempt to reinforce Kars (which leads to a consequence) and then relate that to the subsequent actions of France and Britain, necessitating two steps of reasoning.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium; blind_fec=no

### Worst #4 [hidden_endpoint]

- **Question:** How did the Russian occupation of the Danubian Principalities and the subsequent Ottoman defensive actions at Silistra and Sinop lead to the French and British forces rushing to Gallipoli, and what wa
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Medium | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect the Russian occupation leading to Ottoman defensive actions and the subsequent response from France and Britain, which involves understanding multiple events and their implications.
- **Filter Reason:** answer_consistency=no; blind_pred=Medium

### Worst #5 [missing_bridge]

- **Question:** What was the ultimate consequence for Russia after it stopped advancing in the Balkans in July 1853?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Medium | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** partial
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The solver needs to connect the event of Russia stopping its advance in the Balkans with the eventual outcome of the Treaty of Paris, which requires linking two pieces of information.
- **Filter Reason:** blind_pred=Medium; single_sentence_answerable=partial; path_dependency=none

## 12. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 0.0%
- **Top-1 per path: Answerable + FEC rate:** 50.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **All candidates: Blind Pred Hard rate:** 0.0%

- [FAIL] Top-1 Blind Pred Hard = 0.0% (need >= 20%)
- [FAIL] Top-1 Answerable + FEC = 50.0% (need >= 80%)
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Blind judge: zero Hard

**Criteria met: 1/4**

**RESULT: Pred Hard = 0. STOP 3-level Easy/Medium/Hard claim.**