# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 15:58
**Paths:** 5 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 2
**Total candidates generated:** 20
**API calls:** generation=48, filter=20, judge=40

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 5 |
| Total candidates | 20 |
| Grammar pass | 20 |
| Generation errors | 0 |
| New Hard filter pass | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 20 |
| Blind Pred Hard | 3 (15.0%) |
| Blind Pred Medium | 11 (55.0%) |
| Blind Pred Easy | 6 (30.0%) |

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
| Paths with >= 1 Blind Pred Hard | 1 | 20.0% |
| Paths with Blind Hard + answerable | 1 | 20.0% |
| Paths with Blind Hard + ans + fec + pathdep strong | 0 | 0.0% |
| Paths with >= 1 new-filter-pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | New Filter Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-----------------:|
| hidden_endpoint | 10 | 1 (10%) | 6 (60%) | 3 (30%) | 100% | 70% | 80% | 0% |
| relation_composition | 10 | 2 (20%) | 5 (50%) | 3 (30%) | 100% | 70% | 80% | 0% |

## 6. New Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Medium | 11 |
| answer_consistency=no | 11 |
| blind_fec=no | 6 |
| blind_pred=Easy | 6 |
| single_sentence_answerable=yes | 6 |
| path_dependency=none | 4 |
| answer_consistency=judge_error | 2 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | New Filter-Passing |
|--------|----------:|-------------------:|
| Blind Answerable (yes/partial) | 20/20 (100%) | — |
| Blind Final-Event Consistent | 14/20 (70%) | — |
| PathDep Strong | 16/20 (80%) | — |
| Single-Sent Answerable=no | 7/20 (35%) | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 1 | 20.0% |
| Top-1 Answerable + FEC | 4 | 80.0% |
| Top-1 PathDep Strong | 5 | 100.0% |
| Top-1 New Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Blind Pred | Ans | FEC | PathDep | SSA | NewFilter | Question (truncated) |
|--:|----------|-----------:|-----|-----|---------|-----|-----------|---------------------|
| 1 | relation_composition | Medium | yes | yes | strong | no | N | What was the consequence of the Ottoman forces being stopped |
| 2 | hidden_endpoint | Medium | yes | yes | strong | no | N | What was the consequence of the Ottoman forces being stopped |
| 3 | relation_composition | Medium | yes | yes | strong | partial | N | What resulted from the Anglo-Portuguese Army's Moving eastwa |
| 4 | hidden_endpoint | Medium | yes | yes | strong | partial | N | What public outcry and inquiry led to the Dutch launching sm |
| 5 | relation_composition | Hard | yes | no | strong | no | N | What restriction resulted from The Myyrmanni's crowded shopp |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 20/20
- **blind_context_contains_answer_sentence=false:** 0/20

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What were the consequences of the Russian fleet destroying the Turkish reinforcement attempt at Sinop, and how did this lead to France and Britain rus
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
"What were the consequences of the Russian fleet destroying the Turkish reinforcement attempt at Sinop, and how did this lead to France and Britain rushing forces to Gallipoli? "

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the ques
```

### Sample 2

**Question:** What formal resolution ended the conflict after the Russians left Silistra and the Ottoman forces arrived at Varna?
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
"What formal resolution ended the conflict after the Russians left Silistra and the Ottoman forces arrived at Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. 
```

### Sample 3

**Question:** What was the consequence of the Ottoman forces being stopped at Silistra and the Turkish reinforcement being destroyed by the Russian fleet at Sinop, 
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
"What was the consequence of the Ottoman forces being stopped at Silistra and the Turkish reinforcement being destroyed by the Russian fleet at Sinop, leading to France and Britain's decision to move their forces to Varna?"

## Expected answer
"signed on 30 March"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the a
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

### Best #1 [relation_composition]

- **Question:** What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction related to the crowded shopping mall, which requires understanding the bombing incident, the investigation, and the lack of indictments, necessitating multiple reasoning steps to connect these elements.
- **Filter Reason:** blind_fec=no

### Best #2 [relation_composition]

- **Question:** What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question asks for a restriction resulting from the crowded shopping mall, which requires understanding the bombing incident, the investigation, and the lack of indictments, necessitating multiple reasoning steps to connect these events.
- **Filter Reason:** blind_fec=no

### Best #3 [hidden_endpoint]

- **Question:** Why was the public outcry so intense after the Myyrmanni shopping mall bombing on October 11, 2002, and how did the shopping center's unusual crowdedness with 1,000–2,000 people, including many childr
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Hard | **PathDep:** strong | **Answerable:** yes | **FEC:** no | **SSA:** no
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question requires the solver to connect multiple pieces of information about the bombing, the number of victims, and the implications of the crowdedness, which involves tracing through several sentences and understanding the context deeply.
- **Filter Reason:** blind_fec=no

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What formal resolution ended the conflict after the Russians left Silistra and the Ottoman forces arrived at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the Treaty of Paris ended the war, which is a straightforward extraction of information.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #2 [relation_composition]

- **Question:** What formal resolution ended the conflict after the Russian fleet destroyed the Ottoman forces at Sinop and French and British troops arrived at Varna?
- **Answer:** signed on 30 March
- **Event path:** stopped -> destroyed -> arriving -> signed on
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the Treaty of Paris ended the war, which is a single step.
- **Filter Reason:** answer_consistency=no; blind_pred=Easy; single_sentence_answerable=yes

### Worst #3 [hidden_endpoint]

- **Question:** What resulted from the French army's initial retreat after their center and left flank were overcome during the Battle of Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the initial withdrawal of the French army, which is clearly stated in one sentence.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #4 [hidden_endpoint]

- **Question:** After the French were isolated and their center and left flank were overcome, how did their initial state of retreat appear? 
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes the initial state of the French retreat.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

### Worst #5 [relation_composition]

- **Question:** What restriction resulted from the Dutch's began in the campaign to destroy Portuguese power in the East?
- **Answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **Event path:** began -> launching -> took -> agreed
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **New Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the Dutch agreed not to seek territories or wage war with the Malay kingdoms, which is clearly mentioned in the context.
- **Filter Reason:** answer_consistency=judge_error; blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 20.0%
- **Top-1 per path: Answerable + FEC rate:** 80.0%
- **Top-1 per path: PathDep Strong rate:** 100.0%
- **Top-1 per path: New Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 15.0%

- [PASS] Top-1 Blind Pred Hard >= 20%
- [PASS] Top-1 Answerable + FEC >= 80%
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 New Hard Filter Pass = 0.0% (need >= 20%)
- [PASS] All candidates: Blind Pred Hard > 0

**Criteria met: 4/5**

**RESULT: Hard potential exists, but no publishable Hard candidates passed the new filter.**
Hard rescue candidate generation shows potential; final Hard quality remains unvalidated.