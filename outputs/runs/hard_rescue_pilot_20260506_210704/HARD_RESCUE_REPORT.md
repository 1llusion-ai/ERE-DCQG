# Hard Rescue Pilot Report (Blind Judge Edition)

**Date:** 2026-05-06 21:29
**Paths:** 8 (strict + relaxed Hard)
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 3
**Total candidates generated:** 48
**API calls:** generation=122, filter=24, judge=72

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 8 |
| Total candidates | 48 |
| Grammar pass | 24 |
| Generation errors | 24 |
| Drift check failures | 25 |
| Drift repaired | 19 |
| Strict Hard filter pass | 0 |
| Relaxed Hard filter pass | 0 |
| Unsupported answer type (skipped) | 2 |

## 1b. Answer Type Distribution

| Hard Answer Type | Template Family | N Candidates | Blind Hard | Blind Hard% | FEC yes/partial | FEC% | Strict Pass |
|-----------------|----------------|---:|----------:|----------:|----------------|-----:|------------:|
| agreement_resolution | agreement | 4 | 0 | 0% | 4 | 100% | 0 |
| casualty_damage | casualty | 5 | 0 | 0% | 5 | 100% | 0 |
| investigation_outcome | investigation | 5 | 0 | 0% | 5 | 100% | 0 |
| movement_action_outcome | action | 6 | 0 | 0% | 6 | 100% | 0 |
| restriction_policy | restriction | 4 | 0 | 0% | 2 | 50% | 0 |

## 2. Blind Difficulty Judge — All Candidates

| Metric | Value |
|--------|------:|
| Judged candidates | 24 |
| Blind Pred Hard | 0 (0.0%) |
| Blind Pred Medium | 6 (25.0%) |
| Blind Pred Easy | 18 (75.0%) |

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
| Paths with >= 1 strict filter pass | 0 | 0.0% |
| Paths with >= 1 relaxed filter pass | 0 | 0.0% |

## 5. Per-Strategy Comparison (Blind Judge)

| Strategy | N judged | Blind Hard | Blind Med | Blind Easy | Ans% | FEC% | PathDep Strong% | Strict Pass% | Relaxed Pass% |
|----------|--------:|----------:|---------:|----------:|-----:|-----:|----------------:|-------------:|--------------:|
| hidden_endpoint | 12 | 0 (0%) | 3 (25%) | 9 (75%) | 100% | 92% | 75% | 0% | 0% |
| relation_composition | 12 | 0 (0%) | 3 (25%) | 9 (75%) | 100% | 92% | 75% | 0% | 0% |

## 6. Strict Hard Filter Fail Reason Distribution

| Reason | Count |
|--------|------:|
| blind_pred=Easy | 18 |
| single_sentence_answerable=yes | 18 |
| answer_consistency=no | 9 |
| blind_pred=Medium | 6 |
| path_dependency=none | 6 |
| single_sentence_answerable=partial | 5 |
| answer_consistency=judge_error | 4 |
| blind_fec=no | 2 |
| alignment_asks=no | 2 |
| alignment_natural=no | 2 |
| target_drift=yes | 2 |

## 7. Quality Metrics Summary (Blind Judge)

| Metric | All Judged | Strict Pass | Relaxed Pass |
|--------|----------:|------------:|-------------:|
| Blind Answerable (yes/partial) | 24/24 (100%) | — | — |
| Blind Final-Event Consistent | 22/24 (92%) | — | — |
| PathDep Strong | 18/24 (75%) | — | — |
| Single-Sent Answerable=no | 1/24 (4%) | — | — |
| Alignment asks=yes/partial | 22/24 (92%) | — | — |

## 8. Selected Top-1 Per Path

**5 paths with selected top-1 candidates.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 5 | — |
| Top-1 Blind Pred Hard | 0 | 0.0% |
| Top-1 Answerable + FEC | 4 | 80.0% |
| Top-1 PathDep Strong | 4 | 80.0% |
| Top-1 Strict Filter Pass | 0 | 0.0% |
| Top-1 Relaxed Filter Pass | 0 | 0.0% |

### Top-1 Per Path Detail

| # | Strategy | Answer Type | Template | Blind Pred | Ans | FEC | PathDep | SSA | Strict | Relaxed | Question (truncated) |
|--:|----------|------------|----------|-----------:|-----|-----|---------|-----|--------|---------|---------------------|
| 1 | hidden_endpoint | restriction_policy | restriction | Medium | yes | no | strong | no | N | N | What led to the French and British forces rushing  |
| 2 | hidden_endpoint | movement_action_outcome | action | Medium | yes | yes | strong | partial | N | N | What were the consequences of the French army bein |
| 3 | hidden_endpoint | agreement_resolution | agreement | Easy | yes | yes | none | yes | N | N | What terms did the Dutch agree to with Johor in 16 |
| 4 | relation_composition | investigation_outcome | investigation | Medium | yes | yes | strong | partial | N | N | Why did the public so strongly protest the closure |
| 5 | relation_composition | casualty_damage | casualty | Easy | yes | yes | strong | yes | N | N | What final harm resulted from the uprising? |

## 9. Blind Context Audit

- **blind_context_contains_answer_sentence=true:** 19/24
- **blind_context_contains_answer_sentence=false:** 5/24

## 10. Blind Difficulty Judge Prompt Samples (for audit)

### Sample 1

**Question:** What final restriction was imposed on Russia after the Ottoman reinforcements were destroyed at Sinop and subsequent French and British military movem
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
"What final restriction was imposed on Russia after the Ottoman reinforcements were destroyed at Sinop and subsequent French and British military movements to Gallipoli and Varna?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a c
```

### Sample 2

**Question:** What led to the French and British forces rushing to Gallipoli, and how did the Russian fleet's actions at Sinop influence this decision?
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
"What led to the French and British forces rushing to Gallipoli, and how did the Russian fleet's actions at Sinop influence this decision?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace through a chain of events/facts to reach the answer

```

### Sample 3

**Question:** What final restriction was imposed on Russia after the Ottomans stopped the Russian advance at Silistra and a Turkish reinforcement attempt was destro
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
"What final restriction was imposed on Russia after the Ottomans stopped the Russian advance at Silistra and a Turkish reinforcement attempt was destroyed by the Russian fleet at Sinop?"

## Expected answer
"It forbade Russia from basing warships in the Black Sea"

## Task
Evaluate the REASONING DIFFICULTY of this question from the solver's perspective.

The solver does NOT know the answer. They must:
1. Understand what the question asks
2. Find relevant information across the context
3. Trace throu
```

## 11. Best Samples (Blind Hard + Answerable + PathDep)

*No Blind Hard + answerable + path-dependent candidates found.*

## 12. Worst Samples

### Worst #1 [hidden_endpoint]

- **Question:** What final restriction was imposed on Russia after the Ottoman reinforcements were destroyed at Sinop and subsequent French and British military movements to Gallipoli and Varna?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that states the restriction imposed on Russia, which is clearly mentioned in the context.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #2 [relation_composition]

- **Question:** What final restriction was imposed on Russia after the Ottomans stopped the Russian advance at Silistra and a Turkish reinforcement attempt was destroyed by the Russian fleet at Sinop?
- **Answer:** It forbade Russia from basing warships in the Black Sea
- **Event path:** stopped -> destroyed -> rushed -> forbade
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question directly asks for a specific restriction imposed on Russia, which is clearly stated in a single sentence (S27) of the context.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; answer_consistency=no

### Worst #3 [hidden_endpoint]

- **Question:** What happened to the French soldiers during their retreat after being pushed back to Orthez?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes the French soldiers' retreat.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none; answer_consistency=judge_error

### Worst #4 [relation_composition]

- **Question:** What happened to the retreat of the French army after their center and left flank were overcome?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** none | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing a single sentence (S2) that describes the initial orderly withdrawal of the French army.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes; path_dependency=none

### Worst #5 [relation_composition]

- **Question:** What happened to the French army following their center and left flank being overcome?
- **Answer:** At first the withdrawal was conducted in good
- **Event path:** Moving -> isolated -> overcome -> conducted
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes
- **Strict Hard Filter Pass:** False
- **Relaxed Hard Filter Pass:** False
- **Blind Judge Reason:** The question can be answered by directly referencing the sentence that describes the withdrawal of the French army after their flanks were overcome.
- **Filter Reason:** blind_pred=Easy; single_sentence_answerable=yes

## 13. Success Criteria Evaluation

- **Top-1 per path: Blind Pred Hard rate:** 0.0%
- **Top-1 per path: Answerable + FEC rate:** 80.0%
- **Top-1 per path: PathDep Strong rate:** 80.0%
- **Top-1 per path: Strict Hard Filter Pass rate:** 0.0%
- **Top-1 per path: Relaxed Hard Filter Pass rate:** 0.0%
- **All candidates: Blind Pred Hard rate:** 0.0%

- [FAIL] Top-1 Blind Pred Hard = 0.0% (need >= 20%)
- [PASS] Top-1 Answerable + FEC >= 80%
- [PASS] Top-1 PathDep Strong >= 50%
- [FAIL] Top-1 Strict Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] Top-1 Relaxed Hard Filter Pass = 0.0% (need >= 20%)
- [FAIL] All candidates: Blind Pred Hard = 0

**Criteria met: 2/6**

**RESULT: Pred Hard = 0. No Hard-difficulty evidence.**