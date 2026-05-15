# FEC Failure Analysis — Hard Rescue Pilot (2026-05-06 18:55)

## Purpose

Identify which Hard paths / answer types can produce natural Hard final-answer questions, and which should be filtered out.

## Data Source

Run: `outputs/runs/hard_rescue_pilot_20260506_184120/`

- 5 selected Hard paths, 2 strategies, K=3 → 30 candidates
- 21 grammar-passing candidates judged
- 3 Blind Pred Hard candidates (all with fec=no)

---

## 1. Blind Pred Hard Candidates — Full Detail

### Blind Hard #1

| Field | Value |
|-------|-------|
| doc_id | `81c576926e0c52f158b210c244028f0b` (Crimean War) |
| path_idx | 1 |
| strategy | hidden_endpoint |
| event_triggers | stopped → destroyed → rushed → forbade |
| event_types | Preventing_or_letting → Destroying → Motion → Preventing_or_letting |
| gold_answer_phrase | "It forbade Russia from basing warships in the Black Sea" |
| gold_event_type | Preventing_or_letting |
| answer_type (raw) | action |
| inferred_answer_type | preventing_or_letting |
| relation_group | CAUSE |
| relation_subtypes | TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE |
| support_span | 8 |
| non_temporal_count | 1 |

**Generated Question:** "How did the destruction of the Turkish attempt to reinforce Kars impact the subsequent rush of French and British forces to Gallipoli and Varna?"

| Metric | Value |
|--------|-------|
| Blind Pred | Hard |
| Blind FEC | no |
| Blind SSA | no |
| Blind Answerable | yes |
| Alignment asks | no |
| Alignment natural | no |
| Alignment drift | yes |
| Answer consistency | no |
| Path dependency | strong |

**Alignment Reason:** "The question asks about the impact of the destruction of the Turkish attempt to reinforce Kars on the rush of French and British forces, while the expected answer discusses the Treaty of Paris and its restrictions on Russia."

**Failure Category: A** — Question asks intermediate causal mechanism ("how did X impact Y"), gold answer is final outcome restriction ("It forbade Russia from..."). The question naturally elicits an explanation of causal steps, not a policy restriction.

---

### Blind Hard #2

| Field | Value |
|-------|-------|
| doc_id | `81c576926e0c52f158b210c244028f0b` (Crimean War) |
| path_idx | 1 |
| strategy | relation_composition |
| event_triggers | stopped → destroyed → rushed → forbade |
| event_types | Preventing_or_letting → Destroying → Motion → Preventing_or_letting |
| gold_answer_phrase | "It forbade Russia from basing warships in the Black Sea" |
| gold_event_type | Preventing_or_letting |
| answer_type (raw) | invalid |
| inferred_answer_type | preventing_or_letting |
| relation_group | CAUSE |
| relation_subtypes | TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE |
| support_span | 8 |
| non_temporal_count | 1 |

**Generated Question:** "How did the Russian fleet's destruction of the Turkish attempt to reinforce Kars and the subsequent Ottoman defense at Silistra influence French and British forces to rush to Gallipoli?"

| Metric | Value |
|--------|-------|
| Blind Pred | Hard |
| Blind FEC | no |
| Blind SSA | no |
| Blind Answerable | yes |
| Alignment asks | no |
| Alignment natural | no |
| Alignment drift | yes |
| Answer consistency | judge_error |
| Path dependency | strong |

**Alignment Reason:** "The question asks about the influence of specific military actions on the decision of French and British forces, while the expected answer relates to the terms of the Treaty of Paris, which is not directly about the military actions' influence."

**Failure Category: A** — Same root cause as #1. "How did X influence Y to Z?" expects a causal mechanism, but answer is a policy restriction. The question framing is fundamentally incompatible with the answer type.

---

### Blind Hard #3

| Field | Value |
|-------|-------|
| doc_id | `e253b7fd1109bd5f87966022eea7762f` (Myyrmanni bombing) |
| path_idx | 4 |
| strategy | relation_composition |
| event_triggers | crowded → took place → investigated → closed |
| event_types | Come_together → Process_start → Criminal_investigation → Self_motion |
| gold_answer_phrase | "closed in January 2003 without any indictments as Gerdt was the sole suspect" |
| gold_event_type | Self_motion |
| answer_type (raw) | outcome |
| inferred_answer_type | criminal_investigation |
| relation_group | CAUSE |
| relation_subtypes | TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION |
| support_span | 6 |
| non_temporal_count | 1 |

**Generated Question:** "What restriction resulted from The Myyrmanni's crowded shopping mall on October 11, 2002?"

| Metric | Value |
|--------|-------|
| Blind Pred | Hard |
| Blind FEC | no |
| Blind SSA | no |
| Blind Answerable | yes |
| Alignment asks | no |
| Alignment natural | no |
| Alignment drift | yes |
| Answer consistency | no |
| Path dependency | strong |

**Alignment Reason:** "The question asks about a restriction resulting from the crowded shopping mall, while the expected answer discusses the closure of the investigation without indictments, which is unrelated to the mall being crowded."

**Failure Category: B** — Question asks for a "restriction" but the answer is an investigation outcome ("closed ... without any indictments"). The question framing "What restriction resulted from X?" expects a policy/rule, but the answer is a procedural conclusion. The drift repair generated a restriction-type opening for a criminal_investigation answer.

---

## 2. Failure Category Summary

| Category | Count | Description |
|----------|------:|-------------|
| **A. Question asks intermediate cause, answer is final outcome** | 2 | "How did X impact/influence Y" expects causal mechanism; answer is restriction/policy |
| **B. Question asks wrong answer type** | 1 | "What restriction" expects policy; answer is investigation closure |
| C. Gold answer is weak/truncated | 0 | — |
| D. Final answer too locally explicit (Easy/SSA=yes) | 0 | (These are the Blind Easy candidates, not Hard) |
| E. Path relation mostly temporal | 0 | All 3 have CAUSE relation_group |
| F. Prompt wording error, fixable | 0 | — |
| G. Judge/evaluator false negative | 0 | — |

**Root cause: ALL 3 Blind Hard failures are question-answer type mismatch (A+B).**

---

## 3. Why Blind Hard = FEC Failure (Perfect Correlation)

All 21 judged candidates:

| Blind Pred | Count | FEC=yes | FEC=no | Alignment asks=yes | Alignment drift=yes |
|-----------|------:|--------:|-------:|-------------------:|--------------------:|
| Easy | 9 | 9 (100%) | 0 | 9 (100%) | 0 |
| Medium | 9 | 7 (78%) | 2 (22%) | 6 (67%) | 2 (22%) |
| Hard | 3 | 0 (0%) | 3 (100%) | 0 (0%) | 3 (100%) |

**Key insight:** Blind Hard and FEC=no are perfectly correlated. Every candidate rated Hard by the blind judge has fec=no. This is not coincidence — it's structural:

1. The blind judge rates questions Hard when they require tracing through intermediate events
2. But "tracing through intermediate events" means the question asks about causal chains
3. Causal chain questions ("how did X impact Y") naturally expect causal/mechanism answers
4. The gold answers are final outcomes (restrictions, policies, closures) — not causal mechanisms
5. Therefore the question-answer pair is semantically mismatched → FEC=no

**The question is Hard BECAUSE it drifts, and it drifts BECAUSE the question framing doesn't match the answer type.**

---

## 4. Path-Level Analysis

### Path 1: Crimean War — stopped → destroyed → rushed → forbade

| Property | Value |
|----------|-------|
| doc_id | `81c576926e0c52f158b210c244028f0b` |
| gold_answer_phrase | "It forbade Russia from basing warships in the Black Sea" |
| gold_event_type | Preventing_or_letting |
| inferred_answer_type | preventing_or_letting |
| relation_group | CAUSE |
| non_temporal_count | 1 |
| support_span | 8 |
| single_sentence_risk | high |
| llm_path_judge | path_questionable=yes, can_write_path_dep=yes |

**Verdict:** This path CAN produce a Hard question, but only if the question framing matches the answer type. "What restriction did the Treaty of Paris place on Russia?" is Easy (fec=yes). "How did the war lead to restrictions on Russia?" would need to be framed carefully. The answer type `preventing_or_letting` is promising for Hard if the question asks about the **scope** of the restriction (what exactly was forbidden, under what conditions) rather than a causal chain.

**Problem:** The current generation templates for Hard use causal framing ("how did X impact Y") which is incompatible with restriction answers.

### Path 2: Battle of Orthez — Moving → isolated → overcome → conducted

| Property | Value |
|----------|-------|
| doc_id | `db50381e7d1dd4a41fb4ac60eaebe3a4` |
| gold_answer_phrase | "At first the withdrawal was conducted in good" |
| gold_event_type | Action |
| inferred_answer_type | other |
| relation_group | CAUSE |
| non_temporal_count | 1 |
| support_span | 8 |
| single_sentence_risk | high |
| llm_path_judge | path_questionable=partial, can_write_path_dep=partial |

**Verdict:** This path has a **truncated answer phrase** ("At first the withdrawal was conducted in good" — missing "order"). The path judge flagged it as `partial`. No Blind Hard candidates were generated for this path — all were Easy. **Exclude from Hard.**

### Path 3: Battle of Malacca — began → launching → took → agreed

| Property | Value |
|----------|-------|
| doc_id | `3dcfd60153822a6a8f6a516f161fc506` |
| gold_answer_phrase | "agreed not to seek territories or wage war with the Malay kingdoms" |
| gold_event_type | Agree_or_refuse_to_act |
| inferred_answer_type | sign_agreement |
| relation_group | CAUSE |
| non_temporal_count | 1 |
| support_span | 7 |
| single_sentence_risk | high |
| llm_path_judge | path_questionable=yes, can_write_path_dep=yes |

**Verdict:** This path has a good answer phrase (clear agreement). All Blind candidates were Easy. The answer type `sign_agreement` could work for Hard if the question asks about the **terms** of the agreement in context of prior events. But the path relations are mostly temporal, making Hard generation difficult. **Borderline — keep but deprioritize.**

### Path 4: Myyrmanni bombing — crowded → took place → investigated → closed

| Property | Value |
|----------|-------|
| doc_id | `e253b7fd1109bd5f87966022eea7762f` |
| gold_answer_phrase | "closed in January 2003 without any indictments as Gerdt was the sole suspect" |
| gold_event_type | Self_motion |
| inferred_answer_type | criminal_investigation |
| relation_group | CAUSE |
| non_temporal_count | 1 |
| support_span | 6 |
| single_sentence_risk | high |
| llm_path_judge | path_questionable=yes, can_write_path_dep=yes |

**Verdict:** This path has a clear investigation outcome answer. The Blind Hard candidate asked "What restriction resulted from X?" — wrong framing. The correct Hard framing would be "How did the investigation into the bombing conclude, given that Gerdt was the sole suspect?" or "What was the final outcome of the murder investigation following the crowded mall bombing?" **Promising for Hard with correct answer-type guidance.**

### Path 5: Crimean War (variant) — same as Path 1 but different final event

Path 0 was filtered out by `check_hard_path_suitability` (date fragment: "signed on 30 March"). Correctly excluded.

---

## 5. Answer Type Analysis

| Answer Type | Inferred Count | Blind Hard | FEC=yes (any pred) | Natural Hard Potential |
|-------------|---------------:|-----------:|--------------------:|-----------------------|
| preventing_or_letting | 5 | 2 | 0 | **Low** — restriction answers don't match causal questions |
| sign_agreement | 5 | 0 | 3 | **Medium** — "what terms" works but Hard requires implicit chain |
| criminal_investigation | 5 | 1 | 2 | **High** — "how did investigation conclude" is natural Hard |
| other | 6 | 0 | 4 | **Low** — generic, no clear Hard template |

### Answer Type → Natural Question Framing

| Answer Type | Good Easy Question | Good Hard Question | Bad Hard Question |
|-------------|-------------------|-------------------|-------------------|
| preventing_or_letting | "What limitation did X place on Y?" | "What scope of restrictions emerged from the cascade of events starting with X?" | "How did X impact Y?" (expects causal mechanism) |
| sign_agreement | "What agreement did X and Y reach?" | "What formal terms concluded the conflict that began with X?" | "How did X influence Y?" (expects causal mechanism) |
| criminal_investigation | "How did the investigation conclude?" | "What was the final disposition of the case, given the evidence from X and Y?" | "What restriction resulted from X?" (wrong answer category) |

---

## 6. Weak Trigger / Answer Phrase Analysis

| Path | Trigger | Weak? | Answer Phrase | Truncated? | Phrase Length |
|------|---------|-------|---------------|------------|--------------:|
| 1 | signed on | no | "signed on 30 March" | yes (date fragment) | 4 words |
| 2 | forbade | no | "It forbade Russia from basing warships in the Black Sea" | no | 11 words |
| 3 | conducted | no | "At first the withdrawal was conducted in good" | yes (mid-sentence) | 9 words |
| 4 | agreed | no | "agreed not to seek territories or wage war with the Malay kingdoms" | no | 10 words |
| 5 | closed | no | "closed in January 2003 without any indictments as Gerdt was the sole suspect" | no | 13 words |

- Path 0 (signed on): answer phrase is date fragment → filtered by suitability check → **exclude from Hard**
- Path 3 (conducted): answer phrase truncated mid-sentence → **exclude from Hard**

---

## 7. Relation Group Analysis

| Relation Group | Paths | Blind Hard | FEC=yes Rate | Assessment |
|---------------|------:|-----------:|-------------:|------------|
| CAUSE | 5 | 3 | 0% (Hard), 78% (Medium) | Causal paths can produce Hard, but current templates fail |

All 5 paths are CAUSE/MIXED. The issue is not the relation group — it's how the templates use it.

---

## 8. Conclusions and Recommendations

### 8.1 Path Types to Exclude from Hard

| Path | Reason | Action |
|------|--------|--------|
| Path 0 (Crimean War, signed on) | Date fragment answer phrase | **Exclude** (already filtered) |
| Path 3 (Battle of Orthez, conducted) | Truncated answer phrase, path_judge=partial | **Exclude** |
| Path 2 (Crimean War, forbade) — with causal questions | Causal question framing incompatible with restriction answer | **Exclude unless question type is fixed** |

### 8.2 Answer/Event Types Promising for Hard

| Answer Type | Why | Template Needed |
|-------------|-----|-----------------|
| **criminal_investigation** | "How did the investigation conclude given X and Y?" naturally requires chain reasoning | Yes — investigation-disposition template |
| **sign_agreement** | "What terms concluded the conflict that began with X?" is natural Hard | Yes — agreement-terms-in-context template |
| **preventing_or_letting** | Only works if question asks about scope/conditions, not causal chain | Limited — needs careful framing |

### 8.3 Templates to Keep / Fix / Add

| Template | Status | Action |
|----------|--------|--------|
| hidden_endpoint causal ("How did X impact Y?") | **Broken** for restriction answers | Fix: detect answer type, switch to type-specific template |
| relation_composition causal ("How did X influence Y?") | **Broken** for restriction answers | Fix: same |
| "What restriction resulted from X?" | **Broken** for non-restriction answers | Fix: map to correct answer type |
| "How did the investigation conclude?" | **Missing** | **Add** for criminal_investigation |
| "What formal terms concluded X?" | **Missing** | **Add** for sign_agreement |
| "What scope of restrictions emerged from X?" | **Missing** | **Add** for preventing_or_letting |

### 8.4 Likely Judge False Negatives

**None identified.** All 3 FEC=no judgments are correct:
- #1 and #2: The questions ask about causal mechanisms, but the answer is a restriction. The judge correctly identifies the mismatch.
- #3: The question asks about a restriction, but the answer is an investigation closure. The judge correctly identifies the mismatch.

The alignment judge and FEC judge are working correctly. The problem is in generation, not evaluation.

---

## 9. Root Cause Summary

```
Blind Hard candidates
  → question uses causal framing ("how did X impact/influence Y")
  → this framing naturally elicits intermediate causal mechanisms
  → but the gold answer is a final outcome (restriction/agreement/closure)
  → question-answer type mismatch → FEC=no + alignment drift=yes
  → blind judge rates Hard BECAUSE the question requires chain reasoning
  → but the chain reasoning leads to the wrong answer type
```

**The fundamental issue:** Hard generation templates assume all Hard questions should use causal framing. But causal framing only works when the answer IS a causal mechanism. For restriction/agreement/investigation answers, the question must use type-specific framing that still requires chain reasoning.

**Next step:** Implement answer-type-aware Hard question templates that:
1. Detect the answer type (preventing_or_letting, sign_agreement, criminal_investigation, etc.)
2. Use type-specific question openings that naturally elicit that answer type
3. Still require chain reasoning (mentioning prior events implicitly)
4. Don't use causal framing for non-causal answers
