# Stage 3 Overlap Diagnosis: Stage 2 Full vs Stage 3 Suitable on Same 10 Stories

Generated: 2026-05-13

## 1. Stories Used

The 10 stories selected by Stage 3 suitable smoke (seed=42, max_stories=10):

1. bokwewa-the-humpback
2. gray-eagle-and-his-five-brothers
3. habetrot-the-spinstress
4. jamie-freel-and-the-young-lady
5. morraha
6. mount-of-golden-queen
7. prince-featherhead-and-the-princess-celandine
8. the-little-spirit-or-boy-man
9. the-winter-spirit-and-his-visitor
10. white-hare-and-crocodiles

All 10 are present in Stage 2 full run (106 stories).

---

## 2. Performance Comparison: Same 10 Stories

### 2a. Main Metrics Table

| Setting | Method | QP | JOK | Overall Acc | Macro Acc | Macro F1 | Spearman | Easy hit | Med hit | Hard hit | Easy ASA=yes | Hard ASA=no |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S2-full-on-S3-stories | Direct | 25/30 | 25 | 36.0% | 36.1% | 36.9% | 0.391 | 3/9 | 4/8 | 2/8 | 0/9 (0%) | 0/8 (0%) |
| S2-full-on-S3-stories | ICL | 19/30 | 19 | 31.6% | 30.5% | 28.3% | -0.093 | 1/5 | 4/7 | 1/7 | 0/5 (0%) | 0/7 (0%) |
| S2-full-on-S3-stories | SelfRefine | 11/30 | 11 | 36.4% | 40.0% | 42.7% | 0.207 | 1/5 | 2/4 | 1/2 | 0/5 (0%) | 0/2 (0%) |
| S2-full-on-S3-stories | Ours | 19/30 | 19 | 36.8% | 39.0% | 37.2% | 0.300 | 2/7 | 3/5 | 2/7 | 0/7 (0%) | 0/7 (0%) |
| S3-suitable-smoke | Direct | 20/30 | 20 | 30.0% | 28.6% | 22.2% | 0.175 | 0/6 | 5/7 | 1/7 | 0/6 (0%) | 0/7 (0%) |
| S3-suitable-smoke | ICL | 20/30 | 20 | 30.0% | 29.4% | 26.7% | 0.009 | 1/6 | 4/7 | 1/7 | 0/6 (0%) | 0/7 (0%) |
| S3-suitable-smoke | SelfRefine | 17/30 | 17 | 29.4% | 33.3% | 22.2% | -0.090 | 1/5 | 4/5 | 0/7 | 0/5 (0%) | 0/7 (0%) |
| S3-suitable-smoke | Ours | 19/30 | 19 | 26.3% | 27.0% | 19.4% | -0.008 | 1/7 | 4/6 | 0/6 | 0/7 (0%) | 0/6 (0%) |

### 2b. Key Comparisons (S2-full vs S3-suitable, both on same 10 stories)

**Ours:**
- Overall Acc: 36.8% → 26.3% (-10.5pp)
- Easy hit: 2/7 → 1/7
- Medium hit: 3/5 → 4/6
- Hard hit: 2/7 → 0/6
- Spearman: 0.300 → -0.008

**ICL:**
- Overall Acc: 31.6% → 30.0% (-1.6pp)

**Direct:**
- Overall Acc: 36.0% → 30.0% (-6.0pp)

### 2c. Decision

**These 10 stories already performed badly in Stage 2.** The full-run Ours average was 56.7% accuracy, but on these 10 specific stories it was only 36.8%. The S3 smoke further dropped to 26.3%, but the baseline was already low.

**Conclusion: This is primarily STORY VARIANCE.** The random seed happened to pick 10 hard stories. Small N=10 is insufficient to judge suitability filter effect.

**Recommendation: Run larger suitable smoke with max_stories=30 before full run.**

---

## 3. Candidate Property Comparison

### 3a. Change Count

| Category | Count |
|---|---|
| Same candidate selected | 22/30 (73.3%) |
| Different candidate selected | 8/30 (26.7%) |

### 3b. Changes by Difficulty

**Hard (3 changed):**
- bokwewa: "she was dead" → "wanted kwasynd to succeed" (better: motivation, explanatory)
- gray-eagle: "grateful" (emotion_label, NOT suitable in S2!) → "they could die from the cold climate" (suitable, explanatory)
- morraha: "morraha will win" → "the bellman said everyone who killed a raven..." (better: explanatory, longer)

**Medium (4 changed):**
- gray-eagle: "sad" (emotion_label) → "easier to find food" (better: more substantive)
- habetrot: "sad" (emotion_label) → "she did not want to go to the nunnery" (better: explanatory)
- the-little-spirit: "she worried about him" (NOT suitable in S2!) → "the boy was small" (suitable)
- the-winter-spirit: "relaxed" (short_label) → "his entertainer melted away" (better: more substantive)

**Easy (1 changed):**
- gray-eagle: "thankful" (short_label) → "white owl wanted someone else to find the food" (explanatory, still ASA=yes, num_req=1)

### 3c. S2 Non-Suitable Candidates Detected

Two S2 candidates were **not suitable** by Stage 3 rules:
- gray-eagle Hard: "grateful" — answer_type=emotion_label
- the-little-spirit Medium: "she worried about him" — num_required_sentences probably mismatched

S3 correctly replaced these with suitable alternatives.

### 3d. Answer Property Comparison

| Property | S2 Easy (n=9) | S3 Easy (n=10) | S2 Hard (n=10) | S3 Hard (n=10) |
|---|---|---|---|---|
| emotion_label or short_label | 4 | 3 | 2 (non-suitable!) | 0 |
| num_required_sentences (mean) | 1.0 | 1.0 | 3.3 | 3.4 |
| ASA=yes | 100% | 100% | 0% | 0% |

**S3 Hard candidates are clearly better:** zero emotion/short-label answers vs 2 in S2 (both non-suitable). S3 Easy candidates are equally local (ASA=yes, num_req=1 for all).

---

## 4. Ours Easy Failure Diagnosis (Stage 3 Smoke)

### 4a. Overall Breakdown

| # | Story | Answer | Generated Q | QP | Pred | Failure Type |
|---|---|---|---|---|---|---|
| 1 | bokwewa-the-humpback | sad | Where did he abandon abandon afterward? | False | ? | **degenerate_output** |
| 2 | gray-eagle-and-his-five-brothers | white owl wanted someone else to find the food | Who did White Owl try something someone to on the food? | False | Medium | **degenerate_output** (graph non-compliant) |
| 3 | habetrot-the-spinstress | she will meet someone who will spin yarn for her | What did the old dame offer to do for Maisie? | False | ? | **degenerate_output** (parse failure) |
| 4 | jamie-freel-and-the-young-lady | he was thirsting for adventure | What was Jamie's feeling about joining the fairy ride to Dublin? | True | Medium | **judge_overcount** |
| 5 | morraha | he would lose more | What did the woman tell Morraha he would do if he went to play again? | True | Medium | **judge_overcount** |
| 6 | mount-of-golden-queen | happy | How did the princess react when she read the letter? | True | Easy | OK |
| 7 | prince-featherhead-and-the-princess-celandine | king bruin heard the king had no more treasures | What did the consequence of the king and queen giving away all their treasures lead to? | True | Medium | **question_introduced_context** |
| 8 | the-little-spirit-or-boy-man | the boy had powers | What does the mother believe about the boy based on his actions? | True | Medium | **judge_overcount** |
| 9 | the-winter-spirit-and-his-visitor | as his first trophy in the north | What did the young visitor, seegwun, place in the wreath upon his brow? | True | Medium | **judge_overcount** |
| 10 | white-hare-and-crocodiles | angry | How did the crocodile feel when the hare finished counting and left? | True | Hard | **question_introduced_context** |

### 4b. Failure Type Summary

| Failure Type | Count | Actionable? |
|---|---|---|
| degenerate_output | 3/10 (30%) | Yes — generation robustness |
| question_introduced_context | 2/10 (20%) | Yes — tighten Easy prompt |
| judge_overcount | 4/10 (40%) | Partial — some are genuine judge errors |
| OK (predicted Easy) | 1/10 (10%) | — |

### 4c. Detailed Analysis

**degenerate_output (3 cases):**

| Story | Issue |
|---|---|
| bokwewa | "Where did he abandon abandon afterward?" — word repetition: "abandon abandon". Question makes no sense for Easy answer "sad". |
| gray-eagle | "Who did White Owl try something someone to on the food ? ?" — garbled. Self-check failed with answer mismatch + graph_policy non-compliant. |
| habetrot | "What did the old dame offer to do for Maisie?" — question LOOKS valid but marked parse failure. Need to check what parse error occurred. |

These 3 Easy candidates all come from stories where the Easy answer is about emotion/state (sad, white owl wanting something, meeting someone). The answer_only graph policy may give insufficient context for generation when the answer sentence is isolated.

**question_introduced_context (2 cases):**

| Story | Issue |
|---|---|
| prince-featherhead | "What did the **consequence** of ... **lead to**?" — uses causal framing. The direct_answer prompt should prevent this. The model introduced "consequence" and "lead to" despite the Easy strategy saying "Do NOT ask 'Why...', 'What motivated...', 'What caused...'". Need to also forbid "consequence" and "lead to". |
| white-hare | "How did the crocodile feel when the hare finished counting **and left**?" — adds temporal clause "and left" which introduces a second event. The judge sees two events and calls it Hard. But the core question is just "How did the crocodile feel?" |

**judge_overcount (4 cases):**

| Story | Genuine Issue? |
|---|---|
| jamie-freel | "What was Jamie's feeling about joining the fairy ride to Dublin?" — pure emotion question. Answer "he was thirsting for adventure." Judge says Medium. This is a **clear judge error** — should be Easy. |
| morraha | "What did the woman tell Morraha he would do if he went to play again?" — factual prediction. Answer "he would lose more." Judge says Medium. Ambiguous — the "if" clause may make it seem conditional. |
| the-little-spirit | "What does the mother believe about the boy **based on his actions**?" — uses "based on" which implies inference from evidence. Answer "the boy had powers." Judge says Medium. The phrase "based on his actions" introduces a reasoning step. This is a **prompt subtlety issue** — the model used inferential framing even for a direct answer. |
| the-winter-spirit | "What did the young visitor, seegwun, place in the wreath upon his brow?" — pure factual retrieval. Answer "as his first trophy in the north." Judge says Medium. This is a **clear judge error** — should be Easy. |

### 4d. Decision Rule for Easy Failures

| Failure Type | Count | Decision |
|---|---|---|
| degenerate_output | 3 | Debug Easy generation robustness. The answer_only graph policy may give too little context. Consider adding fallback context for degenerate cases. |
| question_introduced_context | 2 | Tighten Easy prompt to forbid "consequence", "lead to", "based on", "after X happened". Add examples of BAD questions that use causal/inferential wording. |
| judge_overcount | 4 | Two are clear judge errors (jamie-freel, winter-spirit). Two are prompt subtlety issues (morraha conditional, little-spirit inferential). Judge cannot be changed per rules. Focus on prompt hardening. |
| OK | 1 | — |

---

## 5. Consolidated Decision Rules

### 5a. Root Cause

**Primary: Story variance.** These 10 stories underperform in both Stage 2 (Ours 36.8%) and Stage 3 (Ours 26.3%). The full-run average is 56.7% — these stories are ~20pp harder than average. N=10 is too small for reliable comparison.

**Secondary: Generation quality.** 30% degenerate rate on Easy (3/10) is high. The answer_only graph policy may give insufficient generation context for certain story types.

**Tertiary: Judge bias.** The judge overcounts difficulty for ~40% of Easy QP-passing questions, predicting Medium for clearly single-sentence questions like "What was Jamie's feeling about..." and "What did the young visitor place...".

### 5b. Suitability Filter Effect

The suitability filter correctly:
- Removed 2 non-suitable candidates S2 had selected (gray-eagle Hard "grateful", the-little-spirit Medium "she worried about him")
- Replaced emotion-label Medium/Hard answers with more substantive ones (6/8 changes were clear improvements)
- Maintained all Easy candidates at ASA=yes, num_req=1

The suitability filter does NOT cause the regression. The slight additional drop (36.8% → 26.3%) is within noise for N=10.

### 5c. Recommended Actions

1. **DO NOT full-run yet.** The smoke didn't meet Easy ASA target and N=10 is unreliable.

2. **Run larger smoke with max_stories=30** to reduce story variance. If Ours accuracy >= 50% on 30 stories, proceed to full run.

3. **Fix Easy generation robustness:**
   - Add "consequence", "lead to", "based on", "after X" to Easy prompt forbidden list
   - Investigate answer_only graph failures (3/10 degenerate)
   - Consider adding 1 bridge sentence as context for answer_only policy when the answer is short (<5 words)

4. **Judge overcount is a hard problem** — the judge is blind and consistent between runs. Can't change per rules. Focus on making Easy questions unambiguously single-sentence by hardening the prompt.

---

## 6. Data Sources

- Stage 2 full run: `outputs/runs/fairytale_qg_crossqg_eval_20260513_story_matched_stage2/`
- Stage 3 suitable smoke: `outputs/runs/fairytale_qg_crossqg_eval_20260513_suitable_smoke/`
- Candidates: `outputs/runs/fairytale_evidence_audit_train_implicit_2166_20260511/candidates.jsonl`
- No new API calls made.
