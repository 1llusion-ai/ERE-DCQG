# Full Hard Path Feasibility Audit

**Date:** 2026-05-09
**Goal:** Determine whether the full validation set contains enough qualified Hard paths, or whether the old `path_filter_strict_pilot` only sampled a biased subset.

---

## 1. Data Scale Comparison

| Metric | Strict Pilot (old) | Full Validation |
|--------|-------------------:|----------------:|
| Documents sampled | 20 | 707 |
| Total paths | 288 | 10,256 |
| Easy paths | 100 | 3,506 |
| Medium paths | 97 | 3,442 |
| Hard paths | 91 | 3,308 |
| Hard prefilter pass | 75 | 2,598 |

**Conclusion:** The strict pilot used only **2.8% of validation documents**. The full set has **36x more Hard paths** and **35x more prefilter-passing Hard paths**.

---

## 2. Hard Path Rule-Based Statistics (Full Set, N=3308)

### 2.1 single_sentence_risk (rule-based)

| Risk | Count | Rate |
|------|------:|-----:|
| low | 3,257 | 98.5% |
| medium | 29 | 0.9% |
| high | 22 | 0.7% |

**The rule-based prefilter says 98.5% of Hard paths have LOW single-sentence risk.** This means path events span multiple sentences and the answer is unlikely to be found in a single sentence.

### 2.2 answer_phrase_pass

| Pass | Count | Rate |
|------|------:|-----:|
| True | 2,728 | 82.5% |
| False | 580 | 17.5% |

### 2.3 relation_group

| Group | Count | Rate |
|-------|------:|-----:|
| TEMPORAL | 1,894 | 57.3% |
| CAUSE | 1,363 | 41.2% |
| SUBEVENT | 26 | 0.8% |
| MIXED | 25 | 0.8% |

### 2.4 prefilter_pass

| Pass | Count | Rate |
|------|------:|-----:|
| True | 2,598 | 78.5% |
| False | 710 | 21.5% |

---

## 3. LLM Judge Results (Sampled 142 Hard Paths)

Stratified sample of 142 prefilter-passing Hard paths, judged by gpt-4o-mini.

### 3.1 single_sentence_risk (LLM)

| Risk | Count | Rate |
|------|------:|-----:|
| low | 1 | 0.7% |
| medium | 0 | 0.0% |
| high | 141 | 99.3% |

**CRITICAL: The LLM judge marks 99.3% of Hard paths as `single_sentence_risk: high`.**

### 3.2 can_write_path_dependent_question

| Value | Count | Rate |
|-------|------:|-----:|
| yes | 55 | 38.7% |
| partial | 67 | 47.2% |
| no | 20 | 14.1% |

### 3.3 path_questionable

| Value | Count | Rate |
|-------|------:|-----:|
| yes | 55 | 38.7% |
| partial | 67 | 47.2% |
| no | 20 | 14.1% |

### 3.4 recommended_difficulty

| Value | Count | Rate |
|-------|------:|-----:|
| hard | 99 | 69.7% |
| medium | 23 | 16.2% |
| easy | 20 | 14.1% |

### 3.5 expected_required_steps

| Steps | Count | Rate |
|-------|------:|-----:|
| 3+ | 99 | 69.7% |
| 2 | 23 | 16.2% |
| 1 | 20 | 14.1% |

---

## 4. Cross-Tabulation: rule_ssr x LLM_ssr

| rule_ssr | llm_ssr=high | llm_ssr=low | llm_ssr=medium |
|----------|-------------:|------------:|---------------:|
| low | 98 | 1 | 0 |
| medium | 23 | 0 | 0 |
| high | 20 | 0 | 0 |

**The LLM judge and rule-based prefilter disagree fundamentally.** 98 paths the rules call "low risk" are judged "high risk" by the LLM. The LLM appears to interpret `single_sentence_risk` differently -- likely asking "can the FINAL ANSWER be found in one sentence?" rather than "do path events span multiple sentences?"

---

## 5. Policy Bug: path_questionable Inversion

### Old policy condition:
```
can_write_path_dependent_question == "yes" AND path_questionable != "yes"
```

### Problem:
The judge prompt defines `path_questionable="yes"` as **"the path CAN support a good question"** (positive signal). The policy treats `path_questionable="yes"` as a REJECTION criterion. Since `path_questionable` and `can_write_path_dependent_question` are perfectly correlated for Hard paths (both `yes`=55, `partial`=67, `no`=20), the old policy filters out ALL `cwd=yes` paths.

### Result: 0 qualified Hard paths under old policy (due to bug).

### Fixed policy condition:
```
can_write_path_dependent_question == "yes" AND path_questionable != "no"
```

### Result with fix: 55/142 (38.7%) qualified, extrapolated to ~1,006 in full set.

---

## 6. New Policy: Hard single_sentence_risk=high Disqualification

### User's requested policy:
- Hard paths with `single_sentence_risk=high` cannot be kept as qualified Hard.
- Can be kept as diagnostic or downgraded to Medium/Easy.

### Impact:

| Policy | Sample (N=142) | Extrapolated (N=2598) |
|--------|---------------:|----------------------:|
| Old (buggy) | 0 | 0 |
| Fixed (pq!=no) | 55 (38.7%) | ~1,006 |
| Fixed + ssr!=high | 1 (0.7%) | ~18 |
| Fixed + cwd=yes/partial + ssr!=high | 1 (0.7%) | ~18 |

**The `ssr!=high` constraint eliminates 99.3% of Hard paths** because the LLM judge marks nearly all Hard paths as `single_sentence_risk: high`.

---

## 7. The Single LLM-Approved Low-Risk Hard Path

One path was judged `single_sentence_risk=low` by the LLM:

- **doc:** `e0fd1849f976211f...`
- **Events:** developed(S1) -> intensified(S3) -> peaked(S4) -> re-curved(S5)
- **Relation group:** CAUSE
- **Support span:** 7 sentences
- **LLM verdict:** cwd=yes, pq=yes, rec_difficulty=hard, steps=3+
- **LLM reason:** "The path supports a clear question about Rachel's re-curving event, requiring knowledge of its development and intensification."

---

## 8. Root Cause Analysis

### Why the LLM judge flags almost all Hard paths as single_sentence_risk=high:

The LLM judge likely interprets `single_sentence_risk` as:
> "Can the FINAL ANSWER (the final event's trigger/phrase) be found in a single sentence?"

This is almost always true for MAVEN-ERE: each event has a trigger in a specific sentence, and that sentence often contains enough context to identify the answer.

The rule-based prefilter interprets it as:
> "Do the path events span multiple sentences?"

These are fundamentally different questions. A 3-hop path can span 7 sentences (low rule risk) while the final answer is still findable from one sentence (high LLM risk).

### Why 55/142 still have cwd=yes despite high ssr:

The LLM judge separately considers whether a PATH-DEPENDENT QUESTION can be written. Even when the answer is in one sentence, the question can require knowing about prior events. The judge says 38.7% of Hard paths support this.

---

## 9. Conclusions

### What the data says:

1. **The full validation set has a massive Hard path pool** (3,308 raw, 2,598 prefilter-passing). The old pilot's 91 Hard paths from 20 docs was a tiny, biased sample.

2. **The old policy has a bug** (`path_questionable != "yes"` should be `!= "no"`). With the fix, ~1,006 Hard paths qualify.

3. **The LLM judge and rule-based prefilter disagree on `single_sentence_risk`.** The LLM says 99.3% are high risk; rules say 98.5% are low risk.

4. **Applying `ssr!=high` from the LLM judge eliminates virtually all Hard paths** (only 1/142 survives, extrapolated to ~18).

5. **Despite high ssr, 38.7% of Hard paths CAN support path-dependent questions** according to the LLM judge. The `cwd=yes` and `ssr=high` are not mutually exclusive.

### Recommendations:

1. **Fix the policy bug immediately.** Change `path_questionable != "yes"` to `path_questionable != "no"`.

2. **Reconsider the `ssr!=high` constraint.** The LLM's `single_sentence_risk` measures something different from what we want. Options:
   - Use the rule-based `rule_single_sentence_risk` instead (98.5% low).
   - Redefine the LLM judge prompt to ask about path-level multi-sentence dependency, not answer-level findability.
   - Accept `ssr=high` but require `cwd=yes` as the actual Hard quality gate.

3. **For the Hard rescue pool**, use the fixed policy (cwd=yes + pq!=no) which yields ~1,006 paths. This is a viable pool for multi-candidate generation.

4. **If `ssr!=high` is non-negotiable**, we need to either:
   - Fix the LLM judge prompt to distinguish "answer in one sentence" from "path spans one sentence".
   - Or accept that only ~18 Hard paths survive, which is insufficient.

---

## 10. Output Files

| File | Content |
|------|---------|
| `paths.raw.jsonl` | 10,256 raw sampled paths (all 707 docs) |
| `paths.prefiltered.jsonl` | All paths with prefilter results |
| `paths.prefiltered.hard.jsonl` | 3,308 Hard paths with prefilter |
| `hard_prefilter_pass.jsonl` | 2,598 prefilter-passing Hard paths |
| `hard_sample_for_judge.jsonl` | 144 Hard paths sampled for LLM judge |
| `hard_sample_judged.jsonl` | 142 Hard paths with LLM judge results |
| `HARD_PATH_FEASIBILITY_AUDIT.md` | This report |
