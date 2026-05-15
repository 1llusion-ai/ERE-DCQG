# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-13 21:08:44

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Selection mode | story_matched_suitable |
| Target per level | 150 |
| Selected Easy | 30 |
| Selected Medium | 30 |
| Selected Hard | 30 |
| Total selected | 90 |
| Total stories | 30 |
| Total generations | 360 |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 28 | 30 | 93.3% |
| Medium | 29 | 30 | 96.7% |
| Hard | 27 | 30 | 90.0% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 87 | 90 | 96.7% |
| ICL | 82 | 90 | 91.1% |
| SelfRefine | 61 | 90 | 67.8% |
| Ours | 77 | 90 | 85.6% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 21 | 19 | 22 | 62 | 90 | 68.9% |
| ICL | 15 | 22 | 19 | 56 | 90 | 62.2% |
| SelfRefine | 11 | 14 | 14 | 39 | 90 | 43.3% |
| Ours | 21 | 20 | 19 | 60 | 90 | 66.7% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 87 | 3 | 90 | 3.3% |
| ICL | 81 | 9 | 90 | 10.0% |
| SelfRefine | 61 | 29 | 90 | 32.2% |
| Ours | 77 | 13 | 90 | 14.4% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 29.0% | 18/62 (29.0%, 95%CI [19.2, 41.3%]) | 30.4% |
| ICL | 41.1% | 23/56 (41.1%, 95%CI [29.2, 54.1%]) | 37.7% |
| SelfRefine | 43.6% | 17/39 (43.6%, 95%CI [29.3, 59.0%]) | 43.1% |
| Ours | 43.3% | 26/60 (43.3%, 95%CI [31.6, 55.9%]) | 43.0% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 15 | 2 |
| Medium | 6 | 12 | 1 |
| Hard | 3 | 17 | 2 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 3 | 12 | 0 |
| Medium | 5 | 17 | 0 |
| Hard | 1 | 15 | 3 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 6 | 1 |
| Medium | 3 | 10 | 1 |
| Hard | 2 | 9 | 3 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 7 | 13 | 1 |
| Medium | 4 | 16 | 0 |
| Hard | 1 | 15 | 3 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 4/21 (19.0%, 95%CI [7.7, 40.0%]) | 12/19 (63.2%, 95%CI [41.0, 80.9%]) | 2/22 (9.1%, 95%CI [2.5, 27.8%]) |
| ICL | 3/15 (20.0%, 95%CI [7.0, 45.2%]) | 17/22 (77.3%, 95%CI [56.6, 89.9%]) | 3/19 (15.8%, 95%CI [5.5, 37.6%]) |
| SelfRefine | 4/11 (36.4%, 95%CI [15.2, 64.6%]) | 10/14 (71.4%, 95%CI [45.4, 88.3%]) | 3/14 (21.4%, 95%CI [7.6, 47.6%]) |
| Ours | 7/21 (33.3%, 95%CI [17.2, 54.6%]) | 16/20 (80.0%, 95%CI [58.4, 91.9%]) | 3/19 (15.8%, 95%CI [5.5, 37.6%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 25.5% | 23.5% | 38.1% | 14.8% |
| ICL | 34.6% | 25.0% | 51.5% | 27.3% |
| SelfRefine | 41.0% | 40.0% | 51.3% | 31.6% |
| Ours | 39.5% | 42.4% | 50.0% | 26.1% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.045 | 62 |
| ICL | 0.276 | 56 |
| SelfRefine | 0.236 | 39 |
| Ours | 0.316 | 60 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 21 | 4 | 19.0% | 4 | 15 | 2 |
| ICL | 15 | 3 | 20.0% | 3 | 12 | 0 |
| SelfRefine | 11 | 4 | 36.4% | 4 | 6 | 1 |
| Ours | 21 | 7 | 33.3% | 7 | 13 | 1 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 19 | 12 | 63.2% | 6 | 12 | 1 |
| ICL | 22 | 17 | 77.3% | 5 | 17 | 0 |
| SelfRefine | 14 | 10 | 71.4% | 3 | 10 | 1 |
| Ours | 20 | 16 | 80.0% | 4 | 16 | 0 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 22 | 2 | 9.1% | 3 | 17 | 2 |
| ICL | 19 | 3 | 15.8% | 1 | 15 | 3 |
| SelfRefine | 14 | 3 | 21.4% | 2 | 9 | 3 |
| Ours | 19 | 3 | 15.8% | 1 | 15 | 3 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 2/22 (9.1%, 95%CI [2.5, 27.8%]) | |
| ICL | 3/19 (15.8%, 95%CI [5.5, 37.6%]) | |
| SelfRefine | 3/14 (21.4%, 95%CI [7.6, 47.6%]) | |
| Ours | 3/19 (15.8%, 95%CI [5.5, 37.6%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 2/22 (9.1%, 95%CI [2.5, 27.8%]) | |
| ICL | 3/19 (15.8%, 95%CI [5.5, 37.6%]) | |
| SelfRefine | 3/14 (21.4%, 95%CI [7.6, 47.6%]) | |
| Ours | 3/19 (15.8%, 95%CI [5.5, 37.6%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/22 (0.0%, 95%CI [0.0, 14.9%]) | |
| ICL | 0/19 (0.0%, 95%CI [0.0, 16.8%]) | |
| SelfRefine | 0/14 (0.0%, 95%CI [0.0, 21.5%]) | |
| Ours | 0/19 (0.0%, 95%CI [0.0, 16.8%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 2 | 2 |
| ICL | 3 | 3 |
| SelfRefine | 3 | 3 |
| Ours | 3 | 3 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 28 | 0 | 0 | 0 | 28 |
| Medium | 0 | 29 | 0 | 0 | 29 |
| Hard | 0 | 0 | 27 | 0 | 27 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 17 | 28 | 60.7% |
| Medium | 26 | 29 | 89.7% |
| Hard | 21 | 27 | 77.8% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 21 | 33.3% | N/A |
| two_node_relation | 20 | 80.0% | N/A |
| multi_node_chain | 19 | 15.8% | 15.8% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 28 | 33.3% |
| causes | 10 | 11.9% |
| motivates | 9 | 10.7% |
| causes → results_in | 8 | 9.5% |
| explains | 5 | 6.0% |
| results_in | 2 | 2.4% |
| enables | 2 | 2.4% |
| temporal_before → causes → results_in | 2 | 2.4% |
| supports_inference → supports_inference → supports_inference → results_in | 1 | 1.2% |
| motivates → causes | 1 | 1.2% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| results_in | 1 | 10 | 1 | 12 |
| causes | 1 | 7 | 3 | 11 |
| motivates | 0 | 6 | 0 | 6 |
| supports_inference | 0 | 5 | 0 | 5 |
| temporal_before | 0 | 4 | 0 | 4 |
| enables | 0 | 0 | 2 | 2 |
| explains | 0 | 2 | 0 | 2 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 12 | 28 | 42.9% |
| Medium | two_node_relation | 13 | 29 | 44.8% |
| Hard | multi_node_chain | 12 | 27 | 44.4% |

### GP-7. Hard Pure Temporal Chain Count: 0

### GP-8. Easy Answer-Sentence-Alone Rate: 33.3% (7/21)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 52.6% (10/19)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 66.7% (60/90) | 68.9% (62/90) | 62.2% (56/90) | 43.3% (39/90) | -2.2pp | +4.4pp | +23.3pp |
| overall_accuracy | 43.3% | 29.0% | 41.1% | 43.6% | +14.3pp | +2.3pp | -0.3pp |
| macro_accuracy | 43.0% | 30.4% | 37.7% | 43.1% | +12.6pp | +5.4pp | -0.0pp |
| macro_f1 | 39.5% | 25.5% | 34.6% | 41.0% | +14.0pp | +4.9pp | -1.4pp |
| spearman | 0.316 | 0.045 | 0.276 | 0.236 | +0.271 | +0.040 | +0.080 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected >= 150 per level | FAIL | Easy=30, Medium=30, Hard=30 |
| Ours quality_pass not << baselines | PASS | Ours=60/90 |
| Ours overall accuracy >= each baseline | FAIL | Ours=43.3% |
| Ours macro accuracy >= each baseline | FAIL | Ours=43.0% |
| Ours Spearman >= each baseline | PASS | Ours=0.316 (min margin=+0.040 vs SelfRefine) |

**Overall: SOME CRITERIA FAILED**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 35.6% | 33.3% | +2.22pp | [-11.11pp, +15.56pp] | 0.8944 | 45 | no |
| Direct | macro_accuracy | 38.5% | 35.9% | +2.56pp | [-10.93pp, +15.77pp] | 0.7422 | 45 | no |
| Direct | macro_f1 | 32.4% | 31.0% | +1.46pp | [-14.11pp, +16.77pp] | 0.8836 | 45 | no |
| Direct | spearman | 0.237 | 0.222 | +0.015 | [-0.258, +0.281] | 0.9210 | 45 | no |
| ICL | overall_accuracy | 41.9% | 39.5% | +2.33pp | [-6.98pp, +13.95pp] | 0.8274 | 43 | no |
| ICL | macro_accuracy | 39.9% | 37.8% | +2.08pp | [-8.50pp, +13.45pp] | 0.7614 | 43 | no |
| ICL | macro_f1 | 35.6% | 34.2% | +1.45pp | [-12.39pp, +15.71pp] | 0.8450 | 43 | no |
| ICL | spearman | 0.210 | 0.279 | -0.069 | [-0.448, +0.281] | 0.7416 | 43 | no |
| SelfRefine | overall_accuracy | 41.4% | 48.3% | -6.90pp | [-24.14pp, +10.34pp] | 0.5284 | 29 | no |
| SelfRefine | macro_accuracy | 41.1% | 49.4% | -8.33pp | [-25.93pp, +7.54pp] | 0.3368 | 29 | no |
| SelfRefine | macro_f1 | 37.2% | 46.3% | -9.17pp | [-28.77pp, +8.56pp] | 0.3326 | 29 | no |
| SelfRefine | spearman | 0.293 | 0.411 | -0.117 | [-0.516, +0.258] | 0.5602 | 29 | no |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 13.3% (4/30) | 40.0% (12/30) | 6.7% (2/30) | 20.0% (18/90) | 90 |
| ICL | 10.0% (3/30) | 56.7% (17/30) | 10.0% (3/30) | 25.6% (23/90) | 90 |
| SelfRefine | 13.3% (4/30) | 33.3% (10/30) | 10.0% (3/30) | 18.9% (17/90) | 90 |
| Ours | 23.3% (7/30) | 53.3% (16/30) | 10.0% (3/30) | 28.9% (26/90) | 90 |

## 11. Difficulty Judge Prompt Audit

The difficulty judge is **blind**: it sees only the story, generated question, and expected answer.
It does NOT see the target difficulty. Below are 3 sample prompts to confirm.

**Sample 1** (story=Snow-man, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " this is really beautiful , " said a young girl , who had come into the garden with a young man .
[S1] they both stood still near the snow man , and contemplated the glittering scene .
[S2] " summer can not show a more beautiful sight , " she exclaimed , while her eyes sparkled .
[S3] " and we ca n't have such a fellow as this in the summer time , " replied the young man , pointing to the snow man ; " he is capital .
[S4] " the girl laughed , and nodded at the snow man , and then tripped away over the snow with her friend .
[S5] the snow creaked and crackled beneath her feet , as if she had been treading on starch .
[S6] " who are these two ?
[S7] " asked the snow man of the yard - dog .
[S8] " you
... [truncated]
```

**Sample 2** (story=Snow-man, target=Medium):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " this is really beautiful , " said a young girl , who had come into the garden with a young man .
[S1] they both stood still near the snow man , and contemplated the glittering scene .
[S2] " summer can not show a more beautiful sight , " she exclaimed , while her eyes sparkled .
[S3] " and we ca n't have such a fellow as this in the summer time , " replied the young man , pointing to the snow man ; " he is capital .
[S4] " the girl laughed , and nodded at the snow man , and then tripped away over the snow with her friend .
[S5] the snow creaked and crackled beneath her feet , as if she had been treading on starch .
[S6] " who are these two ?
[S7] " asked the snow man of the yard - dog .
[S8] " you
... [truncated]
```

**Sample 3** (story=Snow-man, target=Medium):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " this is really beautiful , " said a young girl , who had come into the garden with a young man .
[S1] they both stood still near the snow man , and contemplated the glittering scene .
[S2] " summer can not show a more beautiful sight , " she exclaimed , while her eyes sparkled .
[S3] " and we ca n't have such a fellow as this in the summer time , " replied the young man , pointing to the snow man ; " he is capital .
[S4] " the girl laughed , and nodded at the snow man , and then tripped away over the snow with her friend .
[S5] the snow creaked and crackled beneath her feet , as if she had been treading on starch .
[S6] " who are these two ?
[S7] " asked the snow man of the yard - dog .
[S8] " you
... [truncated]
```

**Confirmed:** No target difficulty appears in any judge prompt. Blind evaluation is correct.

## 12. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 8 |
| Direct | gen error: no question mark | 5 |
| Direct | other | 5 |
| Direct | not fluent | 4 |
| Direct | gen error: degenerate output | 3 |
| Direct | not answerable | 2 |
| Direct | gen error: empty | 1 |
| ICL | wrong answer | 12 |
| ICL | gen error: degenerate output | 10 |
| ICL | not answerable | 5 |
| ICL | other | 3 |
| ICL | not fluent | 2 |
| ICL | gen error: grammar: word repetition: care | 1 |
| ICL | gen error: empty | 1 |
| SelfRefine | gen error: degenerate initial output | 26 |
| SelfRefine | not answerable | 7 |
| SelfRefine | gen error: no question mark | 5 |
| SelfRefine | wrong answer | 5 |
| SelfRefine | gen error: initial generation failed | 3 |
| SelfRefine | other | 2 |
| SelfRefine | not fluent | 2 |
| SelfRefine | gen error: grammar: word repetition: did | 1 |
| Ours | gen error: degenerate output | 9 |
| Ours | gen error: graph_invalid | 6 |
| Ours | gen error: no question mark | 4 |
| Ours | gen error: self-check failed: answer mismatch | 3 |
| Ours | gen error: question length out of range | 3 |
| Ours | gen error: parse failure | 3 |
| Ours | gen error: self-check failed: answer mismatch; graph_policy non-compliant (answer_only) | 2 |

## 14. Retry & Budget Diagnostics

### 14a. Attempts per Method

| Method | Avg attempts | Max attempts | Total |
|---|---:|---:|---:|
| Direct | 1.78 | 3 | 90 |
| ICL | 1.69 | 3 | 90 |
| SelfRefine | 2.36 | 3 | 90 |
| Ours | 2.70 | 4 | 90 |

### 14b. Attempt Distribution by Method

| Method | 1 attempt | 2 attempts | 3+ attempts |
|---|---:|---:|---:|
| Direct | 41 | 28 | 21 |
| ICL | 49 | 20 | 21 |
| SelfRefine | 29 | 0 | 61 |
| Ours | 26 | 11 | 53 |

### 14c. Ours Repair Prompt Usage

| Metric | Value |
|---|---|
| Repair prompt used | 37/90 (41.1%) |
| Repair success | 10 |

### 14d. Ours Graph Policy Self-Check Failure Rate

| Self-check failures | 20/84 (23.8%) |

### 14e. Failure Reason Distribution by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 8 |
| Direct | not fluent | 6 |
| Direct | not answerable | 5 |
| Direct | unknown | 5 |
| Direct | degenerate output | 3 |
| Direct | empty output | 1 |
| ICL | wrong answer | 12 |
| ICL | degenerate output | 10 |
| ICL | not answerable | 6 |
| ICL | unknown | 3 |
| ICL | not fluent | 2 |
| ICL | empty output | 1 |
| SelfRefine | degenerate output | 26 |
| SelfRefine | not answerable | 9 |
| SelfRefine | not fluent | 6 |
| SelfRefine | wrong answer | 5 |
| SelfRefine | initial generation failed | 3 |
| SelfRefine | None | 2 |
| Ours | degenerate output | 9 |
| Ours | graph issue | 8 |
| Ours | self-check failed | 3 |
| Ours | not fluent | 3 |
| Ours | question length | 3 |
| Ours | parse failure | 3 |
| Ours | not answerable | 1 |

### 14f. Ours Retry Reason Distribution (from attempt traces)

| Retry reason | Count |
|---|---:|
| ok | 110 |
| degenerate output | 81 |
| no question mark | 17 |
| parse failure | 14 |
| question length out of range | 12 |
| empty | 4 |
| grammar: word repetition: what | 1 |
| grammar: word repetition: old | 1 |
| does not end with ? | 1 |
| grammar: word repetition: felt | 1 |

## 15. Similarity Diagnostics (diagnostic only, no filtering)

### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)

| Method | E-M mean | M-H mean | E-H mean | n pairs |
|---|---:|---:|---:|---:|
| Direct | 0.079 | 0.075 | 0.084 | 28 |
| ICL | 0.087 | 0.097 | 0.090 | 22 |
| SelfRefine | 0.099 | 0.097 | 0.078 | 14 |
| Ours | 0.060 | 0.071 | 0.059 | 27 |

### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)

| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |
|---|---:|---:|---:|
| Direct | 0.111 | 0.057 | 0.075 |
| ICL | 0.072 | 0.103 | 0.042 |
| SelfRefine | 0.078 | 0.026 | 0.044 |
| Ours | 0.117 | 0.042 | 0.067 |

### 15c. Difficulty Collapse Counts by Method

| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |
|---|---:|---:|---:|---:|---:|
| Direct | 14 | 4 | 4 | 0 | 0 |
| ICL | 7 | 5 | 4 | 1 | 0 |
| SelfRefine | 2 | 0 | 0 | 0 | 0 |
| Ours | 9 | 3 | 3 | 0 | 0 |

## 16. Stage 2 Focus & Difficulty Realization Diagnostics (Ours)

### 16a. Focus Distribution by Target Difficulty (Ours, quality-pass)

| Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| chain_explanation | 0 | 0 | 19 |
| direct_answer | 21 | 0 | 0 |
| relation_question | 0 | 20 | 0 |

### 16b. Node-Level Focus Distribution (pre-override, for comparison)

| Node Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| bridge | 13 | 9 | 8 |
| motivation | 6 | 4 | 5 |
| outcome | 0 | 3 | 4 |
| state | 2 | 4 | 2 |

### 16c. Easy answer_sentence_alone=yes by Focus Type (Ours, quality-pass, target=Easy)

| Focus | Total | ASA=yes | Rate |
|---|---:|---:|---:|
| direct_answer | 21 | 7 | 33.3% |

### 16d. Hard answer_sentence_alone=no by Focus Type (Ours, quality-pass, target=Hard)

| Focus | Total | ASA=no | Rate |
|---|---:|---:|---:|
| chain_explanation | 19 | 10 | 52.6% |

### 16e. Graph Policy Compliance by Focus Type (Ours, quality-pass)

| Focus | Total | GPC=yes | Rate |
|---|---:|---:|---:|
| chain_explanation | 19 | 17 | 89.5% |
| direct_answer | 21 | 17 | 81.0% |
| relation_question | 20 | 19 | 95.0% |

### 16f. Repair Usage by Target Difficulty (Ours)

| Difficulty | Total | Repair Used | Repair Success | Repair Rate |
|---|---:|---:|---:|---:|
| Easy | 30 | 12 | 4 | 40.0% |
| Medium | 30 | 13 | 5 | 43.3% |
| Hard | 30 | 12 | 1 | 40.0% |

### 16g. Top 10 Easy Failures (Ours, quality-pass, predicted != Easy)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | habetrot-the-spinstress | What will Maisie do next according to the old woman's offer? | she will meet someone who will | Medium | direct_answer | partial | bridge_detected |
| 2 | kings-hares | What kind of person would not be suitable to herd hares acco | peter was a sleepy - head . | Medium | direct_answer | partial | bridge_detected |
| 3 | master-girl | What did the giant say about the tasks? | the giant gave seemingly simpl | Medium | direct_answer | partial | bridge_detected |
| 4 | momotaro-story-of-son-of-peach | What did the dog not know about momotaro? | he did not realize the man was | Medium | direct_answer | partial | bridge_detected |
| 5 | mother-hulda | What did she dip the spindle into the well for? | to wash it . | Medium | direct_answer | partial | bridge_detected |
| 6 | the-winter-spirit-and-his-visi | What was the small white flower with a pink border for in th | as his first trophy in the nor | Medium | direct_answer | partial | bridge_detected |
| 7 | bokwewa-the-humpback | How did Bokwewa feel about his brother yielding to temptatio | sad . | Medium | direct_answer | partial | focus_mismatch_causal_wording |
| 8 | brave-tin-soldier | What caused the soldier to stand on one leg? | they ran out of melted tin . | Medium | direct_answer | partial | focus_mismatch_causal_wording |
| 9 | jamie-freel-and-the-young-lady | How did Jamie feel about joining the ride to Dublin? | he was thirsting for adventure | Medium | direct_answer | no | focus_mismatch_causal_wording |
| 10 | lame-dog | Why did the princess often think of visiting her sisters and | she was bored . | Medium | direct_answer | partial | focus_mismatch_causal_wording |

### 16h. Top 10 Hard Failures (Ours, quality-pass, predicted != Hard)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | morraha | What motivated the man to hide in the top of the parlour chi | the bellman said everyone who  | Easy | chain_explanation | yes | answer_sentence_alone_yes |
| 2 | bokwewa-the-humpback | What motivated Bokwewa to give such detailed advice to Kwasy | wanted kwasynd to succeed . | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 3 | kings-hares | What motivated the chamber-maid to offer to pay a hundred do | she was impressed by it . | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 4 | lame-dog | What motivated the youngest princess to express her willingn | she did not care who she marri | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 5 | master-girl | What motivated the king's son to pretend to be very stupid a | he did not want the giant to k | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 6 | momotaro-story-of-son-of-peach | What motivated Momotaro to express such deep gratitude to hi | his parents take good care of  | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 7 | mother-hulda | What led mother hulda to give the lazy girl a warning and po | because she did n't help . | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 8 | prince-featherhead-and-the-pri | What motivated the old fairy to consider Celandine more favo | she did n't think poorly of ce | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 9 | the-believing-husbands | What motivated the wife to urge the husband to hurry and get | she wanted to rush him to go t | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 10 | the-black-bull-of-norroway | What motivated the princess to ultimately agree to ride the  | she did not want to accept her | Medium | chain_explanation | partial | focus_mismatch_direct_wording |

### 16i. Stage 3.1 Easy Prompt Hardening Diagnostics (Ours)

**Easy forbidden-frame violations:** 11/30 (36.7%)

| Story | Question | Violated Frames | QP | Pred |
|---|---|---|---|---|
| brave-tin-soldier | What caused the soldier to stand on one leg? | what caused | Y | Medium |
| jamie-freel-and-the-young | How did Jamie feel about joining the ride to Dublin? | about joining | Y | Medium |
| lame-dog | Why did the princess often think of visiting her sisters and | why | Y | Medium |
| mount-of-golden-queen | How did she feel when she read it? | when (clause) | Y | Easy |
| prince-featherhead-and-th | How bruin how did he feel when when he the king had had no m | when (clause) | N | Medium |
| the-believing-husbands | How did the girl feel when she saw the pack-saddle over her  | when (clause) | Y | Easy |
| the-brown-bear-of-norway | What happened the same evening after she agreed to marry the | after (clause) | Y | Easy |
| the-rich-brother-and-the- | Where did the greedy wife want to go after finishing her sho | after (clause) | N | Medium |
| the-three-crowns | Where will the youngest prince go after the others are taken | after (clause) | Y | Hard |
| the-winning-of-olwen | What was kilweh looking for when he cried? | when (clause) | Y | Easy |

**Easy degenerate output:** 5/30 (16.7%)

| Story | Error |
|---|---|
| habetrot-the-spinstress | degenerate output |
| the-black-bull-of-norrowa | graph_invalid |
| the-little-spirit-or-boy- | degenerate output |
| the-two-jeebi | graph_invalid |
| three-dogs | degenerate output |

**Easy judge-overcount examples (QP-pass, predicted Medium/Hard, clear single-sentence question):**

| Story | Question | Answer | Pred | Why Likely Overcount |
|---|---|---|---|---|
| giauna-the-beautiful | What did the youth's father tell kung to do? | told kung to be strict with hi | Medium | Clear single-sentence question, judged Medium |
| habetrot-the-spinstress | What will Maisie do next according to the old woman's offer? | she will meet someone who will | Medium | Clear single-sentence question, judged Medium |
| kings-hares | What kind of person would not be suitable to herd hares acco | peter was a sleepy - head . | Medium | Clear single-sentence question, judged Medium |
| master-girl | What did the giant say about the tasks? | the giant gave seemingly simpl | Medium | Clear single-sentence question, judged Medium |
| momotaro-story-of-son-of- | What did the dog not know about momotaro? | he did not realize the man was | Medium | Clear single-sentence question, judged Medium |
| mother-hulda | What did she dip the spindle into the well for? | to wash it . | Medium | Clear single-sentence question, judged Medium |
| the-escape-of-the-mouse | What did the townsfolk do with the saddles?  | the townsfolk only bought from | Medium | Clear single-sentence question, judged Medium |
| the-winter-spirit-and-his | What was the small white flower with a pink border for in th | as his first trophy in the nor | Medium | Clear single-sentence question, judged Medium |

**Easy question-introduced-context examples (QP-pass, wording added context):**

| Story | Question | Answer | Pred | Context Wording |
|---|---|---|---|---|
| brave-tin-soldier | What caused the soldier to stand on one leg? | they ran out of melted tin . | Medium | what caused |
| jamie-freel-and-the-young | How did Jamie feel about joining the ride to Dublin? | he was thirsting for adventure | Medium | about joining |
| lame-dog | Why did the princess often think of visiting her sisters and | she was bored . | Medium | why |
| the-three-crowns | Where will the youngest prince go after the others are taken | they will go into the well . | Hard | after (clause) |

## 17. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: mount-of-golden-queen
- Question: How did the princess feel when she read the letter?
- Target answer: happy .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: the-believing-husbands
- Question: How did the girl feel when she saw the pack-saddle over her head?
- Target answer: she was scared after the pack - saddle almost fell on her .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: lame-dog
- Question: Why did the princess often think of visiting her sisters and seeing how they were?
- Target answer: she was bored .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: Snow-man
- Question: Who had come into the garden?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: Snow-man
- Question: Who knew the snow man had been in the garden longer than himself?
- Target answer: the yard - dog had been there longer than the snow man .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: Snow-man
- Question: Who knew more about the garden visitors since they had been there longer?
- Target answer: the yard - dog had been there longer than the snow man .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: brave-tin-soldier
- Question: Who caught and sold the fish that swallowed the tin soldier?
- Target answer: the fish that swallowed him was caught and sold .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: Snow-man
- Question: What explains why the yard-dog knew the young couple better than the snow man?
- Target answer: the yard - dog had been there longer than the snow man .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**Direct Example:**
- Story: the-believing-husbands
- Question: Why did the wife wait until she heard the funeral passing the window before telling the man to rise?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

**ICL Example:**
- Story: soria-moria-castle
- Question: Why did the youngest princess stroke Halvor's hair and then wish they were in Soria-Moria castle?
- Target answer: tricked halvor into falling asleep .
- Target: Hard, Predicted: Hard

**SelfRefine Example:**
- Story: the-believing-husbands
- Question: Why did the wife wait until she heard the funeral passing the window before telling the man to get up?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

**Ours Example:**
- Story: Snow-man
- Question: What led the yard-dog to ultimately be fastened with a chain and lose his bone?
- Target answer: he had to .
- Target: Hard, Predicted: Hard

