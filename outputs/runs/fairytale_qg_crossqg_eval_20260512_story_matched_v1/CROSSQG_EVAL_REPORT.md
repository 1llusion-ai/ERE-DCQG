# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-12 23:42:21

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Selection mode | story_matched |
| Target per level | 150 |
| Selected Easy | 106 |
| Selected Medium | 106 |
| Selected Hard | 106 |
| Total selected | 318 |
| Total stories | 106 |
| Total generations | 1272 |
| Candidates per level per story | 1 |
| Max stories | None |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 94 | 106 | 88.7% |
| Medium | 101 | 106 | 95.3% |
| Hard | 97 | 106 | 91.5% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 291 | 318 | 91.5% |
| ICL | 285 | 318 | 89.6% |
| SelfRefine | 200 | 318 | 62.9% |
| Ours | 268 | 318 | 84.3% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 58 | 77 | 51 | 186 | 318 | 58.5% |
| ICL | 71 | 75 | 54 | 200 | 318 | 62.9% |
| SelfRefine | 49 | 38 | 37 | 124 | 318 | 39.0% |
| Ours | 64 | 76 | 63 | 203 | 318 | 63.8% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 287 | 31 | 318 | 9.7% |
| ICL | 280 | 38 | 318 | 11.9% |
| SelfRefine | 200 | 118 | 318 | 37.1% |
| Ours | 265 | 53 | 318 | 16.7% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 52.7% | 98/186 (52.7%, 95%CI [45.5, 59.7%]) | 48.7% |
| ICL | 52.0% | 104/200 (52.0%, 95%CI [45.1, 58.8%]) | 48.7% |
| SelfRefine | 52.4% | 65/124 (52.4%, 95%CI [43.7, 61.0%]) | 52.9% |
| Ours | 54.2% | 110/203 (54.2%, 95%CI [47.3, 60.9%]) | 52.4% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 32 | 25 | 1 |
| Medium | 18 | 58 | 1 |
| Hard | 5 | 38 | 8 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 35 | 34 | 2 |
| Medium | 13 | 60 | 2 |
| Hard | 2 | 43 | 9 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 22 | 25 | 2 |
| Medium | 5 | 33 | 0 |
| Hard | 4 | 23 | 10 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 35 | 29 | 0 |
| Medium | 12 | 61 | 3 |
| Hard | 4 | 45 | 14 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 32/58 (55.2%, 95%CI [42.5, 67.3%]) | 58/77 (75.3%, 95%CI [64.6, 83.6%]) | 8/51 (15.7%, 95%CI [8.2, 28.0%]) |
| ICL | 35/71 (49.3%, 95%CI [38.0, 60.7%]) | 60/75 (80.0%, 95%CI [69.6, 87.5%]) | 9/54 (16.7%, 95%CI [9.0, 28.7%]) |
| SelfRefine | 22/49 (44.9%, 95%CI [31.9, 58.7%]) | 33/38 (86.8%, 95%CI [72.7, 94.2%]) | 10/37 (27.0%, 95%CI [15.4, 43.0%]) |
| Ours | 35/64 (54.7%, 95%CI [42.6, 66.3%]) | 61/76 (80.3%, 95%CI [70.0, 87.7%]) | 14/63 (22.2%, 95%CI [13.7, 33.9%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 47.2% | 56.6% | 58.6% | 26.2% |
| ICL | 47.1% | 57.9% | 56.6% | 26.9% |
| SelfRefine | 50.4% | 55.0% | 55.5% | 40.8% |
| Ours | 51.2% | 60.9% | 57.8% | 35.0% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.420 | 186 |
| ICL | 0.447 | 200 |
| SelfRefine | 0.411 | 124 |
| Ours | 0.503 | 203 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 58 | 32 | 55.2% | 32 | 25 | 1 |
| ICL | 71 | 35 | 49.3% | 35 | 34 | 2 |
| SelfRefine | 49 | 22 | 44.9% | 22 | 25 | 2 |
| Ours | 64 | 35 | 54.7% | 35 | 29 | 0 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 77 | 58 | 75.3% | 18 | 58 | 1 |
| ICL | 75 | 60 | 80.0% | 13 | 60 | 2 |
| SelfRefine | 38 | 33 | 86.8% | 5 | 33 | 0 |
| Ours | 76 | 61 | 80.3% | 12 | 61 | 3 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 51 | 8 | 15.7% | 5 | 38 | 8 |
| ICL | 54 | 9 | 16.7% | 2 | 43 | 9 |
| SelfRefine | 37 | 10 | 27.0% | 4 | 23 | 10 |
| Ours | 63 | 14 | 22.2% | 4 | 45 | 14 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 8/51 (15.7%, 95%CI [8.2, 28.0%]) | |
| ICL | 9/54 (16.7%, 95%CI [9.0, 28.7%]) | |
| SelfRefine | 10/37 (27.0%, 95%CI [15.4, 43.0%]) | |
| Ours | 14/63 (22.2%, 95%CI [13.7, 33.9%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 6/51 (11.8%, 95%CI [5.5, 23.4%]) | |
| ICL | 9/54 (16.7%, 95%CI [9.0, 28.7%]) | |
| SelfRefine | 9/37 (24.3%, 95%CI [13.4, 40.1%]) | |
| Ours | 12/63 (19.0%, 95%CI [11.2, 30.4%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/51 (0.0%, 95%CI [0.0, 7.0%]) | |
| ICL | 0/54 (0.0%, 95%CI [0.0, 6.6%]) | |
| SelfRefine | 0/37 (0.0%, 95%CI [0.0, 9.4%]) | |
| Ours | 2/63 (3.2%, 95%CI [0.9, 10.9%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 8 | 8 |
| ICL | 9 | 9 |
| SelfRefine | 10 | 10 |
| Ours | 14 | 14 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 94 | 0 | 0 | 0 | 94 |
| Medium | 0 | 101 | 0 | 0 | 101 |
| Hard | 0 | 0 | 97 | 0 | 97 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 52 | 94 | 55.3% |
| Medium | 91 | 101 | 90.1% |
| Hard | 73 | 97 | 75.3% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 64 | 54.7% | N/A |
| two_node_relation | 76 | 80.3% | N/A |
| multi_node_chain | 63 | 22.2% | 22.2% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 96 | 32.9% |
| motivates | 31 | 10.6% |
| causes | 29 | 9.9% |
| causes → results_in | 19 | 6.5% |
| explains | 12 | 4.1% |
| temporal_before → causes | 10 | 3.4% |
| temporal_before | 10 | 3.4% |
| causes → motivates | 8 | 2.7% |
| results_in | 8 | 2.7% |
| enables | 6 | 2.1% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| results_in | 1 | 29 | 9 | 39 |
| causes | 1 | 30 | 7 | 38 |
| temporal_before | 3 | 14 | 7 | 24 |
| motivates | 0 | 15 | 4 | 19 |
| explains | 4 | 6 | 1 | 11 |
| supports_inference | 0 | 3 | 4 | 7 |
| enables | 0 | 3 | 1 | 4 |
| contrasts_with | 0 | 1 | 1 | 2 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 57 | 94 | 60.6% |
| Medium | two_node_relation | 36 | 101 | 35.6% |
| Hard | multi_node_chain | 50 | 97 | 51.5% |

### GP-7. Hard Pure Temporal Chain Count: 1

### GP-8. Easy Answer-Sentence-Alone Rate: 54.7% (35/64)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 46.0% (29/63)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 63.8% (203/318) | 58.5% (186/318) | 62.9% (200/318) | 39.0% (124/318) | +5.3pp | +0.9pp | +24.8pp |
| overall_accuracy | 54.2% | 52.7% | 52.0% | 52.4% | +1.5pp | +2.2pp | +1.8pp |
| macro_accuracy | 52.4% | 48.7% | 48.7% | 52.9% | +3.7pp | +3.7pp | -0.5pp |
| macro_f1 | 51.2% | 47.2% | 47.1% | 50.4% | +4.1pp | +4.1pp | +0.8pp |
| spearman | 0.503 | 0.420 | 0.447 | 0.411 | +0.083 | +0.056 | +0.092 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected stories >= 100 | PASS | selected stories=106 |
| every story has equal Easy/Med/Hard count | PASS | all 106 stories have 1E/1M/1H |
| all methods have identical denominator | PASS | denominators: {'Direct': 318, 'ICL': 318, 'SelfRefine': 318, 'Ours': 318} |
| Ours quality_pass not << baselines | PASS | Ours=203/318 |
| Ours overall accuracy >= each baseline | PASS | Ours=54.2% |
| Ours macro accuracy >= each baseline | FAIL | Ours=52.4% |
| Ours Spearman >= each baseline | PASS | Ours=0.503 |

**Overall: SOME CRITERIA FAILED**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 61.8% | 56.5% | +5.34pp | [-2.29pp, +12.98pp] | 0.1830 | 131 | no |
| Direct | macro_accuracy | 55.6% | 50.8% | +4.80pp | [-2.11pp, +11.88pp] | 0.1754 | 131 | no |
| Direct | macro_f1 | 55.5% | 49.5% | +6.03pp | [-1.23pp, +13.64pp] | 0.1104 | 131 | no |
| Direct | spearman | 0.508 | 0.463 | +0.045 | [-0.100, +0.186] | 0.5360 | 131 | no |
| ICL | overall_accuracy | 57.4% | 54.1% | +3.38pp | [-4.05pp, +10.81pp] | 0.4304 | 148 | no |
| ICL | macro_accuracy | 53.1% | 49.8% | +3.33pp | [-3.99pp, +10.84pp] | 0.3810 | 148 | no |
| ICL | macro_f1 | 52.8% | 49.0% | +3.77pp | [-4.37pp, +12.03pp] | 0.3722 | 148 | no |
| ICL | spearman | 0.545 | 0.478 | +0.067 | [-0.063, +0.193] | 0.3052 | 148 | no |
| SelfRefine | overall_accuracy | 64.6% | 54.9% | +9.76pp | [+1.22pp, +19.51pp] | 0.0496 | 82 | yes |
| SelfRefine | macro_accuracy | 62.7% | 52.6% | +10.11pp | [+0.31pp, +20.46pp] | 0.0436 | 82 | yes |
| SelfRefine | macro_f1 | 63.5% | 51.3% | +12.25pp | [+1.23pp, +24.11pp] | 0.0322 | 82 | yes |
| SelfRefine | spearman | 0.659 | 0.383 | +0.276 | [+0.094, +0.478] | 0.0016 | 82 | yes |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 30.2% (32/106) | 54.7% (58/106) | 7.5% (8/106) | 30.8% (98/318) | 318 |
| ICL | 33.0% (35/106) | 56.6% (60/106) | 8.5% (9/106) | 32.7% (104/318) | 318 |
| SelfRefine | 20.8% (22/106) | 31.1% (33/106) | 9.4% (10/106) | 20.4% (65/318) | 318 |
| Ours | 33.0% (35/106) | 57.5% (61/106) | 13.2% (14/106) | 34.6% (110/318) | 318 |

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

**Sample 2** (story=Snow-man, target=Easy):

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

**Sample 3** (story=Snow-man, target=Easy):

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
| Direct | gen error: degenerate output | 32 |
| Direct | not answerable | 29 |
| Direct | wrong answer | 27 |
| Direct | other | 15 |
| Direct | not fluent | 9 |
| Direct | gen error: empty | 7 |
| Direct | gen error: no question mark | 7 |
| Direct | gen error: grammar: word repetition: where | 1 |
| Direct | gen error: grammar: word repetition: realized | 1 |
| Direct | gen error: does not end with ? | 1 |
| Direct | gen error: grammar: word repetition: he | 1 |
| Direct | gen error: grammar: bad start: whywhere | 1 |
| Direct | gen error: grammar: word repetition: come | 1 |
| ICL | gen error: degenerate output | 32 |
| ICL | not answerable | 22 |
| ICL | wrong answer | 21 |
| ICL | other | 13 |
| ICL | gen error: empty | 13 |
| ICL | gen error: no question mark | 7 |
| ICL | not fluent | 7 |
| ICL | gen error: grammar: word repetition: why | 2 |
| ICL | gen error: grammar: word repetition: feel | 1 |
| SelfRefine | gen error: degenerate initial output | 88 |
| SelfRefine | gen error: initial generation failed | 30 |
| SelfRefine | not answerable | 23 |
| SelfRefine | wrong answer | 16 |
| SelfRefine | gen error: no question mark | 15 |
| SelfRefine | other | 10 |
| SelfRefine | not fluent | 7 |
| SelfRefine | gen error: answer leakage | 1 |
| SelfRefine | gen error: grammar: word repetition: night | 1 |
| SelfRefine | gen error: does not end with ? | 1 |
| SelfRefine | gen error: grammar: too short | 1 |
| SelfRefine | gen error: grammar: word repetition: why | 1 |
| Ours | gen error: degenerate output | 35 |
| Ours | gen error: graph_invalid | 26 |
| Ours | gen error: self-check failed: answer mismatch | 11 |
| Ours | gen error: parse failure | 11 |
| Ours | gen error: no question mark | 5 |
| Ours | gen error: question length out of range | 4 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1-2 sentences; focus mismatch; graph_policy non-compliant (multi_node_chain) | 3 |
| Ours | gen error: does not end with ? | 3 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch; graph_policy non-compliant (answer_only) | 3 |
| Ours | gen error: grammar: word repetition: why | 2 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch | 2 |
| Ours | gen error: self-check failed: answer mismatch; graph_policy non-compliant (answer_only) | 2 |
| Ours | not fluent | 1 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1 sentence; focus mismatch; graph_policy non-compliant (two_node_relation) | 1 |
| Ours | gen error: grammar: bad start: whywhy | 1 |
| Ours | gen error: self-check failed: graph_policy non-compliant (answer_only) | 1 |
| Ours | gen error: empty | 1 |
| Ours | gen error: self-check failed: focus mismatch | 1 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1-2 sentences; graph_policy non-compliant (multi_node_chain) | 1 |
| Ours | gen error: grammar: word repetition: he | 1 |

## 13. Story-Matched Diagnostics

### 13a. Story Summary

| Selected stories | 106 |
| Candidates per story | 1 × 3 levels |

**Equal Easy/Med/Hard per story:** YES (expected 4 per level per story)

### 13b. Story-Level Average Accuracy by Method (quality-pass, judge-ok)

| Method | Mean story acc | Median story acc | Std | N stories |
|---|---:|---:|---:|---:|
| Direct | 50.9% | 66.7% | 39.2% | 76 |
| ICL | 50.0% | 66.7% | 38.1% | 76 |
| SelfRefine | 41.8% | 100.0% | 44.0% | 55 |
| Ours | 51.1% | 66.7% | 36.6% | 79 |

### 13c. Story-Level Win/Tie/Loss (Ours vs Baseline)

| Baseline | Ours Wins | Ties | Ours Losses | N stories |
|---|---:|---:|---:|---:|
| Direct | 36 | 37 | 33 | 106 |
| ICL | 37 | 37 | 32 | 106 |
| SelfRefine | 49 | 29 | 28 | 106 |

### 13d. Story-Level Spearman (stories with all 3 levels valid)

| Method | Mean story rho | N valid stories | N skipped |
|---|---:|---:|---:|
| Direct | 0.366 | 24 | 82 |
| ICL | 0.457 | 26 | 80 |
| SelfRefine | 0.325 | 8 | 98 |
| Ours | 0.654 | 28 | 78 |

### 13e. Per-Story Failure Counts by Method

| Method | Stories with 0 fails | 1 fail | 2 fails | 3 fails |
|---|---:|---:|---:|---:|
| Direct | 24 | 42 | 30 | 10 |
| ICL | 26 | 48 | 26 | 6 |
| SelfRefine | 8 | 24 | 52 | 22 |
| Ours | 28 | 46 | 27 | 5 |

## 14. Retry & Budget Diagnostics

### 14a. Attempts per Method

| Method | Avg attempts | Max attempts | Total |
|---|---:|---:|---:|
| Direct | 1.79 | 3 | 318 |
| ICL | 1.86 | 3 | 318 |
| SelfRefine | 2.26 | 3 | 318 |
| Ours | 2.94 | 4 | 318 |

### 14b. Attempt Distribution by Method

| Method | 1 attempt | 2 attempts | 3+ attempts |
|---|---:|---:|---:|
| Direct | 156 | 73 | 89 |
| ICL | 139 | 84 | 95 |
| SelfRefine | 118 | 0 | 200 |
| Ours | 79 | 32 | 207 |

### 14c. Ours Repair Prompt Usage

| Metric | Value |
|---|---|
| Repair prompt used | 143/318 (45.0%) |
| Repair success | 17 |

### 14d. Ours Graph Policy Self-Check Failure Rate

| Self-check failures | 76/292 (26.0%) |

### 14e. Failure Reason Distribution by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 38 |
| Direct | degenerate output | 32 |
| Direct | wrong answer | 27 |
| Direct | unknown | 15 |
| Direct | not fluent | 13 |
| Direct | empty output | 7 |
| ICL | degenerate output | 32 |
| ICL | not answerable | 30 |
| ICL | wrong answer | 21 |
| ICL | unknown | 13 |
| ICL | empty output | 13 |
| ICL | not fluent | 9 |
| SelfRefine | degenerate output | 88 |
| SelfRefine | not answerable | 31 |
| SelfRefine | initial generation failed | 30 |
| SelfRefine | not fluent | 18 |
| SelfRefine | wrong answer | 17 |
| SelfRefine | None | 10 |
| Ours | graph issue | 37 |
| Ours | degenerate output | 35 |
| Ours | self-check failed | 14 |
| Ours | parse failure | 11 |
| Ours | not answerable | 7 |
| Ours | not fluent | 6 |
| Ours | question length | 4 |
| Ours | empty output | 1 |

### 14f. Ours Retry Reason Distribution (from attempt traces)

| Retry reason | Count |
|---|---:|
| ok | 453 |
| degenerate output | 353 |
| parse failure | 45 |
| no question mark | 29 |
| question length out of range | 15 |
| does not end with ? | 13 |
| empty | 10 |
| grammar: word repetition: why | 3 |
| grammar: word repetition: what | 2 |
| grammar: bad start: whom | 2 |

## 15. Similarity Diagnostics (diagnostic only, no filtering)

### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)

| Method | E-M mean | M-H mean | E-H mean | n pairs |
|---|---:|---:|---:|---:|
| Direct | 0.084 | 0.098 | 0.083 | 87 |
| ICL | 0.102 | 0.094 | 0.077 | 81 |
| SelfRefine | 0.074 | 0.083 | 0.071 | 42 |
| Ours | 0.088 | 0.094 | 0.071 | 78 |

### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)

| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |
|---|---:|---:|---:|
| Direct | 0.104 | 0.103 | 0.067 |
| ICL | 0.080 | 0.101 | 0.063 |
| SelfRefine | 0.055 | 0.040 | 0.037 |
| Ours | 0.116 | 0.099 | 0.060 |

### 15c. Difficulty Collapse Counts by Method

| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |
|---|---:|---:|---:|---:|---:|
| Direct | 24 | 4 | 4 | 0 | 0 |
| ICL | 26 | 8 | 7 | 1 | 0 |
| SelfRefine | 8 | 2 | 2 | 0 | 0 |
| Ours | 28 | 6 | 6 | 0 | 0 |

## 16. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: Snow-man
- Question: Who were they in the garden with the snow man ?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: Snow-man
- Question: Who came into the garden with a young man?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: Snow-man
- Question: Who came into the garden with a young man?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: Snow-man
- Question: Who had come into the garden together?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: Snow-man
- Question: What does the snow man want to do but doesn't know how to do?
- Target answer: move away from that place .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: a-legend-of-knockmany
- Question: What was Far Rua's intention towards Finn according to the story?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: a-legend-of-knockmany
- Question: Why did Far Rua travel to the causeway?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: Snow-man
- Question: What larger goal was the snow man pursuing by wanting to slide on the ice?
- Target answer: move away from that place .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**Direct Example:**
- Story: hans-in-luck
- Question: Why did poor Hans grow pale with fright and ask for help?
- Target answer: because the pig he had was stolen .
- Target: Hard, Predicted: Hard

**ICL Example:**
- Story: bokwewa-the-humpback
- Question: Why did Kwasynd want the woman to be his wife even though she did not respond to him?
- Target answer: she was dead .
- Target: Hard, Predicted: Hard

**SelfRefine Example:**
- Story: how-princess-pride-was-broken
- Question: Why was the princess so annoyed about the price of the musical box according to the story?
- Target answer: annoyed .
- Target: Hard, Predicted: Hard

**Ours Example:**
- Story: gold-tree-and-silver-tree
- Question: How did the second wife come to discover Princess Gold-Tree in the forbidden room?
- Target answer: gold - tree was in the room .
- Target: Hard, Predicted: Hard

