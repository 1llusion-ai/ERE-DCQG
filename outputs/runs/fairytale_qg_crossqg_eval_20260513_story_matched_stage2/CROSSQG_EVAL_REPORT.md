# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-13 15:55:08

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
| Easy | 92 | 106 | 86.8% |
| Medium | 99 | 106 | 93.4% |
| Hard | 95 | 106 | 89.6% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 301 | 318 | 94.7% |
| ICL | 281 | 318 | 88.4% |
| SelfRefine | 189 | 318 | 59.4% |
| Ours | 263 | 318 | 82.7% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 73 | 74 | 56 | 203 | 318 | 63.8% |
| ICL | 66 | 74 | 62 | 202 | 318 | 63.5% |
| SelfRefine | 45 | 50 | 29 | 124 | 318 | 39.0% |
| Ours | 64 | 61 | 54 | 179 | 318 | 56.3% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 296 | 22 | 318 | 6.9% |
| ICL | 278 | 40 | 318 | 12.6% |
| SelfRefine | 189 | 129 | 318 | 40.6% |
| Ours | 260 | 58 | 318 | 18.2% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 52.7% | 107/203 (52.7%, 95%CI [45.9, 59.5%]) | 49.9% |
| ICL | 51.0% | 103/202 (51.0%, 95%CI [44.1, 57.8%]) | 49.1% |
| SelfRefine | 56.5% | 70/124 (56.5%, 95%CI [47.7, 64.9%]) | 51.2% |
| Ours | 57.0% | 101/179 (56.4%, 95%CI [49.1, 63.5%]) | 56.2% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 35 | 37 | 1 |
| Medium | 12 | 62 | 0 |
| Hard | 5 | 41 | 10 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 31 | 34 | 1 |
| Medium | 12 | 60 | 2 |
| Hard | 6 | 44 | 12 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 22 | 22 | 1 |
| Medium | 8 | 42 | 0 |
| Hard | 3 | 20 | 6 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 33 | 31 | 0 |
| Medium | 7 | 51 | 3 |
| Hard | 6 | 30 | 18 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 35/73 (47.9%, 95%CI [36.9, 59.2%]) | 62/74 (83.8%, 95%CI [73.8, 90.5%]) | 10/56 (17.9%, 95%CI [10.0, 29.8%]) |
| ICL | 31/66 (47.0%, 95%CI [35.4, 58.8%]) | 60/74 (81.1%, 95%CI [70.7, 88.4%]) | 12/62 (19.4%, 95%CI [11.4, 30.9%]) |
| SelfRefine | 22/45 (48.9%, 95%CI [35.0, 63.0%]) | 42/50 (84.0%, 95%CI [71.5, 91.7%]) | 6/29 (20.7%, 95%CI [9.8, 38.4%]) |
| Ours | 33/64 (51.6%, 95%CI [39.6, 63.4%]) | 51/61 (83.6%, 95%CI [72.4, 90.8%]) | 18/54 (33.3%, 95%CI [22.2, 46.6%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 47.9% | 56.0% | 57.9% | 29.9% |
| ICL | 47.2% | 53.9% | 56.6% | 31.2% |
| SelfRefine | 50.8% | 56.4% | 62.7% | 33.3% |
| Ours | 55.7% | 60.0% | 59.0% | 48.0% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.422 | 203 |
| ICL | 0.407 | 202 |
| SelfRefine | 0.411 | 124 |
| Ours | 0.500 | 179 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 73 | 35 | 47.9% | 35 | 37 | 1 |
| ICL | 66 | 31 | 47.0% | 31 | 34 | 1 |
| SelfRefine | 45 | 22 | 48.9% | 22 | 22 | 1 |
| Ours | 64 | 33 | 51.6% | 33 | 31 | 0 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 74 | 62 | 83.8% | 12 | 62 | 0 |
| ICL | 74 | 60 | 81.1% | 12 | 60 | 2 |
| SelfRefine | 50 | 42 | 84.0% | 8 | 42 | 0 |
| Ours | 61 | 51 | 83.6% | 7 | 51 | 3 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 56 | 10 | 17.9% | 5 | 41 | 10 |
| ICL | 62 | 12 | 19.4% | 6 | 44 | 12 |
| SelfRefine | 29 | 6 | 20.7% | 3 | 20 | 6 |
| Ours | 54 | 18 | 33.3% | 6 | 30 | 18 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 10/56 (17.9%, 95%CI [10.0, 29.8%]) | |
| ICL | 12/62 (19.4%, 95%CI [11.4, 30.9%]) | |
| SelfRefine | 6/29 (20.7%, 95%CI [9.8, 38.4%]) | |
| Ours | 18/54 (33.3%, 95%CI [22.2, 46.6%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 9/56 (16.1%, 95%CI [8.7, 27.8%]) | |
| ICL | 12/62 (19.4%, 95%CI [11.4, 30.9%]) | |
| SelfRefine | 5/29 (17.2%, 95%CI [7.6, 34.5%]) | |
| Ours | 17/54 (31.5%, 95%CI [20.7, 44.7%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/56 (0.0%, 95%CI [0.0, 6.4%]) | |
| ICL | 0/62 (0.0%, 95%CI [0.0, 5.8%]) | |
| SelfRefine | 0/29 (0.0%, 95%CI [0.0, 11.7%]) | |
| Ours | 1/54 (1.9%, 95%CI [0.3, 9.8%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 10 | 10 |
| ICL | 12 | 12 |
| SelfRefine | 6 | 6 |
| Ours | 18 | 18 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 92 | 0 | 0 | 0 | 92 |
| Medium | 0 | 99 | 0 | 0 | 99 |
| Hard | 0 | 0 | 95 | 0 | 95 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 61 | 92 | 66.3% |
| Medium | 82 | 99 | 82.8% |
| Hard | 69 | 95 | 72.6% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 64 | 51.6% | N/A |
| two_node_relation | 61 | 83.6% | N/A |
| multi_node_chain | 54 | 33.3% | 33.3% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 93 | 32.5% |
| motivates | 34 | 11.9% |
| causes | 31 | 10.8% |
| causes → results_in | 17 | 5.9% |
| temporal_before → causes | 12 | 4.2% |
| explains | 12 | 4.2% |
| temporal_before | 10 | 3.5% |
| causes → motivates | 9 | 3.1% |
| results_in | 6 | 2.1% |
| enables | 4 | 1.4% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| causes | 5 | 21 | 12 | 38 |
| results_in | 3 | 14 | 8 | 25 |
| temporal_before | 2 | 14 | 9 | 25 |
| motivates | 1 | 12 | 7 | 20 |
| explains | 1 | 7 | 4 | 12 |
| supports_inference | 0 | 4 | 3 | 7 |
| enables | 0 | 1 | 2 | 3 |
| contrasts_with | 0 | 1 | 0 | 1 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 32 | 92 | 34.8% |
| Medium | two_node_relation | 44 | 99 | 44.4% |
| Hard | multi_node_chain | 46 | 95 | 48.4% |

### GP-7. Hard Pure Temporal Chain Count: 1

### GP-8. Easy Answer-Sentence-Alone Rate: 51.6% (33/64)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 55.6% (30/54)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 56.3% (179/318) | 63.8% (203/318) | 63.5% (202/318) | 39.0% (124/318) | -7.5pp | -7.2pp | +17.3pp |
| overall_accuracy | 57.0% | 52.7% | 51.0% | 56.5% | +4.3pp | +6.0pp | +0.5pp |
| macro_accuracy | 56.2% | 49.9% | 49.1% | 51.2% | +6.3pp | +7.0pp | +5.0pp |
| macro_f1 | 55.7% | 47.9% | 47.2% | 50.8% | +7.7pp | +8.4pp | +4.8pp |
| spearman | 0.500 | 0.422 | 0.407 | 0.411 | +0.078 | +0.093 | +0.090 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected stories >= 100 | PASS | selected stories=106 |
| every story has equal Easy/Med/Hard count | PASS | all 106 stories have 1E/1M/1H |
| all methods have identical denominator | PASS | denominators: {'Direct': 318, 'ICL': 318, 'SelfRefine': 318, 'Ours': 318} |
| Ours quality_pass not << baselines | PASS | Ours=179/318 |
| Ours overall accuracy >= each baseline | PASS | Ours=57.0% |
| Ours macro accuracy >= each baseline | PASS | Ours=56.2% |
| Ours Spearman >= each baseline | PASS | Ours=0.500 |

**Overall: Ours has the highest global overall accuracy, macro accuracy, macro F1, and corrected Spearman. However, paired bootstrap CIs include 0 for all comparisons (Section 9), so statistical significance is not established at n=150 per level. Do not claim statistically significant superiority.**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 57.4% | 53.2% | +4.26pp | [-2.84pp, +11.35pp] | 0.2946 | 141 | no |
| Direct | macro_accuracy | 55.7% | 50.8% | +4.87pp | [-2.78pp, +12.63pp] | 0.2196 | 141 | no |
| Direct | macro_f1 | 55.3% | 49.8% | +5.50pp | [-3.60pp, +14.56pp] | 0.2348 | 141 | no |
| Direct | spearman | 0.463 | 0.492 | -0.029 | [-0.165, +0.102] | 0.6776 | 141 | no |
| ICL | overall_accuracy | 59.3% | 54.5% | +4.88pp | [-2.44pp, +12.20pp] | 0.2192 | 123 | no |
| ICL | macro_accuracy | 57.5% | 52.4% | +5.11pp | [-2.18pp, +12.38pp] | 0.1696 | 123 | no |
| ICL | macro_f1 | 57.5% | 51.5% | +5.95pp | [-2.38pp, +14.27pp] | 0.1568 | 123 | no |
| ICL | spearman | 0.515 | 0.451 | +0.064 | [-0.103, +0.231] | 0.4610 | 123 | no |
| SelfRefine | overall_accuracy | 62.1% | 56.3% | +5.75pp | [-2.30pp, +13.79pp] | 0.1906 | 87 | no |
| SelfRefine | macro_accuracy | 58.6% | 52.3% | +6.33pp | [-2.24pp, +15.23pp] | 0.1478 | 87 | no |
| SelfRefine | macro_f1 | 59.7% | 52.5% | +7.15pp | [-2.97pp, +17.94pp] | 0.1634 | 87 | no |
| SelfRefine | spearman | 0.524 | 0.449 | +0.075 | [-0.084, +0.227] | 0.3418 | 87 | no |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 33.0% (35/106) | 58.5% (62/106) | 9.4% (10/106) | 33.6% (107/318) | 318 |
| ICL | 29.2% (31/106) | 56.6% (60/106) | 11.3% (12/106) | 32.4% (103/318) | 318 |
| SelfRefine | 20.8% (22/106) | 39.6% (42/106) | 5.7% (6/106) | 22.0% (70/318) | 318 |
| Ours | 31.1% (33/106) | 48.1% (51/106) | 17.0% (18/106) | 32.1% (102/318) | 318 |

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
| Direct | not answerable | 35 |
| Direct | wrong answer | 25 |
| Direct | gen error: degenerate output | 16 |
| Direct | other | 12 |
| Direct | not fluent | 12 |
| Direct | gen error: empty | 8 |
| Direct | gen error: no question mark | 3 |
| Direct | gen error: grammar: word repetition: con | 1 |
| Direct | gen error: grammar: word repetition: realized | 1 |
| Direct | gen error: grammar: word repetition: his | 1 |
| Direct | gen error: grammar: word repetition: why | 1 |
| ICL | gen error: degenerate output | 36 |
| ICL | not answerable | 24 |
| ICL | wrong answer | 24 |
| ICL | gen error: empty | 11 |
| ICL | other | 8 |
| ICL | not fluent | 5 |
| ICL | gen error: no question mark | 4 |
| ICL | gen error: answer leakage | 1 |
| ICL | gen error: does not end with ? | 1 |
| ICL | gen error: grammar: bad start: because | 1 |
| ICL | gen error: grammar: word repetition: feel | 1 |
| SelfRefine | gen error: degenerate initial output | 101 |
| SelfRefine | gen error: initial generation failed | 28 |
| SelfRefine | not answerable | 23 |
| SelfRefine | wrong answer | 15 |
| SelfRefine | gen error: no question mark | 11 |
| SelfRefine | not fluent | 8 |
| SelfRefine | other | 7 |
| SelfRefine | gen error: grammar: word repetition: become | 1 |
| Ours | gen error: degenerate output | 40 |
| Ours | gen error: graph_invalid | 32 |
| Ours | gen error: self-check failed: answer mismatch | 11 |
| Ours | gen error: no question mark | 11 |
| Ours | not fluent | 10 |
| Ours | gen error: parse failure | 9 |
| Ours | gen error: self-check failed: answer mismatch; graph_policy non-compliant (answer_only) | 5 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1-2 sentences; focus mismatch; graph_policy non-compliant (multi_node_chain) | 3 |
| Ours | gen error: question length out of range | 3 |
| Ours | gen error: self-check failed: answer mismatch; graph_policy non-compliant (multi_node_chain) | 2 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch; graph_policy non-compliant (two_node_relation) | 2 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch; graph_policy non-compliant (multi_node_chain) | 2 |
| Ours | not answerable | 1 |
| Ours | gen error: self-check failed: focus mismatch; graph_policy non-compliant (multi_node_chain) | 1 |
| Ours | gen error: does not end with ? | 1 |
| Ours | gen error: grammar: word repetition: mount | 1 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1 sentence; focus mismatch; graph_policy non-compliant (two_node_relation) | 1 |
| Ours | wrong answer | 1 |
| Ours | gen error: grammar: word repetition: help | 1 |
| Ours | gen error: grammar: bad start: whatwhat | 1 |
| Ours | gen error: grammar: word repetition: easier | 1 |

## 13. Story-Matched Diagnostics

### 13a. Story Summary

| Selected stories | 106 |
| Candidates per story | 1 × 3 levels |

**Equal Easy/Med/Hard per story:** YES (expected 4 per level per story)

### 13b. Story-Level Average Accuracy by Method (quality-pass, judge-ok)

| Method | Mean story acc | Median story acc | Std | N stories |
|---|---:|---:|---:|---:|
| Direct | 51.4% | 66.7% | 37.6% | 78 |
| ICL | 50.5% | 50.0% | 37.2% | 79 |
| SelfRefine | 45.4% | 100.0% | 44.2% | 60 |
| Ours | 49.7% | 66.7% | 40.8% | 71 |

### 13c. Story-Level Win/Tie/Loss (Ours vs Baseline)

| Baseline | Ours Wins | Ties | Ours Losses | N stories |
|---|---:|---:|---:|---:|
| Direct | 29 | 45 | 32 | 106 |
| ICL | 38 | 35 | 33 | 106 |
| SelfRefine | 33 | 43 | 30 | 106 |

### 13d. Story-Level Spearman (stories with all 3 levels valid)

| Method | Mean story rho | N valid stories | N skipped |
|---|---:|---:|---:|
| Direct | 0.317 | 25 | 81 |
| ICL | 0.408 | 27 | 79 |
| SelfRefine | 0.433 | 6 | 100 |
| Ours | 0.589 | 19 | 87 |

### 13e. Per-Story Failure Counts by Method

| Method | Stories with 0 fails | 1 fail | 2 fails | 3 fails |
|---|---:|---:|---:|---:|
| Direct | 25 | 54 | 20 | 7 |
| ICL | 27 | 47 | 27 | 5 |
| SelfRefine | 6 | 31 | 44 | 25 |
| Ours | 19 | 44 | 34 | 9 |

## 14. Retry & Budget Diagnostics

### 14a. Attempts per Method

| Method | Avg attempts | Max attempts | Total |
|---|---:|---:|---:|
| Direct | 1.71 | 3 | 318 |
| ICL | 1.85 | 3 | 318 |
| SelfRefine | 2.19 | 3 | 318 |
| Ours | 2.73 | 4 | 318 |

### 14b. Attempt Distribution by Method

| Method | 1 attempt | 2 attempts | 3+ attempts |
|---|---:|---:|---:|
| Direct | 165 | 80 | 73 |
| ICL | 143 | 79 | 96 |
| SelfRefine | 129 | 0 | 189 |
| Ours | 96 | 33 | 189 |

### 14c. Ours Repair Prompt Usage

| Metric | Value |
|---|---|
| Repair prompt used | 122/318 (38.4%) |
| Repair success | 19 |

### 14d. Ours Graph Policy Self-Check Failure Rate

| Self-check failures | 74/286 (25.9%) |

### 14e. Failure Reason Distribution by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 38 |
| Direct | wrong answer | 25 |
| Direct | degenerate output | 16 |
| Direct | not fluent | 16 |
| Direct | unknown | 12 |
| Direct | empty output | 8 |
| ICL | degenerate output | 36 |
| ICL | not answerable | 31 |
| ICL | wrong answer | 24 |
| ICL | empty output | 11 |
| ICL | unknown | 8 |
| ICL | not fluent | 6 |
| SelfRefine | degenerate output | 101 |
| SelfRefine | not answerable | 33 |
| SelfRefine | initial generation failed | 28 |
| SelfRefine | wrong answer | 16 |
| SelfRefine | not fluent | 9 |
| SelfRefine | None | 7 |
| Ours | graph issue | 48 |
| Ours | degenerate output | 40 |
| Ours | not fluent | 15 |
| Ours | not answerable | 12 |
| Ours | self-check failed | 11 |
| Ours | parse failure | 9 |
| Ours | question length | 3 |
| Ours | wrong answer | 1 |

### 14f. Ours Retry Reason Distribution (from attempt traces)

| Retry reason | Count |
|---|---:|
| ok | 426 |
| degenerate output | 271 |
| parse failure | 52 |
| no question mark | 44 |
| question length out of range | 24 |
| empty | 11 |
| grammar: word repetition: what | 7 |
| does not end with ? | 6 |
| grammar: word repetition: his | 2 |
| grammar: word repetition: mount | 2 |

## 15. Similarity Diagnostics (diagnostic only, no filtering)

### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)

| Method | E-M mean | M-H mean | E-H mean | n pairs |
|---|---:|---:|---:|---:|
| Direct | 0.088 | 0.101 | 0.078 | 92 |
| ICL | 0.089 | 0.091 | 0.078 | 81 |
| SelfRefine | 0.064 | 0.080 | 0.069 | 46 |
| Ours | 0.083 | 0.081 | 0.064 | 75 |

### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)

| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |
|---|---:|---:|---:|
| Direct | 0.135 | 0.105 | 0.094 |
| ICL | 0.079 | 0.074 | 0.089 |
| SelfRefine | 0.072 | 0.051 | 0.036 |
| Ours | 0.086 | 0.092 | 0.045 |

### 15c. Difficulty Collapse Counts by Method

| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |
|---|---:|---:|---:|---:|---:|
| Direct | 25 | 5 | 5 | 0 | 0 |
| ICL | 27 | 8 | 8 | 0 | 0 |
| SelfRefine | 6 | 3 | 3 | 0 | 0 |
| Ours | 19 | 5 | 5 | 0 | 0 |

## 16. Stage 2 Focus & Difficulty Realization Diagnostics (Ours)

### 16a. Focus Distribution by Target Difficulty (Ours, quality-pass)

| Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| chain_explanation | 0 | 0 | 54 |
| direct_answer | 64 | 0 | 0 |
| relation_question | 0 | 61 | 0 |

### 16b. Node-Level Focus Distribution (pre-override, for comparison)

| Node Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| bridge | 32 | 26 | 15 |
| motivation | 18 | 27 | 15 |
| outcome | 1 | 3 | 12 |
| state | 13 | 5 | 12 |

### 16c. Easy answer_sentence_alone=yes by Focus Type (Ours, quality-pass, target=Easy)

| Focus | Total | ASA=yes | Rate |
|---|---:|---:|---:|
| direct_answer | 64 | 33 | 51.6% |

### 16d. Hard answer_sentence_alone=no by Focus Type (Ours, quality-pass, target=Hard)

| Focus | Total | ASA=no | Rate |
|---|---:|---:|---:|
| chain_explanation | 54 | 30 | 55.6% |

### 16e. Graph Policy Compliance by Focus Type (Ours, quality-pass)

| Focus | Total | GPC=yes | Rate |
|---|---:|---:|---:|
| chain_explanation | 54 | 49 | 90.7% |
| direct_answer | 64 | 52 | 81.2% |
| relation_question | 61 | 60 | 98.4% |

### 16f. Repair Usage by Target Difficulty (Ours)

| Difficulty | Total | Repair Used | Repair Success | Repair Rate |
|---|---:|---:|---:|---:|
| Easy | 106 | 32 | 6 | 30.2% |
| Medium | 106 | 44 | 7 | 41.5% |
| Hard | 106 | 46 | 6 | 43.4% |

### 16g. Top 10 Easy Failures (Ours, quality-pass, predicted != Easy)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | dschang-liang | What did the old man tell dschang liang to do? | wanted dschang liang to fetch  | Medium | direct_answer | partial | answer_local_wording |
| 2 | a-legend-of-knockmany | What did Far Rua hear that invited him to the top of Cullamo | he heard three whistles . | Medium | direct_answer | partial | bridge_detected |
| 3 | canonbie-dick-and-thomas-of-er | What kind of gold did the stranger give to Canonbie Dick? | the coins were not the gold th | Medium | direct_answer | partial | bridge_detected |
| 4 | comrade | Who did the youth have for a comrade after their journey? | the youth would have company . | Medium | direct_answer | partial | bridge_detected |
| 5 | goblin-huckster | What did the huckster always have at Christmas that made the | the huckster had jam at christ | Medium | direct_answer | partial | bridge_detected |
| 6 | lasse-my-thrall | What did he do when he saw that something was written on the | read the words outloud . | Medium | direct_answer | partial | bridge_detected |
| 7 | osseo-the-son-of-the-evening-s | What did Osseo do that made Oweenee happy despite his povert | he was a good man . | Medium | direct_answer | partial | bridge_detected |
| 8 | prince-featherhead-and-the-pri | What did King Bruin learn about the king and queen's situati | king bruin heard the king had  | Medium | direct_answer | partial | bridge_detected |
| 9 | strong-desire-and-the-red-sorc | What was the gender of the person who came to be the wife of | he was a man . | Medium | direct_answer | partial | bridge_detected |
| 10 | the-escape-of-the-mouse | Where did the townsfolk buy their saddles from? | the townsfolk only bought from | Medium | direct_answer | partial | bridge_detected |

### 16h. Top 10 Hard Failures (Ours, quality-pass, predicted != Hard)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | brave-tin-soldier | What led to the little dancer's destruction after the door o | she fluttered into the stove a | Easy | chain_explanation | yes | answer_sentence_alone_yes |
| 2 | the-seal-catcher-and-the-merma | What motivated the seals to crowd around their comrade and s | happy . | Easy | chain_explanation | yes | answer_sentence_alone_yes |
| 3 | the-ugly-duckling | What motivated the old duck to believe the egg was a turkey' | she did not believe that it 's | Easy | chain_explanation | yes | answer_sentence_alone_yes |
| 4 | the-well-o-the-worlds-end | What motivated the old woman to run forward and try to stop  | scared . | Easy | chain_explanation | yes | answer_sentence_alone_yes |
| 5 | white-hare-and-crocodiles | What motivated the crocodile to claim there are more crocodi | there are many crocodiles arou | Easy | chain_explanation | yes | answer_sentence_alone_yes |
| 6 | the-black-bull-of-norroway | Why was the princess ultimately so upset after being lifted  | upset . | Medium | chain_explanation | partial | chain_not_realized |
| 7 | a-lost-paradise | What motivated the charcoal-burner and his wife to follow th | grateful . | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 8 | canonbie-dick-and-thomas-of-er | What motivated the stranger to reveal his identity to Canonb | the stranger wanted to reveal  | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 9 | comrade | What led the youth to lose the shears while guarding them, g | he was not paying attention . | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 10 | dschang-liang | What motivated Dschang Liang to flee from the emperor's rage | one of the traveling coaches w | Medium | chain_explanation | partial | focus_mismatch_direct_wording |

## 17. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: Snow-man
- Question: Who came into the garden with a young man?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: Snow-man
- Question: Who came into the garden with a young man?
- Target answer: a young girl and a young man .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: brother-sister
- Question: Why did the brother and sister have no good days after their mother died?
- Target answer: their stepmother was mean to them .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: Snow-man
- Question: Who had come into the garden with the young man?
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
- Question: Why was Far Rua coming to the causeway according to the story?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: a-lost-paradise
- Question: Why did the charcoal-burner and his wife become poorer and poorer?
- Target answer: they could not find work .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: Snow-man
- Question: Why did the snow man want to slide along yonder on the ice?
- Target answer: move away from that place .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**Direct Example:**
- Story: bokwewa-the-humpback
- Question: Why did Kwasynd lift the woman and carry her to his brother?
- Target answer: she was dead .
- Target: Hard, Predicted: Hard

**ICL Example:**
- Story: hans-in-luck
- Question: Why did the peasant suspect Hans of having a stolen pig?
- Target answer: because the pig he had was stolen .
- Target: Hard, Predicted: Hard

**SelfRefine Example:**
- Story: bokwewa-the-humpback
- Question: Why did Kwasynd want to restore the woman's life to his brother?
- Target answer: she was dead .
- Target: Hard, Predicted: Hard

**Ours Example:**
- Story: Snow-man
- Question: What led the yard-dog to ultimately be chained up and lose his bone?
- Target answer: he had to .
- Target: Hard, Predicted: Hard

