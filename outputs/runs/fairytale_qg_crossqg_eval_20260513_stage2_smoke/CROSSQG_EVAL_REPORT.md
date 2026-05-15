# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-13 09:45:20

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Selection mode | story_matched |
| Target per level | 150 |
| Selected Easy | 10 |
| Selected Medium | 10 |
| Selected Hard | 10 |
| Total selected | 30 |
| Total stories | 10 |
| Total generations | 120 |
| Candidates per level per story | 1 |
| Max stories | 10 |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 8 | 10 | 80.0% |
| Medium | 10 | 10 | 100.0% |
| Hard | 10 | 10 | 100.0% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 29 | 30 | 96.7% |
| ICL | 29 | 30 | 96.7% |
| SelfRefine | 23 | 30 | 76.7% |
| Ours | 27 | 30 | 90.0% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 4 | 8 | 4 | 16 | 30 | 53.3% |
| ICL | 7 | 9 | 4 | 20 | 30 | 66.7% |
| SelfRefine | 7 | 5 | 5 | 17 | 30 | 56.7% |
| Ours | 6 | 8 | 7 | 21 | 30 | 70.0% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 28 | 2 | 30 | 6.7% |
| ICL | 29 | 1 | 30 | 3.3% |
| SelfRefine | 23 | 7 | 30 | 23.3% |
| Ours | 27 | 3 | 30 | 10.0% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 43.8% | 7/16 (43.8%, 95%CI [23.1, 66.8%]) | 37.5% |
| ICL | 70.0% | 14/20 (70.0%, 95%CI [48.1, 85.5%]) | 57.1% |
| SelfRefine | 47.1% | 8/17 (47.1%, 95%CI [26.2, 69.0%]) | 45.7% |
| Ours | 66.7% | 14/21 (66.7%, 95%CI [45.4, 82.8%]) | 65.1% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 2 | 2 | 0 |
| Medium | 3 | 5 | 0 |
| Hard | 1 | 3 | 0 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 5 | 2 | 0 |
| Medium | 0 | 9 | 0 |
| Hard | 0 | 4 | 0 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 3 | 0 |
| Medium | 2 | 3 | 0 |
| Hard | 2 | 2 | 1 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 2 | 0 |
| Medium | 0 | 8 | 0 |
| Hard | 0 | 5 | 2 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 2/4 (50.0%, 95%CI [15.0, 85.0%]) | 5/8 (62.5%, 95%CI [30.6, 86.3%]) | 0/4 (0.0%, 95%CI [0.0, 49.0%]) |
| ICL | 5/7 (71.4%, 95%CI [35.9, 91.8%]) | 9/9 (100.0%, 95%CI [70.1, 100.0%]) | 0/4 (0.0%, 95%CI [0.0, 49.0%]) |
| SelfRefine | 4/7 (57.1%, 95%CI [25.0, 84.2%]) | 3/5 (60.0%, 95%CI [23.1, 88.2%]) | 1/5 (20.0%, 95%CI [3.6, 62.4%]) |
| Ours | 4/6 (66.7%, 95%CI [30.0, 90.3%]) | 8/8 (100.0%, 95%CI [67.6, 100.0%]) | 2/7 (28.6%, 95%CI [8.2, 64.1%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 31.9% | 40.0% | 55.6% | 0.0% |
| ICL | 52.8% | 83.3% | 75.0% | 0.0% |
| SelfRefine | 44.3% | 53.3% | 46.2% | 33.3% |
| Ours | 64.7% | 80.0% | 69.6% | 44.4% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.183 | 16 |
| ICL | 0.702 | 20 |
| SelfRefine | 0.220 | 17 |
| Ours | 0.703 | 21 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 4 | 2 | 50.0% | 2 | 2 | 0 |
| ICL | 7 | 5 | 71.4% | 5 | 2 | 0 |
| SelfRefine | 7 | 4 | 57.1% | 4 | 3 | 0 |
| Ours | 6 | 4 | 66.7% | 4 | 2 | 0 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 8 | 5 | 62.5% | 3 | 5 | 0 |
| ICL | 9 | 9 | 100.0% | 0 | 9 | 0 |
| SelfRefine | 5 | 3 | 60.0% | 2 | 3 | 0 |
| Ours | 8 | 8 | 100.0% | 0 | 8 | 0 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 4 | 0 | 0.0% | 1 | 3 | 0 |
| ICL | 4 | 0 | 0.0% | 0 | 4 | 0 |
| SelfRefine | 5 | 1 | 20.0% | 2 | 2 | 1 |
| Ours | 7 | 2 | 28.6% | 0 | 5 | 2 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 0/4 (0.0%, 95%CI [0.0, 49.0%]) | |
| ICL | 0/4 (0.0%, 95%CI [0.0, 49.0%]) | |
| SelfRefine | 1/5 (20.0%, 95%CI [3.6, 62.4%]) | |
| Ours | 2/7 (28.6%, 95%CI [8.2, 64.1%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/4 (0.0%, 95%CI [0.0, 49.0%]) | |
| ICL | 0/4 (0.0%, 95%CI [0.0, 49.0%]) | |
| SelfRefine | 0/5 (0.0%, 95%CI [0.0, 43.4%]) | |
| Ours | 2/7 (28.6%, 95%CI [8.2, 64.1%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/4 (0.0%, 95%CI [0.0, 49.0%]) | |
| ICL | 0/4 (0.0%, 95%CI [0.0, 49.0%]) | |
| SelfRefine | 0/5 (0.0%, 95%CI [0.0, 43.4%]) | |
| Ours | 1/7 (14.3%, 95%CI [2.6, 51.3%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 0 | 0 |
| ICL | 0 | 0 |
| SelfRefine | 1 | 1 |
| Ours | 2 | 2 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 8 | 0 | 0 | 0 | 8 |
| Medium | 0 | 10 | 0 | 0 | 10 |
| Hard | 0 | 0 | 10 | 0 | 10 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 5 | 8 | 62.5% |
| Medium | 8 | 10 | 80.0% |
| Hard | 9 | 10 | 90.0% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 6 | 66.7% | N/A |
| two_node_relation | 8 | 100.0% | N/A |
| multi_node_chain | 7 | 28.6% | 28.6% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 8 | 28.6% |
| causes | 4 | 14.3% |
| motivates | 3 | 10.7% |
| causes → results_in | 2 | 7.1% |
| results_in | 2 | 7.1% |
| explains → causes | 1 | 3.6% |
| temporal_before → causes → motivates | 1 | 3.6% |
| temporal_before → temporal_before → causes → motivates | 1 | 3.6% |
| supports_inference → motivates | 1 | 3.6% |
| causes → causes | 1 | 3.6% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| causes | 0 | 4 | 2 | 6 |
| motivates | 0 | 2 | 1 | 3 |
| temporal_before | 0 | 2 | 0 | 2 |
| supports_inference | 0 | 1 | 1 | 2 |
| results_in | 0 | 2 | 0 | 2 |
| explains | 0 | 1 | 0 | 1 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 0 | 8 | 0.0% |
| Medium | two_node_relation | 2 | 10 | 20.0% |
| Hard | multi_node_chain | 4 | 10 | 40.0% |

### GP-7. Hard Pure Temporal Chain Count: 0

### GP-8. Easy Answer-Sentence-Alone Rate: 66.7% (4/6)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 85.7% (6/7)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 70.0% (21/30) | 53.3% (16/30) | 66.7% (20/30) | 56.7% (17/30) | +16.7pp | +3.3pp | +13.3pp |
| overall_accuracy | 66.7% | 43.8% | 70.0% | 47.1% | +22.9pp | -3.3pp | +19.6pp |
| macro_accuracy | 65.1% | 37.5% | 57.1% | 45.7% | +27.6pp | +7.9pp | +19.4pp |
| macro_f1 | 64.7% | 31.9% | 52.8% | 44.3% | +32.8pp | +11.9pp | +20.4pp |
| spearman | 0.703 | 0.183 | 0.702 | 0.220 | +0.520 | +0.001 | +0.483 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected stories >= 100 | FAIL | selected stories=10 |
| every story has equal Easy/Med/Hard count | PASS | all 10 stories have 1E/1M/1H |
| all methods have identical denominator | PASS | denominators: {'Direct': 30, 'ICL': 30, 'SelfRefine': 30, 'Ours': 30} |
| Ours quality_pass not << baselines | PASS | Ours=21/30 |
| Ours overall accuracy >= each baseline | FAIL | Ours=66.7% |
| Ours macro accuracy >= each baseline | PASS | Ours=65.1% |
| Ours Spearman >= each baseline | PASS | Ours=0.703 |

**Overall: SOME CRITERIA FAILED**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 75.0% | 41.7% | +33.33pp | [+8.33pp, +58.33pp] | 0.0168 | 12 | yes |
| Direct | macro_accuracy | 66.7% | 38.9% | +27.78pp | [+6.67pp, +54.17pp] | 0.0168 | 12 | yes |
| Direct | macro_f1 | 70.0% | 31.5% | +38.52pp | [+6.38pp, +65.66pp] | 0.0168 | 12 | yes |
| Direct | spearman | 0.719 | 0.236 | +0.483 | [-0.094, +1.135] | 0.1124 | 12 | no |
| ICL | overall_accuracy | 75.0% | 75.0% | +0.00pp | [-18.75pp, +18.75pp] | 1.0000 | 16 | no |
| ICL | macro_accuracy | 64.4% | 60.0% | +4.44pp | [-16.67pp, +33.33pp] | 0.9574 | 16 | no |
| ICL | macro_f1 | 68.3% | 56.3% | +12.04pp | [-17.59pp, +34.66pp] | 0.7750 | 16 | no |
| ICL | spearman | 0.703 | 0.751 | -0.048 | [-0.363, +0.218] | 0.8772 | 16 | no |
| SelfRefine | overall_accuracy | 71.4% | 50.0% | +21.43pp | [+0.00pp, +42.86pp] | 0.0698 | 14 | no |
| SelfRefine | macro_accuracy | 71.7% | 51.7% | +20.00pp | [+0.00pp, +41.11pp] | 0.0698 | 14 | no |
| SelfRefine | macro_f1 | 71.4% | 47.5% | +23.95pp | [+1.83pp, +48.98pp] | 0.0224 | 14 | yes |
| SelfRefine | spearman | 0.763 | 0.319 | +0.444 | [+0.091, +0.812] | 0.0094 | 14 | yes |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 20.0% (2/10) | 50.0% (5/10) | 0.0% (0/10) | 23.3% (7/30) | 30 |
| ICL | 50.0% (5/10) | 90.0% (9/10) | 0.0% (0/10) | 46.7% (14/30) | 30 |
| SelfRefine | 40.0% (4/10) | 30.0% (3/10) | 10.0% (1/10) | 26.7% (8/30) | 30 |
| Ours | 40.0% (4/10) | 80.0% (8/10) | 20.0% (2/10) | 46.7% (14/30) | 30 |

## 11. Difficulty Judge Prompt Audit

The difficulty judge is **blind**: it sees only the story, generated question, and expected answer.
It does NOT see the target difficulty. Below are 3 sample prompts to confirm.

**Sample 1** (story=a-legend-of-knockmany, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " my curse upon you !
[S1] " she exclaimed , " you 've disgraced me .
[S2] i now change you into a grey stone .
[S3] lie there as a testimony of what has happened , and may evil betide the first living man that will ever attempt to move or injure you !
[S4] " and , sure enough , there it lies to this day , with the mark of the four fingers and thumb imprinted on it , exactly as it came out of her hand .
[S5] " never mind , " said granua , " i must only do the best i can with far rua .
[S6] if all fail , i 'll give him a cast of heather broth , or a panada of oak bark .
[S7] but , above all things , think of some plan to get finn out of the scrape he 's in , or he 's a lost man .
[S8] you know you us
... [truncated]
```

**Sample 2** (story=a-legend-of-knockmany, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " my curse upon you !
[S1] " she exclaimed , " you 've disgraced me .
[S2] i now change you into a grey stone .
[S3] lie there as a testimony of what has happened , and may evil betide the first living man that will ever attempt to move or injure you !
[S4] " and , sure enough , there it lies to this day , with the mark of the four fingers and thumb imprinted on it , exactly as it came out of her hand .
[S5] " never mind , " said granua , " i must only do the best i can with far rua .
[S6] if all fail , i 'll give him a cast of heather broth , or a panada of oak bark .
[S7] but , above all things , think of some plan to get finn out of the scrape he 's in , or he 's a lost man .
[S8] you know you us
... [truncated]
```

**Sample 3** (story=a-legend-of-knockmany, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " my curse upon you !
[S1] " she exclaimed , " you 've disgraced me .
[S2] i now change you into a grey stone .
[S3] lie there as a testimony of what has happened , and may evil betide the first living man that will ever attempt to move or injure you !
[S4] " and , sure enough , there it lies to this day , with the mark of the four fingers and thumb imprinted on it , exactly as it came out of her hand .
[S5] " never mind , " said granua , " i must only do the best i can with far rua .
[S6] if all fail , i 'll give him a cast of heather broth , or a panada of oak bark .
[S7] but , above all things , think of some plan to get finn out of the scrape he 's in , or he 's a lost man .
[S8] you know you us
... [truncated]
```

**Confirmed:** No target difficulty appears in any judge prompt. Blind evaluation is correct.

## 12. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 4 |
| Direct | gen error: degenerate output | 3 |
| Direct | not fluent | 3 |
| Direct | wrong answer | 1 |
| Direct | gen error: grammar: word repetition: night | 1 |
| Direct | gen error: empty | 1 |
| Direct | gen error: does not end with ? | 1 |
| ICL | wrong answer | 3 |
| ICL | other | 2 |
| ICL | not answerable | 2 |
| ICL | gen error: does not end with ? | 2 |
| ICL | gen error: degenerate output | 1 |
| SelfRefine | gen error: degenerate initial output | 4 |
| SelfRefine | gen error: initial generation failed | 3 |
| SelfRefine | other | 2 |
| SelfRefine | not answerable | 2 |
| SelfRefine | wrong answer | 1 |
| SelfRefine | gen error: no question mark | 1 |
| Ours | gen error: degenerate output | 2 |
| Ours | gen error: graph_invalid | 2 |
| Ours | not fluent | 1 |
| Ours | gen error: no question mark | 1 |
| Ours | gen error: grammar: word repetition: come | 1 |
| Ours | gen error: does not end with ? | 1 |
| Ours | gen error: parse failure | 1 |

## 13. Story-Matched Diagnostics

### 13a. Story Summary

| Selected stories | 10 |
| Candidates per story | 1 × 3 levels |

**Equal Easy/Med/Hard per story:** YES (expected 4 per level per story)

### 13b. Story-Level Average Accuracy by Method (quality-pass, judge-ok)

| Method | Mean story acc | Median story acc | Std | N stories |
|---|---:|---:|---:|---:|
| Direct | 48.3% | 100.0% | 45.0% | 6 |
| ICL | 75.0% | 66.7% | 21.4% | 10 |
| SelfRefine | 40.0% | 66.7% | 38.2% | 6 |
| Ours | 73.3% | 100.0% | 28.1% | 10 |

### 13c. Story-Level Win/Tie/Loss (Ours vs Baseline)

| Baseline | Ours Wins | Ties | Ours Losses | N stories |
|---|---:|---:|---:|---:|
| Direct | 5 | 3 | 2 | 10 |
| ICL | 2 | 6 | 2 | 10 |
| SelfRefine | 6 | 4 | 0 | 10 |

### 13d. Story-Level Spearman (stories with all 3 levels valid)

| Method | Mean story rho | N valid stories | N skipped |
|---|---:|---:|---:|
| Direct | 0.000 | 1 | 9 |
| ICL | 0.866 | 3 | 7 |
| SelfRefine | 0.433 | 2 | 8 |
| Ours | 0.289 | 3 | 7 |

### 13e. Per-Story Failure Counts by Method

| Method | Stories with 0 fails | 1 fail | 2 fails | 3 fails |
|---|---:|---:|---:|---:|
| Direct | 1 | 4 | 5 | 0 |
| ICL | 3 | 4 | 3 | 0 |
| SelfRefine | 2 | 4 | 3 | 1 |
| Ours | 3 | 5 | 2 | 0 |

## 14. Retry & Budget Diagnostics

### 14a. Attempts per Method

| Method | Avg attempts | Max attempts | Total |
|---|---:|---:|---:|
| Direct | 1.83 | 3 | 30 |
| ICL | 1.77 | 3 | 30 |
| SelfRefine | 2.53 | 3 | 30 |
| Ours | 2.63 | 4 | 30 |

### 14b. Attempt Distribution by Method

| Method | 1 attempt | 2 attempts | 3+ attempts |
|---|---:|---:|---:|
| Direct | 13 | 9 | 8 |
| ICL | 16 | 5 | 9 |
| SelfRefine | 7 | 0 | 23 |
| Ours | 8 | 5 | 17 |

### 14c. Ours Repair Prompt Usage

| Metric | Value |
|---|---|
| Repair prompt used | 6/30 (20.0%) |
| Repair success | 1 |

### 14d. Ours Graph Policy Self-Check Failure Rate

| Self-check failures | 6/28 (21.4%) |

### 14e. Failure Reason Distribution by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not fluent | 5 |
| Direct | not answerable | 4 |
| Direct | degenerate output | 3 |
| Direct | wrong answer | 1 |
| Direct | empty output | 1 |
| ICL | not answerable | 4 |
| ICL | wrong answer | 3 |
| ICL | unknown | 2 |
| ICL | degenerate output | 1 |
| SelfRefine | degenerate output | 4 |
| SelfRefine | initial generation failed | 3 |
| SelfRefine | not answerable | 3 |
| SelfRefine | None | 2 |
| SelfRefine | wrong answer | 1 |
| Ours | not answerable | 3 |
| Ours | degenerate output | 2 |
| Ours | graph issue | 2 |
| Ours | not fluent | 1 |
| Ours | parse failure | 1 |

### 14f. Ours Retry Reason Distribution (from attempt traces)

| Retry reason | Count |
|---|---:|
| ok | 37 |
| degenerate output | 19 |
| no question mark | 9 |
| question length out of range | 5 |
| grammar: word repetition: come | 3 |
| parse failure | 3 |
| grammar: word repetition: what | 2 |
| does not end with ? | 1 |

## 15. Similarity Diagnostics (diagnostic only, no filtering)

### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)

| Method | E-M mean | M-H mean | E-H mean | n pairs |
|---|---:|---:|---:|---:|
| Direct | 0.042 | 0.080 | 0.054 | 9 |
| ICL | 0.063 | 0.076 | 0.071 | 9 |
| SelfRefine | 0.067 | 0.069 | 0.086 | 5 |
| Ours | 0.060 | 0.081 | 0.053 | 8 |

### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)

| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |
|---|---:|---:|---:|
| Direct | 0.050 | 0.083 | 0.050 |
| ICL | 0.050 | 0.100 | 0.025 |
| SelfRefine | 0.050 | 0.000 | 0.025 |
| Ours | 0.133 | 0.158 | 0.033 |

### 15c. Difficulty Collapse Counts by Method

| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |
|---|---:|---:|---:|---:|---:|
| Direct | 1 | 1 | 1 | 0 | 0 |
| ICL | 3 | 0 | 0 | 0 | 0 |
| SelfRefine | 2 | 1 | 1 | 0 | 0 |
| Ours | 3 | 2 | 2 | 0 | 0 |

## 16. Stage 2 Focus & Difficulty Realization Diagnostics (Ours)

### 16a. Focus Distribution by Target Difficulty (Ours, quality-pass)

| Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| chain_explanation | 0 | 0 | 7 |
| direct_answer | 6 | 0 | 0 |
| relation_question | 0 | 8 | 0 |

### 16b. Node-Level Focus Distribution (pre-override, for comparison)

| Node Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| bridge | 2 | 2 | 3 |
| motivation | 1 | 4 | 1 |
| outcome | 0 | 1 | 1 |
| state | 3 | 1 | 2 |

### 16c. Easy answer_sentence_alone=yes by Focus Type (Ours, quality-pass, target=Easy)

| Focus | Total | ASA=yes | Rate |
|---|---:|---:|---:|
| direct_answer | 6 | 4 | 66.7% |

### 16d. Hard answer_sentence_alone=no by Focus Type (Ours, quality-pass, target=Hard)

| Focus | Total | ASA=no | Rate |
|---|---:|---:|---:|
| chain_explanation | 7 | 6 | 85.7% |

### 16e. Graph Policy Compliance by Focus Type (Ours, quality-pass)

| Focus | Total | GPC=yes | Rate |
|---|---:|---:|---:|
| chain_explanation | 7 | 7 | 100.0% |
| direct_answer | 6 | 4 | 66.7% |
| relation_question | 8 | 8 | 100.0% |

### 16f. Repair Usage by Target Difficulty (Ours)

| Difficulty | Total | Repair Used | Repair Success | Repair Rate |
|---|---:|---:|---:|---:|
| Easy | 10 | 0 | 0 | 0.0% |
| Medium | 10 | 2 | 1 | 20.0% |
| Hard | 10 | 4 | 0 | 40.0% |

### 16g. Top 10 Easy Failures (Ours, quality-pass, predicted != Easy)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | a-legend-of-knockmany | What did FarRua know by the smoke and whistles on the hillto | he heard three whistles . | Medium | direct_answer | partial | bridge_detected |
| 2 | goblin-huckster | What did the huckster always have at Christmas that made the | the huckster had jam at christ | Medium | direct_answer | partial | bridge_detected |

### 16h. Top 10 Hard Failures (Ours, quality-pass, predicted != Hard)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | a-legend-of-knockmany | What motivated Finn to suddenly feel affection for his wife  | happy . | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 2 | goblin-huckster | What motivated the little goblin to stand still on tiptoe, p | because he thought the room lo | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 3 | strong-desire-and-the-red-sorc | What motivated Red Head to launch his royal bark and cross o | strong desire rejected everyon | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 4 | the-one-handed-girl | What motivated the prince to seek his parents' consent after | he wanted their approval to ma | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 5 | weendigoes-and-the-bone-dwarf | What motivated the youngest boy to leave and hide in the hol | the beautiful boy did not want | Medium | chain_explanation | no | focus_mismatch_direct_wording |

## 17. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: cuchulain-of-muirthemne
- Question: When did Conchubar suggest unyoking the chariots?
- Target answer: it was night .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: goblin-huckster
- Question: What did the huckster always have at Christmas that kept the goblin with him?
- Target answer: the huckster had jam at christmas .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: cuchulain-of-muirthemne
- Question: Why did Conchubar want to unyoke the chariots?
- Target answer: it was night .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: cuchulain-of-muirthemne
- Question: What time of day was it when Conchubar gave the order to unyoke the chariots?
- Target answer: it was night .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: a-legend-of-knockmany
- Question: Why did Far Rua travel to the causeway?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: a-legend-of-knockmany
- Question: What was Far Rua's intention towards Finn according to the story?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: a-legend-of-knockmany
- Question: What was Far Rua's intention when he traveled to the causeway?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: a-legend-of-knockmany
- Question: Why was Far Rua determined to catch Finn?
- Target answer: he wanted to give finn a considerable beating .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**SelfRefine Example:**
- Story: the-bones-of-djulung
- Question: Why did the youngest sister feel suddenly tired and fall asleep for days after her sisters caught and cooked the djulung without her knowledge?
- Target answer: the fish was eaten by her sisters .
- Target: Hard, Predicted: Hard

**Ours Example:**
- Story: notscha
- Question: What motivated Notscha to destroy himself according to the story?
- Target answer: he wanted to save his parents .
- Target: Hard, Predicted: Hard

