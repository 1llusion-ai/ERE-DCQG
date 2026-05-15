# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-13 17:33:07

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Selection mode | story_matched_suitable |
| Target per level | 150 |
| Selected Easy | 10 |
| Selected Medium | 10 |
| Selected Hard | 10 |
| Total selected | 30 |
| Total stories | 10 |
| Total generations | 120 |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 10 | 10 | 100.0% |
| Medium | 10 | 10 | 100.0% |
| Hard | 10 | 10 | 100.0% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 27 | 30 | 90.0% |
| ICL | 27 | 30 | 90.0% |
| SelfRefine | 21 | 30 | 70.0% |
| Ours | 28 | 30 | 93.3% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 6 | 7 | 7 | 20 | 30 | 66.7% |
| ICL | 6 | 7 | 7 | 20 | 30 | 66.7% |
| SelfRefine | 5 | 5 | 7 | 17 | 30 | 56.7% |
| Ours | 7 | 6 | 6 | 19 | 30 | 63.3% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 27 | 3 | 30 | 10.0% |
| ICL | 27 | 3 | 30 | 10.0% |
| SelfRefine | 21 | 9 | 30 | 30.0% |
| Ours | 28 | 2 | 30 | 6.7% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 30.0% | 6/20 (30.0%, 95%CI [14.5, 51.9%]) | 28.6% |
| ICL | 30.0% | 6/20 (30.0%, 95%CI [14.5, 51.9%]) | 29.4% |
| SelfRefine | 29.4% | 5/17 (29.4%, 95%CI [13.3, 53.1%]) | 33.3% |
| Ours | 26.3% | 5/19 (26.3%, 95%CI [11.8, 48.8%]) | 27.0% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 0 | 6 | 0 |
| Medium | 2 | 5 | 0 |
| Hard | 0 | 6 | 1 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 1 | 4 | 1 |
| Medium | 2 | 4 | 1 |
| Hard | 1 | 5 | 1 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 1 | 4 | 0 |
| Medium | 1 | 4 | 0 |
| Hard | 2 | 5 | 0 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 1 | 5 | 1 |
| Medium | 2 | 4 | 0 |
| Hard | 0 | 6 | 0 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 0/6 (0.0%, 95%CI [0.0, 39.0%]) | 5/7 (71.4%, 95%CI [35.9, 91.8%]) | 1/7 (14.3%, 95%CI [2.6, 51.3%]) |
| ICL | 1/6 (16.7%, 95%CI [3.0, 56.4%]) | 4/7 (57.1%, 95%CI [25.0, 84.2%]) | 1/7 (14.3%, 95%CI [2.6, 51.3%]) |
| SelfRefine | 1/5 (20.0%, 95%CI [3.6, 62.4%]) | 4/5 (80.0%, 95%CI [37.6, 96.4%]) | 0/7 (0.0%, 95%CI [0.0, 35.4%]) |
| Ours | 1/7 (14.3%, 95%CI [2.6, 51.3%]) | 4/6 (66.7%, 95%CI [30.0, 90.3%]) | 0/6 (0.0%, 95%CI [0.0, 39.0%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 22.2% | 0.0% | 41.7% | 25.0% |
| ICL | 26.7% | 20.0% | 40.0% | 20.0% |
| SelfRefine | 22.2% | 22.2% | 44.4% | 0.0% |
| Ours | 19.4% | 20.0% | 38.1% | 0.0% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.175 | 20 |
| ICL | 0.009 | 20 |
| SelfRefine | -0.090 | 17 |
| Ours | -0.008 | 19 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 6 | 0 | 0.0% | 0 | 6 | 0 |
| ICL | 6 | 1 | 16.7% | 1 | 4 | 1 |
| SelfRefine | 5 | 1 | 20.0% | 1 | 4 | 0 |
| Ours | 7 | 1 | 14.3% | 1 | 5 | 1 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 7 | 5 | 71.4% | 2 | 5 | 0 |
| ICL | 7 | 4 | 57.1% | 2 | 4 | 1 |
| SelfRefine | 5 | 4 | 80.0% | 1 | 4 | 0 |
| Ours | 6 | 4 | 66.7% | 2 | 4 | 0 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 7 | 1 | 14.3% | 0 | 6 | 1 |
| ICL | 7 | 1 | 14.3% | 1 | 5 | 1 |
| SelfRefine | 7 | 0 | 0.0% | 2 | 5 | 0 |
| Ours | 6 | 0 | 0.0% | 0 | 6 | 0 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 1/7 (14.3%, 95%CI [2.6, 51.3%]) | |
| ICL | 1/7 (14.3%, 95%CI [2.6, 51.3%]) | |
| SelfRefine | 0/7 (0.0%, 95%CI [0.0, 35.4%]) | |
| Ours | 0/6 (0.0%, 95%CI [0.0, 39.0%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 1/7 (14.3%, 95%CI [2.6, 51.3%]) | |
| ICL | 1/7 (14.3%, 95%CI [2.6, 51.3%]) | |
| SelfRefine | 0/7 (0.0%, 95%CI [0.0, 35.4%]) | |
| Ours | 0/6 (0.0%, 95%CI [0.0, 39.0%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/7 (0.0%, 95%CI [0.0, 35.4%]) | |
| ICL | 0/7 (0.0%, 95%CI [0.0, 35.4%]) | |
| SelfRefine | 0/7 (0.0%, 95%CI [0.0, 35.4%]) | |
| Ours | 0/6 (0.0%, 95%CI [0.0, 39.0%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 1 | 1 |
| ICL | 1 | 1 |
| SelfRefine | 0 | 0 |
| Ours | 0 | 0 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 10 | 0 | 0 | 0 | 10 |
| Medium | 0 | 10 | 0 | 0 | 10 |
| Hard | 0 | 0 | 10 | 0 | 10 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 7 | 10 | 70.0% |
| Medium | 9 | 10 | 90.0% |
| Hard | 6 | 10 | 60.0% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 7 | 14.3% | N/A |
| two_node_relation | 6 | 66.7% | N/A |
| multi_node_chain | 6 | 0.0% | 0.0% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 10 | 33.3% |
| motivates | 4 | 13.3% |
| causes | 3 | 10.0% |
| enables | 2 | 6.7% |
| supports_inference → supports_inference → supports_inference → results_in | 1 | 3.3% |
| motivates → causes | 1 | 3.3% |
| causes → explains → results_in | 1 | 3.3% |
| explains → results_in | 1 | 3.3% |
| causes → results_in | 1 | 3.3% |
| temporal_before → causes | 1 | 3.3% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| supports_inference | 0 | 4 | 0 | 4 |
| results_in | 0 | 4 | 0 | 4 |
| causes | 0 | 4 | 0 | 4 |
| explains | 0 | 2 | 0 | 2 |
| temporal_before | 0 | 2 | 0 | 2 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 3 | 10 | 30.0% |
| Medium | two_node_relation | 4 | 10 | 40.0% |
| Hard | multi_node_chain | 4 | 10 | 40.0% |

### GP-7. Hard Pure Temporal Chain Count: 0

### GP-8. Easy Answer-Sentence-Alone Rate: 14.3% (1/7)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 66.7% (4/6)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 63.3% (19/30) | 66.7% (20/30) | 66.7% (20/30) | 56.7% (17/30) | -3.3pp | -3.3pp | +6.7pp |
| overall_accuracy | 26.3% | 30.0% | 30.0% | 29.4% | -3.7pp | -3.7pp | -3.1pp |
| macro_accuracy | 27.0% | 28.6% | 29.4% | 33.3% | -1.6pp | -2.4pp | -6.3pp |
| macro_f1 | 19.4% | 22.2% | 26.7% | 22.2% | -2.9pp | -7.3pp | -2.9pp |
| spearman | -0.008 | 0.175 | 0.009 | -0.090 | -0.182 | -0.017 | +0.083 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected >= 150 per level | FAIL | Easy=10, Medium=10, Hard=10 |
| Ours quality_pass not << baselines | PASS | Ours=19/30 |
| Ours overall accuracy >= each baseline | FAIL | Ours=26.3% |
| Ours macro accuracy >= each baseline | FAIL | Ours=27.0% |
| Ours Spearman >= each baseline | FAIL | Ours=-0.008 |

**Overall: SOME CRITERIA FAILED**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 25.0% | 31.2% | -6.25pp | [-25.00pp, +12.50pp] | 0.7812 | 16 | no |
| Direct | macro_accuracy | 22.2% | 28.9% | -6.67pp | [-28.89pp, +13.33pp] | 0.6682 | 16 | no |
| Direct | macro_f1 | 14.0% | 25.1% | -11.11pp | [-29.26pp, +6.67pp] | 0.4564 | 16 | no |
| Direct | spearman | -0.176 | 0.176 | -0.353 | [-0.827, +0.047] | 0.1750 | 16 | no |
| ICL | overall_accuracy | 29.4% | 29.4% | +0.00pp | [-17.65pp, +17.65pp] | 1.0000 | 17 | no |
| ICL | macro_accuracy | 27.8% | 27.8% | +0.00pp | [-16.67pp, +16.67pp] | 1.0000 | 17 | no |
| ICL | macro_f1 | 21.4% | 21.5% | -0.04pp | [-10.03pp, +10.92pp] | 1.0000 | 17 | no |
| ICL | spearman | -0.009 | -0.155 | +0.147 | [-0.316, +0.588] | 0.5436 | 17 | no |
| SelfRefine | overall_accuracy | 23.1% | 30.8% | -7.69pp | [-30.77pp, +15.38pp] | 0.7700 | 13 | no |
| SelfRefine | macro_accuracy | 25.0% | 33.3% | -8.33pp | [-33.33pp, +26.67pp] | 0.7700 | 13 | no |
| SelfRefine | macro_f1 | 19.8% | 23.7% | -3.94pp | [-21.31pp, +14.83pp] | 0.6814 | 13 | no |
| SelfRefine | spearman | 0.075 | -0.142 | +0.217 | [-0.416, +0.870] | 0.5100 | 13 | no |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0.0% (0/10) | 50.0% (5/10) | 10.0% (1/10) | 20.0% (6/30) | 30 |
| ICL | 10.0% (1/10) | 40.0% (4/10) | 10.0% (1/10) | 20.0% (6/30) | 30 |
| SelfRefine | 10.0% (1/10) | 40.0% (4/10) | 0.0% (0/10) | 16.7% (5/30) | 30 |
| Ours | 10.0% (1/10) | 40.0% (4/10) | 0.0% (0/10) | 16.7% (5/30) | 30 |

## 11. Difficulty Judge Prompt Audit

The difficulty judge is **blind**: it sees only the story, generated question, and expected answer.
It does NOT see the target difficulty. Below are 3 sample prompts to confirm.

**Sample 1** (story=bokwewa-the-humpback, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] he was soon , having already yielded to temptation by the way , overcome by their fair looks and soft speeches , and he was not long afterward seen beating corn with the women , having entirely abandoned all further quest for his lost wife .
[S1] meantime , bokwewa , alone in the lodge , often musing upon the discourse of the spirit - wife , who was gone , waited patiently his brother 's return .
[S2] after the lapse of several years , when no tidings could be had , he set out in search of him , and he arrived in safety among the soft and idle people of the south .
[S3] he met the same allurements by the way , and they gathered around him on his coming as they had around his brother kwasynd ; but bo
... [truncated]
```

**Sample 2** (story=bokwewa-the-humpback, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] he was soon , having already yielded to temptation by the way , overcome by their fair looks and soft speeches , and he was not long afterward seen beating corn with the women , having entirely abandoned all further quest for his lost wife .
[S1] meantime , bokwewa , alone in the lodge , often musing upon the discourse of the spirit - wife , who was gone , waited patiently his brother 's return .
[S2] after the lapse of several years , when no tidings could be had , he set out in search of him , and he arrived in safety among the soft and idle people of the south .
[S3] he met the same allurements by the way , and they gathered around him on his coming as they had around his brother kwasynd ; but bo
... [truncated]
```

**Sample 3** (story=bokwewa-the-humpback, target=Hard):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] she made no resistance , but turning as she left the lodge , she cast upon bokwewa a smile of kind regard , and was at once , with her companion , gone from his view .
[S1] he ran to the door and glanced about .
[S2] he saw nothing ; but looking far off in the sky , he thought that he could discover , at a great distance , a shining track , and the dim figures of two who were vanishing in heaven .
[S3] when his brother returned , bokwewa related all to him exactly as it had happened .
[S4] the face of kwasynd changed , and was dark as the night .
[S5] for several days he would not taste food .
[S6] sometimes he would fall to weeping for a long time , and now only it seemed that he remembered how gen
... [truncated]
```

**Confirmed:** No target difficulty appears in any judge prompt. Blind evaluation is correct.

## 12. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 5 |
| Direct | not answerable | 2 |
| Direct | gen error: degenerate output | 2 |
| Direct | gen error: empty | 1 |
| ICL | gen error: degenerate output | 4 |
| ICL | not answerable | 2 |
| ICL | wrong answer | 2 |
| ICL | other | 1 |
| ICL | gen error: no question mark | 1 |
| SelfRefine | gen error: degenerate initial output | 8 |
| SelfRefine | wrong answer | 2 |
| SelfRefine | not answerable | 2 |
| SelfRefine | gen error: initial generation failed | 1 |
| Ours | gen error: degenerate output | 4 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch; graph_policy non-compliant (answer_only) | 1 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1-2 sentences; focus mismatch; graph_policy non-compliant (multi_node_chain) | 1 |
| Ours | gen error: parse failure | 1 |
| Ours | gen error: grammar: word repetition: did | 1 |
| Ours | gen error: question length out of range | 1 |
| Ours | gen error: does not end with ? | 1 |
| Ours | not fluent | 1 |

## 14. Retry & Budget Diagnostics

### 14a. Attempts per Method

| Method | Avg attempts | Max attempts | Total |
|---|---:|---:|---:|
| Direct | 2.00 | 3 | 30 |
| ICL | 2.10 | 3 | 30 |
| SelfRefine | 2.40 | 3 | 30 |
| Ours | 3.00 | 4 | 30 |

### 14b. Attempt Distribution by Method

| Method | 1 attempt | 2 attempts | 3+ attempts |
|---|---:|---:|---:|
| Direct | 10 | 10 | 10 |
| ICL | 9 | 9 | 12 |
| SelfRefine | 9 | 0 | 21 |
| Ours | 6 | 4 | 20 |

### 14c. Ours Repair Prompt Usage

| Metric | Value |
|---|---|
| Repair prompt used | 11/30 (36.7%) |
| Repair success | 3 |

### 14d. Ours Graph Policy Self-Check Failure Rate

| Self-check failures | 8/30 (26.7%) |

### 14e. Failure Reason Distribution by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 5 |
| Direct | not answerable | 2 |
| Direct | degenerate output | 2 |
| Direct | empty output | 1 |
| ICL | degenerate output | 4 |
| ICL | not answerable | 3 |
| ICL | wrong answer | 2 |
| ICL | unknown | 1 |
| SelfRefine | degenerate output | 8 |
| SelfRefine | wrong answer | 2 |
| SelfRefine | not answerable | 2 |
| SelfRefine | initial generation failed | 1 |
| Ours | degenerate output | 4 |
| Ours | graph issue | 2 |
| Ours | not fluent | 2 |
| Ours | parse failure | 1 |
| Ours | not answerable | 1 |
| Ours | question length | 1 |

### 14f. Ours Retry Reason Distribution (from attempt traces)

| Retry reason | Count |
|---|---:|
| ok | 41 |
| degenerate output | 27 |
| parse failure | 5 |
| does not end with ? | 3 |
| no question mark | 3 |
| grammar: word repetition: what | 3 |
| empty | 2 |
| grammar: word repetition: did | 2 |
| question length out of range | 2 |
| grammar: word repetition: abandon | 1 |

## 15. Similarity Diagnostics (diagnostic only, no filtering)

### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)

| Method | E-M mean | M-H mean | E-H mean | n pairs |
|---|---:|---:|---:|---:|
| Direct | 0.124 | 0.106 | 0.072 | 8 |
| ICL | 0.095 | 0.093 | 0.084 | 8 |
| SelfRefine | 0.208 | 0.092 | 0.082 | 4 |
| Ours | 0.053 | 0.086 | 0.072 | 10 |

### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)

| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |
|---|---:|---:|---:|
| Direct | 0.083 | 0.083 | 0.067 |
| ICL | 0.083 | 0.025 | 0.000 |
| SelfRefine | 0.033 | 0.000 | 0.033 |
| Ours | 0.050 | 0.100 | 0.000 |

### 15c. Difficulty Collapse Counts by Method

| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |
|---|---:|---:|---:|---:|---:|
| Direct | 4 | 2 | 2 | 0 | 0 |
| ICL | 4 | 1 | 1 | 0 | 0 |
| SelfRefine | 3 | 2 | 2 | 0 | 0 |
| Ours | 2 | 0 | 0 | 0 | 0 |

## 16. Stage 2 Focus & Difficulty Realization Diagnostics (Ours)

### 16a. Focus Distribution by Target Difficulty (Ours, quality-pass)

| Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| chain_explanation | 0 | 0 | 6 |
| direct_answer | 7 | 0 | 0 |
| relation_question | 0 | 6 | 0 |

### 16b. Node-Level Focus Distribution (pre-override, for comparison)

| Node Focus | Easy (n) | Medium (n) | Hard (n) |
|---|---:|---:|---:|
| bridge | 3 | 2 | 1 |
| motivation | 3 | 1 | 3 |
| outcome | 1 | 2 | 0 |
| state | 0 | 1 | 2 |

### 16c. Easy answer_sentence_alone=yes by Focus Type (Ours, quality-pass, target=Easy)

| Focus | Total | ASA=yes | Rate |
|---|---:|---:|---:|
| direct_answer | 7 | 1 | 14.3% |

### 16d. Hard answer_sentence_alone=no by Focus Type (Ours, quality-pass, target=Hard)

| Focus | Total | ASA=no | Rate |
|---|---:|---:|---:|
| chain_explanation | 6 | 4 | 66.7% |

### 16e. Graph Policy Compliance by Focus Type (Ours, quality-pass)

| Focus | Total | GPC=yes | Rate |
|---|---:|---:|---:|
| chain_explanation | 6 | 4 | 66.7% |
| direct_answer | 7 | 7 | 100.0% |
| relation_question | 6 | 6 | 100.0% |

### 16f. Repair Usage by Target Difficulty (Ours)

| Difficulty | Total | Repair Used | Repair Success | Repair Rate |
|---|---:|---:|---:|---:|
| Easy | 10 | 3 | 1 | 30.0% |
| Medium | 10 | 4 | 1 | 40.0% |
| Hard | 10 | 4 | 1 | 40.0% |

### 16g. Top 10 Easy Failures (Ours, quality-pass, predicted != Easy)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | prince-featherhead-and-the-pri | What did the consequence of the king and queen giving away a | king bruin heard the king had  | Medium | direct_answer | partial | bridge_detected |
| 2 | the-little-spirit-or-boy-man | What does the mother believe about the boy based on his acti | the boy had powers . | Medium | direct_answer | partial | bridge_detected |
| 3 | the-winter-spirit-and-his-visi | What did the young visitor, seegwun, place in the wreath upo | as his first trophy in the nor | Medium | direct_answer | partial | bridge_detected |
| 4 | white-hare-and-crocodiles | How did the crocodile feel when the hare finished counting a | angry . | Hard | direct_answer | no | focus_mismatch_causal_wording |
| 5 | jamie-freel-and-the-young-lady | What was Jamie's feeling about joining the fairy ride to Dub | he was thirsting for adventure | Medium | direct_answer | no | multi_sentence_required |
| 6 | morraha | What did the woman tell Morraha he would do if he went to pl | he would lose more . | Medium | direct_answer | no | multi_sentence_required |

### 16h. Top 10 Hard Failures (Ours, quality-pass, predicted != Hard)

| # | Story | Question | Answer | Pred | Focus | ASA | Failure Reason |
|---|---|---|---|---|---|---|---|
| 1 | bokwewa-the-humpback | What motivated Bokwewa to give such detailed advice to Kwasy | wanted kwasynd to succeed . | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 2 | habetrot-the-spinstress | What motivated maisie's mother to feel disappointed in her d | maisie was not a good spinster | Medium | chain_explanation | partial | focus_mismatch_direct_wording |
| 3 | morraha | What motivated the man to hide in the top of the parlour chi | the bellman said everyone who  | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 4 | mount-of-golden-queen | What motivated the eagle to say 'yes' and agree to take the  | the giant asked him to . | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 5 | the-winter-spirit-and-his-visi | What motivated the old man to share his stories with the you | the old man was no longer alon | Medium | chain_explanation | no | focus_mismatch_direct_wording |
| 6 | white-hare-and-crocodiles | What led the crocodile to believe there are more crocodiles  | there are many crocodiles arou | Medium | chain_explanation | partial | focus_mismatch_direct_wording |

## 17. Examples

### Best Easy examples (quality-pass, correct prediction)

**ICL Example:**
- Story: mount-of-golden-queen
- Question: How did the princess feel when she read the letter?
- Target answer: happy .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: mount-of-golden-queen
- Question: How did the princess feel when she read the letter?
- Target answer: happy .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: mount-of-golden-queen
- Question: How did the princess react when she read the letter?
- Target answer: happy .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: habetrot-the-spinstress
- Question: Why did Maisie try not to think of the sunshine outside and sit down soberly with her distaff?
- Target answer: she did not want to go to the nunnery .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: habetrot-the-spinstress
- Question: Why did Maisie try to spin the lint into thread despite her lack of experience?
- Target answer: she did not want to go to the nunnery .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: morraha
- Question: Why did Morraha go against his wife's advice and play cards with the young man again?
- Target answer: he did not know how to meet the young man 's demands .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: habetrot-the-spinstress
- Question: Why was Maisie motivated to sit at her spinning-wheel?
- Target answer: she did not want to go to the nunnery .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**Direct Example:**
- Story: white-hare-and-crocodiles
- Question: Why did the crocodile claim that hares would be as nothing compared to crocodiles?
- Target answer: there are many crocodiles around the world .
- Target: Hard, Predicted: Hard

**ICL Example:**
- Story: the-little-spirit-or-boy-man
- Question: Why did the boy-man decide to run off with the trout and make the brothers think it was the fish moving on its own?
- Target answer: he wanted to play a trick on the four men .
- Target: Hard, Predicted: Hard

