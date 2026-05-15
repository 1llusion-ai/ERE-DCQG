# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-12 17:55:55

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Selection mode | story_matched |
| Target per level | 5 |
| Selected Easy | 5 |
| Selected Medium | 5 |
| Selected Hard | 5 |
| Total selected | 15 |
| Total stories | 5 |
| Total generations | 60 |
| Candidates per level per story | 1 |
| Max stories | 5 |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 4 | 5 | 80.0% |
| Medium | 5 | 5 | 100.0% |
| Hard | 5 | 5 | 100.0% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 15 | 15 | 100.0% |
| ICL | 13 | 15 | 86.7% |
| SelfRefine | 8 | 15 | 53.3% |
| Ours | 13 | 15 | 86.7% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 5 | 5 | 2 | 12 | 15 | 80.0% |
| ICL | 3 | 3 | 2 | 8 | 15 | 53.3% |
| SelfRefine | 2 | 2 | 1 | 5 | 15 | 33.3% |
| Ours | 4 | 4 | 3 | 11 | 15 | 73.3% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 15 | 0 | 15 | 0.0% |
| ICL | 12 | 3 | 15 | 20.0% |
| SelfRefine | 8 | 7 | 15 | 46.7% |
| Ours | 13 | 2 | 15 | 13.3% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 58.3% | 7/12 (58.3%, 95%CI [32.0, 80.7%]) | 46.7% |
| ICL | 50.0% | 4/8 (50.0%, 95%CI [21.5, 78.5%]) | 44.4% |
| SelfRefine | 60.0% | 3/5 (60.0%, 95%CI [23.1, 88.2%]) | 50.0% |
| Ours | 54.5% | 6/11 (54.5%, 95%CI [28.0, 78.7%]) | 50.0% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 3 | 2 | 0 |
| Medium | 1 | 4 | 0 |
| Hard | 0 | 2 | 0 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 1 | 1 | 1 |
| Medium | 0 | 3 | 0 |
| Hard | 0 | 2 | 0 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 1 | 1 | 0 |
| Medium | 0 | 2 | 0 |
| Hard | 1 | 0 | 0 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 3 | 1 | 0 |
| Medium | 1 | 3 | 0 |
| Hard | 0 | 3 | 0 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 3/5 (60.0%, 95%CI [23.1, 88.2%]) | 4/5 (80.0%, 95%CI [37.6, 96.4%]) | 0/2 (0.0%, 95%CI [0.0, 65.8%]) |
| ICL | 1/3 (33.3%, 95%CI [6.1, 79.2%]) | 3/3 (100.0%, 95%CI [43.8, 100.0%]) | 0/2 (0.0%, 95%CI [0.0, 65.8%]) |
| SelfRefine | 1/2 (50.0%, 95%CI [9.5, 90.5%]) | 2/2 (100.0%, 95%CI [34.2, 100.0%]) | 0/1 (0.0%, 95%CI [0.0, 79.3%]) |
| Ours | 3/4 (75.0%, 95%CI [30.1, 95.4%]) | 3/4 (75.0%, 95%CI [30.1, 95.4%]) | 0/3 (0.0%, 95%CI [0.0, 56.2%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 42.7% | 66.7% | 61.5% | 0.0% |
| ICL | 38.9% | 50.0% | 66.7% | 0.0% |
| SelfRefine | 43.3% | 50.0% | 80.0% | 0.0% |
| Ours | 43.2% | 75.0% | 54.5% | 0.0% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.498 | 12 |
| ICL | 0.000 | 8 |
| SelfRefine | -0.152 | 5 |
| Ours | 0.633 | 11 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 5 | 3 | 60.0% | 3 | 2 | 0 |
| ICL | 3 | 1 | 33.3% | 1 | 1 | 1 |
| SelfRefine | 2 | 1 | 50.0% | 1 | 1 | 0 |
| Ours | 4 | 3 | 75.0% | 3 | 1 | 0 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 5 | 4 | 80.0% | 1 | 4 | 0 |
| ICL | 3 | 3 | 100.0% | 0 | 3 | 0 |
| SelfRefine | 2 | 2 | 100.0% | 0 | 2 | 0 |
| Ours | 4 | 3 | 75.0% | 1 | 3 | 0 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 2 | 0 | 0.0% | 0 | 2 | 0 |
| ICL | 2 | 0 | 0.0% | 0 | 2 | 0 |
| SelfRefine | 1 | 0 | 0.0% | 1 | 0 | 0 |
| Ours | 3 | 0 | 0.0% | 0 | 3 | 0 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 0/2 (0.0%, 95%CI [0.0, 65.8%]) | |
| ICL | 0/2 (0.0%, 95%CI [0.0, 65.8%]) | |
| SelfRefine | 0/1 (0.0%, 95%CI [0.0, 79.3%]) | |
| Ours | 0/3 (0.0%, 95%CI [0.0, 56.2%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/2 (0.0%, 95%CI [0.0, 65.8%]) | |
| ICL | 0/2 (0.0%, 95%CI [0.0, 65.8%]) | |
| SelfRefine | 0/1 (0.0%, 95%CI [0.0, 79.3%]) | |
| Ours | 0/3 (0.0%, 95%CI [0.0, 56.2%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/2 (0.0%, 95%CI [0.0, 65.8%]) | |
| ICL | 0/2 (0.0%, 95%CI [0.0, 65.8%]) | |
| SelfRefine | 0/1 (0.0%, 95%CI [0.0, 79.3%]) | |
| Ours | 0/3 (0.0%, 95%CI [0.0, 56.2%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 0 | 0 |
| ICL | 0 | 0 |
| SelfRefine | 0 | 0 |
| Ours | 0 | 0 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 4 | 0 | 0 | 0 | 4 |
| Medium | 0 | 5 | 0 | 0 | 5 |
| Hard | 0 | 0 | 5 | 0 | 5 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 3 | 4 | 75.0% |
| Medium | 5 | 5 | 100.0% |
| Hard | 4 | 5 | 80.0% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 4 | 75.0% | N/A |
| two_node_relation | 4 | 75.0% | N/A |
| multi_node_chain | 3 | 0.0% | 0.0% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 4 | 28.6% |
| causes | 3 | 21.4% |
| motivates | 2 | 14.3% |
| causes → motivates | 1 | 7.1% |
| temporal_before → temporal_before → causes → motivates | 1 | 7.1% |
| motivates → causes | 1 | 7.1% |
| causes → causes | 1 | 7.1% |
| results_in → temporal_before | 1 | 7.1% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| temporal_before | 0 | 3 | 0 | 3 |
| causes | 0 | 2 | 0 | 2 |
| motivates | 0 | 2 | 0 | 2 |
| results_in | 0 | 1 | 0 | 1 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 2 | 4 | 50.0% |
| Medium | two_node_relation | 3 | 5 | 60.0% |
| Hard | multi_node_chain | 4 | 5 | 80.0% |

### GP-7. Hard Pure Temporal Chain Count: 0

### GP-8. Easy Answer-Sentence-Alone Rate: 75.0% (3/4)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 0.0% (0/3)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 73.3% (11/15) | 80.0% (12/15) | 53.3% (8/15) | 33.3% (5/15) | -6.7pp | +20.0pp | +40.0pp |
| overall_accuracy | 54.5% | 58.3% | 50.0% | 60.0% | -3.8pp | +4.5pp | -5.5pp |
| macro_accuracy | 50.0% | 46.7% | 44.4% | 50.0% | +3.3pp | +5.6pp | +0.0pp |
| macro_f1 | 43.2% | 42.7% | 38.9% | 43.3% | +0.4pp | +4.3pp | -0.2pp |
| spearman | 0.633 | 0.498 | 0.000 | -0.152 | +0.135 | +0.633 | +0.785 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected stories >= 100 | FAIL | selected stories=5 |
| every story has equal Easy/Med/Hard count | PASS | all 5 stories have 1E/1M/1H |
| all methods have identical denominator | PASS | denominators: {'Direct': 15, 'ICL': 15, 'SelfRefine': 15, 'Ours': 15} |
| Ours quality_pass not << baselines | PASS | Ours=11/15 |
| Ours overall accuracy >= each baseline | FAIL | Ours=54.5% |
| Ours macro accuracy >= each baseline | PASS | Ours=50.0% |
| Ours Spearman >= each baseline | PASS | Ours=0.633 |

**Overall: SOME CRITERIA FAILED**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 60.0% | 60.0% | +0.00pp | [-30.00pp, +30.00pp] | 1.0000 | 10 | no |
| Direct | macro_accuracy | 50.0% | 50.0% | +0.00pp | [-26.67pp, +26.67pp] | 1.0000 | 10 | no |
| Direct | macro_f1 | 45.0% | 45.0% | +0.00pp | [-22.31pp, +23.30pp] | 1.0000 | 10 | no |
| Direct | spearman | 0.609 | 0.609 | +0.000 | [-0.378, +0.382] | 1.0000 | 10 | no |
| ICL | overall_accuracy | 42.9% | 57.1% | -14.29pp | [-42.86pp, +0.00pp] | 0.6878 | 7 | no |
| ICL | macro_accuracy | 38.9% | 50.0% | -11.11pp | [-33.33pp, +0.00pp] | 0.6878 | 7 | no |
| ICL | macro_f1 | 33.3% | 44.4% | -11.11pp | [-31.11pp, +0.00pp] | 0.6878 | 7 | no |
| ICL | spearman | 0.418 | 0.540 | -0.122 | [-0.554, +0.529] | 1.0000 | 7 | no |
| SelfRefine | overall_accuracy | 100.0% | 100.0% | +0.00pp | [+0.00pp, +0.00pp] | 1.0000 | 2 | no |
| SelfRefine | macro_accuracy | 66.7% | 66.7% | +0.00pp | [+0.00pp, +0.00pp] | 1.0000 | 2 | no |
| SelfRefine | macro_f1 | 66.7% | 66.7% | +0.00pp | [+0.00pp, +0.00pp] | 1.0000 | 2 | no |
| SelfRefine | spearman | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 2 | no |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 60.0% (3/5) | 80.0% (4/5) | 0.0% (0/5) | 46.7% (7/15) | 15 |
| ICL | 20.0% (1/5) | 60.0% (3/5) | 0.0% (0/5) | 26.7% (4/15) | 15 |
| SelfRefine | 20.0% (1/5) | 40.0% (2/5) | 0.0% (0/5) | 20.0% (3/15) | 15 |
| Ours | 60.0% (3/5) | 60.0% (3/5) | 0.0% (0/5) | 40.0% (6/15) | 15 |

## 11. Difficulty Judge Prompt Audit

The difficulty judge is **blind**: it sees only the story, generated question, and expected answer.
It does NOT see the target difficulty. Below are 3 sample prompts to confirm.

**Sample 1** (story=cuchulain-of-muirthemne, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] and when the dark night was coming on , conchubar said to his people , " it is best for us to unyoke the chariots now , and to look for some place where we can spend the night .
[S1] " then fergus went forward to look for some place , and what he came to was a very small poor - looking house .
[S2] a man and a woman were in it , and when they saw him they said , " bring your companions here along with you , and they will be welcome .
[S3] " fergus went back to his companions and told them what he had seen .
[S4] but bricriu said : " where is the use of going into a house like that , with neither room nor provisions nor coverings in it .
[S5] it is not worth our while to be going there .
[S6] " then 
... [truncated]
```

**Sample 2** (story=cuchulain-of-muirthemne, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] and when the dark night was coming on , conchubar said to his people , " it is best for us to unyoke the chariots now , and to look for some place where we can spend the night .
[S1] " then fergus went forward to look for some place , and what he came to was a very small poor - looking house .
[S2] a man and a woman were in it , and when they saw him they said , " bring your companions here along with you , and they will be welcome .
[S3] " fergus went back to his companions and told them what he had seen .
[S4] but bricriu said : " where is the use of going into a house like that , with neither room nor provisions nor coverings in it .
[S5] it is not worth our while to be going there .
[S6] " then 
... [truncated]
```

**Sample 3** (story=cuchulain-of-muirthemne, target=Medium):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] " try and know them again , " said the man , " for the fifty young girls are in this house , and this woman beside me is their mistress , dechtire .
[S1] it was they themselves , changed into birds , that went to emain macha to bring you here .
[S2] " then dechtire gave bricriu a purple cloak with gold fringes .
[S3] he went back to find his companions .
[S4] but while he was going he thought to himself , " conchubar would give great treasure to find these fifty young girls again , and his sister along with them .
[S5] i will not tell him i have found them .
[S6] i will only say i have found a house with beautiful women in it , and no more than that .
[S7] " when conchubar saw bricriu he asked news 
... [truncated]
```

**Confirmed:** No target difficulty appears in any judge prompt. Blind evaluation is correct.

## 12. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 2 |
| Direct | other | 1 |
| ICL | gen error: empty | 2 |
| ICL | other | 2 |
| ICL | gen error: degenerate output | 1 |
| ICL | wrong answer | 1 |
| ICL | not answerable | 1 |
| SelfRefine | gen error: degenerate initial output | 5 |
| SelfRefine | gen error: initial generation failed | 2 |
| SelfRefine | wrong answer | 1 |
| SelfRefine | not fluent | 1 |
| SelfRefine | gen error: no question mark | 1 |
| Ours | gen error: degenerate output | 2 |
| Ours | gen error: does not end with ? | 1 |
| Ours | gen error: graph_invalid | 1 |

## 13. Story-Matched Diagnostics

### 13a. Story Summary

| Selected stories | 5 |
| Candidates per story | 1 × 3 levels |

**Equal Easy/Med/Hard per story:** YES (expected 4 per level per story)

### 13b. Story-Level Average Accuracy by Method (quality-pass, judge-ok)

| Method | Mean story acc | Median story acc | Std | N stories |
|---|---:|---:|---:|---:|
| Direct | 56.7% | 66.7% | 32.7% | 4 |
| ICL | 33.3% | 50.0% | 27.9% | 3 |
| SelfRefine | 46.7% | 100.0% | 45.2% | 3 |
| Ours | 66.7% | 66.7% | 29.8% | 5 |

### 13c. Story-Level Win/Tie/Loss (Ours vs Baseline)

| Baseline | Ours Wins | Ties | Ours Losses | N stories |
|---|---:|---:|---:|---:|
| Direct | 2 | 2 | 1 | 5 |
| ICL | 3 | 0 | 2 | 5 |
| SelfRefine | 3 | 1 | 1 | 5 |

### 13d. Story-Level Spearman (stories with all 3 levels valid)

| Method | Mean story rho | N valid stories | N skipped |
|---|---:|---:|---:|
| Direct | 0.866 | 2 | 3 |
| ICL | 0.866 | 1 | 4 |
| SelfRefine | -0.866 | 1 | 4 |
| Ours | 0.577 | 3 | 2 |

### 13e. Per-Story Failure Counts by Method

| Method | Stories with 0 fails | 1 fail | 2 fails | 3 fails |
|---|---:|---:|---:|---:|
| Direct | 2 | 3 | 0 | 0 |
| ICL | 1 | 2 | 1 | 1 |
| SelfRefine | 1 | 0 | 2 | 2 |
| Ours | 3 | 0 | 2 | 0 |

## 14. Retry & Budget Diagnostics

### 14a. Attempts per Method

| Method | Avg attempts | Max attempts | Total |
|---|---:|---:|---:|
| Direct | 1.93 | 3 | 15 |
| ICL | 1.60 | 3 | 15 |
| SelfRefine | 2.07 | 3 | 15 |
| Ours | 3.00 | 4 | 15 |

### 14b. Attempt Distribution by Method

| Method | 1 attempt | 2 attempts | 3+ attempts |
|---|---:|---:|---:|
| Direct | 5 | 6 | 4 |
| ICL | 9 | 3 | 3 |
| SelfRefine | 7 | 0 | 8 |
| Ours | 2 | 4 | 9 |

### 14c. Ours Repair Prompt Usage

| Metric | Value |
|---|---|
| Repair prompt used | 9/15 (60.0%) |
| Repair success | 0 |

### 14d. Ours Graph Policy Self-Check Failure Rate

| Self-check failures | 2/14 (14.3%) |

### 14e. Failure Reason Distribution by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 2 |
| Direct | unknown | 1 |
| ICL | empty output | 2 |
| ICL | unknown | 2 |
| ICL | degenerate output | 1 |
| ICL | wrong answer | 1 |
| ICL | not answerable | 1 |
| SelfRefine | degenerate output | 5 |
| SelfRefine | initial generation failed | 2 |
| SelfRefine | wrong answer | 1 |
| SelfRefine | not fluent | 1 |
| SelfRefine | not answerable | 1 |
| Ours | degenerate output | 2 |
| Ours | not answerable | 1 |
| Ours | graph issue | 1 |

### 14f. Ours Retry Reason Distribution (from attempt traces)

| Retry reason | Count |
|---|---:|
| degenerate output | 25 |
| ok | 16 |
| no question mark | 2 |
| does not end with ? | 1 |
| grammar: word repetition: how | 1 |

## 15. Similarity Diagnostics (diagnostic only, no filtering)

### 15a. Per-Method Average Question Lexical Similarity (char 4-gram Jaccard)

| Method | E-M mean | M-H mean | E-H mean | n pairs |
|---|---:|---:|---:|---:|
| Direct | 0.083 | 0.082 | 0.121 | 5 |
| ICL | 0.050 | 0.054 | 0.110 | 3 |
| SelfRefine | 0.060 | 0.108 | 0.237 | 2 |
| Ours | 0.134 | 0.040 | 0.065 | 4 |

### 15b. Evidence Sentence Overlap by Method (Jaccard of judge-used evidence)

| Method | E-M evidence overlap | M-H evidence overlap | E-H evidence overlap |
|---|---:|---:|---:|
| Direct | 0.200 | 0.100 | 0.067 |
| ICL | 0.167 | 0.133 | 0.107 |
| SelfRefine | 0.100 | 0.000 | 0.000 |
| Ours | 0.067 | 0.200 | 0.100 |

### 15c. Difficulty Collapse Counts by Method

| Method | Stories w/ 3 QP | All 3 same pred | Collapse to Medium | Collapse to Easy | Collapse to Hard |
|---|---:|---:|---:|---:|---:|
| Direct | 2 | 0 | 0 | 0 | 0 |
| ICL | 1 | 0 | 0 | 0 | 0 |
| SelfRefine | 1 | 0 | 0 | 0 | 0 |
| Ours | 3 | 1 | 1 | 0 | 0 |

## 16. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: cuchulain-of-muirthemne
- Question: When did Conchubar want to unyoke the chariots?
- Target answer: it was night .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: weendigoes-and-the-bone-dwarf
- Question: What was the hunter's companion covered in when he returned in the morning?
- Target answer: the weendigo was smeared in blood .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: weendigoes-and-the-bone-dwarf
- Question: How was the weendigo when he returned in the morning?
- Target answer: the weendigo was smeared in blood .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: cuchulain-of-muirthemne
- Question: When did Conchubar want to unyoke the chariots?
- Target answer: it was night .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: cuchulain-of-muirthemne
- Question: Why did Bricriu decide not to tell Conchubar he had found the fifty young girls?
- Target answer: he wanted to get money out of conchubar .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: goblin-huckster
- Question: What transformation did the book create in the room?
- Target answer: it made such a glorious tree of light .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: cuchulain-of-muirthemne
- Question: Why did Bricriu decide not to tell Conchubar he had found the fifty young girls?
- Target answer: he wanted to get money out of conchubar .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: goblin-huckster
- Question: What did the student's book do that amazed the goblin?
- Target answer: it made such a glorious tree of light .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

