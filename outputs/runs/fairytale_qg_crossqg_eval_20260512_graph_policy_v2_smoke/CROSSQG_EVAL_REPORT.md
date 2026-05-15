# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-12 11:50:46

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Target per level | 10 |
| Selected Easy | 10 |
| Selected Medium | 10 |
| Selected Hard | 10 |
| Total selected | 30 |
| Total generations | 120 |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 9 | 10 | 90.0% |
| Medium | 9 | 10 | 90.0% |
| Hard | 9 | 10 | 90.0% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 29 | 30 | 96.7% |
| ICL | 27 | 30 | 90.0% |
| SelfRefine | 19 | 30 | 63.3% |
| Ours | 27 | 30 | 90.0% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 7 | 7 | 5 | 19 | 30 | 63.3% |
| ICL | 6 | 5 | 6 | 17 | 30 | 56.7% |
| SelfRefine | 5 | 2 | 6 | 13 | 30 | 43.3% |
| Ours | 7 | 6 | 7 | 20 | 30 | 66.7% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 29 | 1 | 30 | 3.3% |
| ICL | 27 | 3 | 30 | 10.0% |
| SelfRefine | 19 | 11 | 30 | 36.7% |
| Ours | 27 | 3 | 30 | 10.0% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 68.4% | 13/19 (68.4%, 95%CI [46.0, 84.6%]) | 65.7% |
| ICL | 58.8% | 10/17 (58.8%, 95%CI [36.0, 78.4%]) | 61.1% |
| SelfRefine | 69.2% | 9/13 (69.2%, 95%CI [42.4, 87.3%]) | 74.4% |
| Ours | 60.0% | 12/20 (60.0%, 95%CI [38.7, 78.1%]) | 61.9% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 3 | 0 |
| Medium | 0 | 7 | 0 |
| Hard | 1 | 2 | 2 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 2 | 0 |
| Medium | 0 | 5 | 0 |
| Hard | 2 | 3 | 1 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 2 | 3 | 0 |
| Medium | 0 | 2 | 0 |
| Hard | 0 | 1 | 5 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 4 | 3 | 0 |
| Medium | 0 | 6 | 0 |
| Hard | 1 | 4 | 2 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 4/7 (57.1%, 95%CI [25.0, 84.2%]) | 7/7 (100.0%, 95%CI [64.6, 100.0%]) | 2/5 (40.0%, 95%CI [11.8, 76.9%]) |
| ICL | 4/6 (66.7%, 95%CI [30.0, 90.3%]) | 5/5 (100.0%, 95%CI [56.6, 100.0%]) | 1/6 (16.7%, 95%CI [3.0, 56.4%]) |
| SelfRefine | 2/5 (40.0%, 95%CI [11.8, 76.9%]) | 2/2 (100.0%, 95%CI [34.2, 100.0%]) | 5/6 (83.3%, 95%CI [43.6, 97.0%]) |
| Ours | 4/7 (57.1%, 95%CI [25.0, 84.2%]) | 6/6 (100.0%, 95%CI [61.0, 100.0%]) | 2/7 (28.6%, 95%CI [8.2, 64.1%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 65.8% | 66.7% | 73.7% | 57.1% |
| ICL | 54.0% | 66.7% | 66.7% | 28.6% |
| SelfRefine | 66.0% | 57.1% | 50.0% | 90.9% |
| Ours | 58.1% | 66.7% | 63.2% | 44.4% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.529 | 19 |
| ICL | 0.354 | 17 |
| SelfRefine | 0.833 | 13 |
| Ours | 0.516 | 20 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 7 | 4 | 57.1% | 4 | 3 | 0 |
| ICL | 6 | 4 | 66.7% | 4 | 2 | 0 |
| SelfRefine | 5 | 2 | 40.0% | 2 | 3 | 0 |
| Ours | 7 | 4 | 57.1% | 4 | 3 | 0 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 7 | 7 | 100.0% | 0 | 7 | 0 |
| ICL | 5 | 5 | 100.0% | 0 | 5 | 0 |
| SelfRefine | 2 | 2 | 100.0% | 0 | 2 | 0 |
| Ours | 6 | 6 | 100.0% | 0 | 6 | 0 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 5 | 2 | 40.0% | 1 | 2 | 2 |
| ICL | 6 | 1 | 16.7% | 2 | 3 | 1 |
| SelfRefine | 6 | 5 | 83.3% | 0 | 1 | 5 |
| Ours | 7 | 2 | 28.6% | 1 | 4 | 2 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 2/5 (40.0%, 95%CI [11.8, 76.9%]) | |
| ICL | 1/6 (16.7%, 95%CI [3.0, 56.4%]) | |
| SelfRefine | 5/6 (83.3%, 95%CI [43.6, 97.0%]) | |
| Ours | 2/7 (28.6%, 95%CI [8.2, 64.1%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 2/5 (40.0%, 95%CI [11.8, 76.9%]) | |
| ICL | 1/6 (16.7%, 95%CI [3.0, 56.4%]) | |
| SelfRefine | 5/6 (83.3%, 95%CI [43.6, 97.0%]) | |
| Ours | 2/7 (28.6%, 95%CI [8.2, 64.1%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/5 (0.0%, 95%CI [0.0, 43.4%]) | |
| ICL | 0/6 (0.0%, 95%CI [0.0, 39.0%]) | |
| SelfRefine | 0/6 (0.0%, 95%CI [0.0, 39.0%]) | |
| Ours | 0/7 (0.0%, 95%CI [0.0, 35.4%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 2 | 2 |
| ICL | 1 | 1 |
| SelfRefine | 5 | 5 |
| Ours | 2 | 2 |

## Ours Graph Policy Diagnostics

### GP-1. Graph Policy Distribution by Target Difficulty

| Target | answer_only | two_node_relation | multi_node_chain | fallback | total |
|---|---:|---:|---:|---:|---:|
| Easy | 9 | 0 | 0 | 0 | 9 |
| Medium | 0 | 9 | 0 | 0 | 9 |
| Hard | 0 | 0 | 9 | 0 | 9 |

### GP-2. Graph Policy Compliance Rate by Target Difficulty

| Target | compliant | total | pct |
|---|---:|---:|---:|
| Easy | 6 | 9 | 66.7% |
| Medium | 9 | 9 | 100.0% |
| Hard | 7 | 9 | 77.8% |

### GP-3. Per-Policy Difficulty Accuracy

| Policy | n_valid | accuracy | hard_hit |
|---|---:|---:|---:|
| answer_only | 7 | 57.1% | N/A |
| two_node_relation | 6 | 100.0% | N/A |
| multi_node_chain | 7 | 28.6% | 28.6% |
| other | 0 | N/A | N/A |

### GP-4. Selected Relation Chain Distribution (top 10)

| Relation chain | count | pct |
|---|---:|---:|
| (none) | 9 | 33.3% |
| causes | 4 | 14.8% |
| temporal_before → causes | 3 | 11.1% |
| temporal_before | 2 | 7.4% |
| results_in → causes | 2 | 7.4% |
| motivates | 1 | 3.7% |
| explains | 1 | 3.7% |
| supports_inference | 1 | 3.7% |
| causes → results_in | 1 | 3.7% |
| temporal_before → temporal_before | 1 | 3.7% |

### GP-5. Hard: Relation Type vs Predicted Difficulty

| Relation | Easy | Medium | Hard | total |
|---|---:|---:|---:|---:|
| causes | 0 | 4 | 1 | 5 |
| temporal_before | 0 | 2 | 2 | 4 |
| results_in | 0 | 2 | 1 | 3 |
| explains | 1 | 0 | 0 | 1 |
| supports_inference | 1 | 0 | 0 | 1 |

### GP-6. Repair Prompt Usage by Difficulty and Graph Policy

| Target | Policy | repair_used | total | pct |
|---|---|---:|---:|---:|
| Easy | answer_only | 4 | 9 | 44.4% |
| Medium | two_node_relation | 4 | 9 | 44.4% |
| Hard | multi_node_chain | 4 | 9 | 44.4% |

### GP-7. Hard Pure Temporal Chain Count: 1

### GP-8. Easy Answer-Sentence-Alone Rate: 57.1% (4/7)

### GP-9. Hard Answer-Sentence-Alone=No Rate: 57.1% (4/7)

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 66.7% (20/30) | 63.3% (19/30) | 56.7% (17/30) | 43.3% (13/30) | +3.3pp | +10.0pp | +23.3pp |
| overall_accuracy | 60.0% | 68.4% | 58.8% | 69.2% | -8.4pp | +1.2pp | -9.2pp |
| macro_accuracy | 61.9% | 65.7% | 61.1% | 74.4% | -3.8pp | +0.8pp | -12.5pp |
| macro_f1 | 58.1% | 65.8% | 54.0% | 66.0% | -7.7pp | +4.1pp | -7.9pp |
| spearman | 0.516 | 0.529 | 0.354 | 0.833 | -0.013 | +0.163 | -0.316 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected >= 150 per level | FAIL | Easy=10, Medium=10, Hard=10 |
| Ours quality_pass not << baselines | PASS | Ours=20/30 |
| Ours overall accuracy >= each baseline | FAIL | Ours=60.0% |
| Ours macro accuracy >= each baseline | FAIL | Ours=61.9% |
| Ours Spearman >= each baseline | FAIL | Ours=0.516 |

**Overall: SOME CRITERIA FAILED**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 75.0% | 68.8% | +6.25pp | [-12.50pp, +25.00pp] | 0.7824 | 16 | no |
| Direct | macro_accuracy | 72.2% | 66.7% | +5.56pp | [-14.29pp, +25.00pp] | 0.7824 | 16 | no |
| Direct | macro_f1 | 73.1% | 68.0% | +5.16pp | [-16.67pp, +28.13pp] | 0.7624 | 16 | no |
| Direct | spearman | 0.577 | 0.700 | -0.124 | [-0.616, +0.271] | 0.7186 | 16 | no |
| ICL | overall_accuracy | 57.1% | 64.3% | -7.14pp | [-35.71pp, +21.43pp] | 0.8198 | 14 | no |
| ICL | macro_accuracy | 55.6% | 72.2% | -16.67pp | [-42.22pp, +13.33pp] | 0.3468 | 14 | no |
| ICL | macro_f1 | 52.2% | 60.2% | -7.94pp | [-47.82pp, +26.67pp] | 0.6754 | 14 | no |
| ICL | spearman | 0.338 | 0.393 | -0.055 | [-0.756, +0.656] | 0.8614 | 14 | no |
| SelfRefine | overall_accuracy | 50.0% | 70.0% | -20.00pp | [-60.00pp, +30.00pp] | 0.5236 | 10 | no |
| SelfRefine | macro_accuracy | 63.3% | 76.7% | -13.33pp | [-52.38pp, +25.00pp] | 0.5396 | 10 | no |
| SelfRefine | macro_f1 | 49.2% | 65.2% | -15.98pp | [-61.90pp, +27.41pp] | 0.4636 | 10 | no |
| SelfRefine | spearman | 0.458 | 0.832 | -0.374 | [-0.967, +0.000] | 0.0648 | 10 | no |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 40.0% (4/10) | 70.0% (7/10) | 20.0% (2/10) | 43.3% (13/30) | 30 |
| ICL | 40.0% (4/10) | 50.0% (5/10) | 10.0% (1/10) | 33.3% (10/30) | 30 |
| SelfRefine | 20.0% (2/10) | 20.0% (2/10) | 50.0% (5/10) | 30.0% (9/30) | 30 |
| Ours | 40.0% (4/10) | 60.0% (6/10) | 20.0% (2/10) | 40.0% (12/30) | 30 |

## 11. Difficulty Judge Prompt Audit

The difficulty judge is **blind**: it sees only the story, generated question, and expected answer.
It does NOT see the target difficulty. Below are 3 sample prompts to confirm.

**Sample 1** (story=the-winning-of-olwen, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] she entered , and sat down on a bench beside kilweh , and he spake to her : ' ah , maiden , since first i heard thy name i have loved thee -- will you not come away with me from this evil place ?
[S1] ' ' that i can not do , ' answered she , ' for i have given my word to my father not to go without his knowledge , for his life will only last till i am betrothed .
[S2] whatever is , must be , but this counsel i will give you .
[S3] go , and ask me of my father , and whatsoever he shall required of you grant it , and you shall win me ; but if thou deny him anything thou wilt not obtain me , and it will be well for you if you escape with thy life .
[S4] ' ' all this i promise , ' said he .

Question: "
... [truncated]
```

**Sample 2** (story=the-winning-of-olwen, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] she entered , and sat down on a bench beside kilweh , and he spake to her : ' ah , maiden , since first i heard thy name i have loved thee -- will you not come away with me from this evil place ?
[S1] ' ' that i can not do , ' answered she , ' for i have given my word to my father not to go without his knowledge , for his life will only last till i am betrothed .
[S2] whatever is , must be , but this counsel i will give you .
[S3] go , and ask me of my father , and whatsoever he shall required of you grant it , and you shall win me ; but if thou deny him anything thou wilt not obtain me , and it will be well for you if you escape with thy life .
[S4] ' ' all this i promise , ' said he .

Question: "
... [truncated]
```

**Sample 3** (story=the-bracket-bull, target=Easy):

```
You are a reading-comprehension difficulty judge. Evaluate the question below.

Story:
[S0] when the day came , the bracket bull said , " take the horn off me and eat your enough -- that 's the last luck you have .
[S1] i am to fight with the other bull immediately , and i shall escape from him to - day , but he will have me dead to - morrow by twelve o'clock .
[S2] " himself and the other bull fought that day , and the bracket bull came back in the evening , and he himself and the lad passed that night in the wood .
[S3] when the next day came , the bracket bull said to him , " twist the horn off me and eat your enough -- that 's the last luck you 'll have .
[S4] listen now to the thing that i 'm telling you .
[S5] when you 'll see me dead , go and cut a strip of skin of the back and a st
... [truncated]
```

**Confirmed:** No target difficulty appears in any judge prompt. Blind evaluation is correct.

## 12. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | wrong answer | 4 |
| Direct | not fluent | 3 |
| Direct | not answerable | 2 |
| Direct | gen error: degenerate output | 1 |
| Direct | gen error: no question mark | 1 |
| ICL | not answerable | 5 |
| ICL | gen error: degenerate output | 4 |
| ICL | gen error: no question mark | 1 |
| ICL | not fluent | 1 |
| ICL | wrong answer | 1 |
| ICL | other | 1 |
| SelfRefine | gen error: degenerate initial output | 9 |
| SelfRefine | gen error: initial generation failed | 2 |
| SelfRefine | not fluent | 2 |
| SelfRefine | not answerable | 2 |
| SelfRefine | gen error: no question mark | 1 |
| SelfRefine | wrong answer | 1 |
| Ours | gen error: graph_invalid | 3 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch; graph_policy non-compliant (answer_only) | 2 |
| Ours | gen error: self-check failed: answer mismatch | 2 |
| Ours | gen error: degenerate output | 2 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1-2 sentences; focus mismatch; graph_policy non-compliant (multi_node_chain) | 1 |

## 13. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: how-the-river-gods-wedding-was-broken-off
- Question: How did the sorcerer feel when he begged for mercy?
- Target answer: scared .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: the-bracket-bull
- Question: How did the bracket bull feel when he knew the lad needed to cut a strip of skin off him?
- Target answer: sad .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: money-box
- Question: How did they all feel about the comedy?
- Target answer: amused .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: thomas-the-rhymer
- Question: What was Thomas known for in the country?
- Target answer: he was famous for his prophecies .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: the-three-crowns
- Question: Why did the princess give the prince advice to put a big stone in the basket instead of himself?
- Target answer: she wanted to see if his brothers would do something bad to them .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: the-tale-of-benjamin-bunny
- Question: What did Old Mr. Bunny do after driving the cat into the green-house?
- Target answer: he locked the door and whipped his son benjamin .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: thomas-the-rhymer
- Question: Where were Thomas and the fairy queen when Thomas was asked to speak no single word to anyone save the fairy queen?
- Target answer: they were at their destination .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: the-three-crowns
- Question: What was the princess's motivation for instructing the prince to put a heavy stone in the basket instead of himself?
- Target answer: she wanted to see if his brothers would do something bad to them .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**Direct Example:**
- Story: the-believing-husbands
- Question: Why did the wife wait until she heard the funeral passing the window before telling the man to rise?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

**ICL Example:**
- Story: king-kojata
- Question: Why did the magician and his servants retrace their steps and ask the monk if they saw anyone on horseback?
- Target answer: her father was pursuing them .
- Target: Hard, Predicted: Hard

**SelfRefine Example:**
- Story: the-believing-husbands
- Question: Why did the wife wait until she heard the funeral passing the window before telling the man to rise?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

**Ours Example:**
- Story: the-believing-husbands
- Question: What was the wife's motivation in stopping the husband initially and then urging him to hurry?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

