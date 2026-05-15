# FairytaleQA CrossQG Evaluation Report

Generated: 2026-05-12 10:04:17

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Target per level | 150 |
| Selected Easy | 150 |
| Selected Medium | 150 |
| Selected Hard | 150 |
| Total selected | 450 |
| Total generations | 1800 |

### Graph Extraction Success

| Difficulty | Valid | Total | Pct |
|---|---:|---:|---:|
| Easy | 131 | 150 | 87.3% |
| Medium | 133 | 150 | 88.7% |
| Hard | 142 | 150 | 94.7% |

### Parse Success by Method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 418 | 450 | 92.9% |
| ICL | 406 | 450 | 90.2% |
| SelfRefine | 290 | 450 | 64.4% |
| Ours | 373 | 450 | 82.9% |

## 2. Quality Pass by Method and Difficulty

| Method | Easy QP | Medium QP | Hard QP | Total QP | Total | Pct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 106 | 89 | 81 | 276 | 450 | 61.3% |
| ICL | 94 | 88 | 84 | 266 | 450 | 59.1% |
| SelfRefine | 56 | 58 | 63 | 177 | 450 | 39.3% |
| Ours | 94 | 90 | 89 | 273 | 450 | 60.7% |

## 3. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 410 | 40 | 450 | 8.9% |
| ICL | 398 | 52 | 450 | 11.6% |
| SelfRefine | 290 | 160 | 450 | 35.6% |
| Ours | 368 | 82 | 450 | 18.2% |

## 4. CrossQG Primary Metrics

### 4a. Overall Difficulty Accuracy (quality-pass, judge-ok)

| Method | Accuracy | Wilson 95% CI | Macro Accuracy |
|---|---|---|---|
| Direct | 46.4% | 128/276 (46.4%, 95%CI [40.6, 52.3%]) | 45.7% |
| ICL | 48.5% | 129/266 (48.5%, 95%CI [42.6, 54.5%]) | 48.5% |
| SelfRefine | 49.7% | 88/177 (49.7%, 95%CI [42.4, 57.0%]) | 50.4% |
| Ours | 50.5% | 138/273 (50.5%, 95%CI [44.7, 56.4%]) | 50.6% |

### 4b. Confusion Matrix by Method (quality-pass, judge-ok)

**Direct:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 45 | 59 | 2 |
| Medium | 12 | 71 | 6 |
| Hard | 7 | 62 | 12 |

**ICL:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 34 | 59 | 1 |
| Medium | 11 | 71 | 6 |
| Hard | 6 | 54 | 24 |

**SelfRefine:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 27 | 29 | 0 |
| Medium | 9 | 45 | 4 |
| Hard | 4 | 43 | 16 |

**Ours:**

| Target \ Predicted | Easy | Medium | Hard |
|---|---:|---:|---:|
| Easy | 40 | 54 | 0 |
| Medium | 8 | 74 | 8 |
| Hard | 4 | 61 | 24 |

### 4c. Per-Level Hit Rate by Method (quality-pass, judge-ok)

| Method | Easy hit | Medium hit | Hard hit |
|---|---:|---:|---:|
| Direct | 45/106 (42.5%, 95%CI [33.5, 52.0%]) | 71/89 (79.8%, 95%CI [70.3, 86.8%]) | 12/81 (14.8%, 95%CI [8.7, 24.1%]) |
| ICL | 34/94 (36.2%, 95%CI [27.2, 46.2%]) | 71/88 (80.7%, 95%CI [71.2, 87.6%]) | 24/84 (28.6%, 95%CI [20.0, 39.0%]) |
| SelfRefine | 27/56 (48.2%, 95%CI [35.7, 61.0%]) | 45/58 (77.6%, 95%CI [65.3, 86.4%]) | 16/63 (25.4%, 95%CI [16.3, 37.3%]) |
| Ours | 40/94 (42.6%, 95%CI [33.0, 52.6%]) | 74/90 (82.2%, 95%CI [73.1, 88.8%]) | 24/89 (27.0%, 95%CI [18.8, 37.0%]) |

### 4d. Macro F1 Score by Method (quality-pass, judge-ok)

| Method | Macro F1 | Easy F1 | Medium F1 | Hard F1 |
|---|---:|---:|---:|---:|
| Direct | 42.4% | 52.9% | 50.5% | 23.8% |
| ICL | 46.9% | 46.9% | 52.2% | 41.7% |
| SelfRefine | 48.7% | 56.2% | 51.4% | 38.6% |
| Ours | 49.2% | 54.8% | 53.0% | 39.7% |

### 4e. Spearman Correlation by Method (quality-pass, judge-ok)

| Method | Spearman rho | N |
|---|---:|---:|
| Direct | 0.374 | 276 |
| ICL | 0.418 | 266 |
| SelfRefine | 0.479 | 177 |
| Ours | 0.486 | 273 |

## 5. Per-Level Detailed Metrics (quality-pass, judge-ok)

### Easy Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 106 | 45 | 42.5% | 45 | 59 | 2 |
| ICL | 94 | 34 | 36.2% | 34 | 59 | 1 |
| SelfRefine | 56 | 27 | 48.2% | 27 | 29 | 0 |
| Ours | 94 | 40 | 42.6% | 40 | 54 | 0 |

### Medium Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 89 | 71 | 79.8% | 12 | 71 | 6 |
| ICL | 88 | 71 | 80.7% | 11 | 71 | 6 |
| SelfRefine | 58 | 45 | 77.6% | 9 | 45 | 4 |
| Ours | 90 | 74 | 82.2% | 8 | 74 | 8 |

### Hard Target

| Method | N (QP) | Correct | Accuracy | Predicted Easy | Predicted Medium | Predicted Hard |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 81 | 12 | 14.8% | 7 | 62 | 12 |
| ICL | 84 | 24 | 28.6% | 6 | 54 | 24 |
| SelfRefine | 63 | 16 | 25.4% | 4 | 43 | 16 |
| Ours | 89 | 24 | 27.0% | 4 | 61 | 24 |

## 6. Hard-Only Diagnostics (Secondary)

### 6a. Hard Hit Rate (Hard target, quality-pass, judge-ok)

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 12/81 (14.8%, 95%CI [8.7, 24.1%]) | |
| ICL | 24/84 (28.6%, 95%CI [20.0, 39.0%]) | |
| SelfRefine | 16/63 (25.4%, 95%CI [16.3, 37.3%]) | |
| Ours | 24/89 (27.0%, 95%CI [18.8, 37.0%]) | |

### 6b. HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 11/81 (13.6%, 95%CI [7.8, 22.7%]) | |
| ICL | 24/84 (28.6%, 95%CI [20.0, 39.0%]) | |
| SelfRefine | 15/63 (23.8%, 95%CI [15.0, 35.6%]) | |
| Ours | 23/89 (25.8%, 95%CI [17.9, 35.8%]) | |

### 6c. Strict HRP-v2 by Method (Hard target, quality-pass, judge-ok)

| Method | Strict HRP-v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/81 (0.0%, 95%CI [0.0, 4.5%]) | |
| ICL | 0/84 (0.0%, 95%CI [0.0, 4.4%]) | |
| SelfRefine | 0/63 (0.0%, 95%CI [0.0, 5.7%]) | |
| Ours | 2/89 (2.2%, 95%CI [0.6, 7.8%]) | |

### 6d. Unique Stories Among Predicted Hard (Hard target, quality-pass, judge-ok)

| Method | Unique stories | Hard count |
|---|---:|---:|
| Direct | 12 | 12 |
| ICL | 22 | 24 |
| SelfRefine | 15 | 16 |
| Ours | 23 | 24 |

## 7. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 60.7% (273/450) | 61.3% (276/450) | 59.1% (266/450) | 39.3% (177/450) | -0.7pp | +1.6pp | +21.3pp |
| overall_accuracy | 50.5% | 46.4% | 48.5% | 49.7% | +4.2pp | +2.1pp | +0.8pp |
| macro_accuracy | 50.6% | 45.7% | 48.5% | 50.4% | +4.9pp | +2.1pp | +0.2pp |
| macro_f1 | 49.2% | 42.4% | 46.9% | 48.7% | +6.8pp | +2.2pp | +0.4pp |
| spearman | 0.486 | 0.374 | 0.418 | 0.479 | +0.112 | +0.068 | +0.007 |

## 8. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| selected >= 150 per level | PASS | Easy=150, Medium=150, Hard=150 |
| Ours quality_pass not << baselines | PASS | Ours=273/450 |
| Ours overall accuracy >= each baseline | PASS | Ours=50.5% |
| Ours macro accuracy >= each baseline | PASS | Ours=50.6% |
| Ours Spearman >= each baseline | PASS | Ours=0.486 (min margin=+0.007 vs SelfRefine) |

**Overall: Ours has the highest global overall accuracy, macro accuracy, macro F1, and corrected Spearman. However, paired bootstrap CIs include 0 for all comparisons (Section 9), so statistical significance is not established at n=150 per level. Do not claim statistically significant superiority.**

## 9. Paired Bootstrap Significance Diagnostics

Paired bootstrap (10k resamples) on quality-pass, judge-ok rows.
Pairing key: (story_name, question, answer, target_difficulty).
Metrics below are computed on the paired subset (not global).
Significance = 95% CI excludes 0.  Approximate p = 2 * min(P(diff<=0), P(diff>=0)).

| Baseline | Metric | Ours (paired) | Baseline (paired) | Diff | 95% CI | approx p | N | Sig? |
|---|---|---:|---:|---:|---|---:|---:|---|
| Direct | overall_accuracy | 50.8% | 49.2% | +1.57pp | [-5.24pp, +8.38pp] | 0.6878 | 191 | no |
| Direct | macro_accuracy | 50.0% | 47.9% | +2.12pp | [-4.83pp, +8.83pp] | 0.5288 | 191 | no |
| Direct | macro_f1 | 47.9% | 45.1% | +2.85pp | [-5.25pp, +10.57pp] | 0.4818 | 191 | no |
| Direct | spearman | 0.474 | 0.469 | +0.005 | [-0.122, +0.125] | 0.9332 | 191 | no |
| ICL | overall_accuracy | 49.7% | 48.6% | +1.13pp | [-6.78pp, +9.04pp] | 0.8410 | 177 | no |
| ICL | macro_accuracy | 49.8% | 48.7% | +1.06pp | [-6.93pp, +9.18pp] | 0.7986 | 177 | no |
| ICL | macro_f1 | 48.1% | 47.5% | +0.56pp | [-8.49pp, +9.83pp] | 0.8974 | 177 | no |
| ICL | spearman | 0.481 | 0.435 | +0.046 | [-0.098, +0.189] | 0.5248 | 177 | no |
| SelfRefine | overall_accuracy | 48.4% | 52.3% | -3.91pp | [-13.28pp, +5.47pp] | 0.4418 | 128 | no |
| SelfRefine | macro_accuracy | 48.5% | 52.5% | -4.05pp | [-13.03pp, +4.96pp] | 0.3716 | 128 | no |
| SelfRefine | macro_f1 | 47.4% | 52.1% | -4.68pp | [-14.61pp, +5.40pp] | 0.3562 | 128 | no |
| SelfRefine | spearman | 0.438 | 0.552 | -0.114 | [-0.268, +0.029] | 0.1198 | 128 | no |

## 10. End-to-End Accuracy (all candidates)

Denominator = all selected candidates per method (including graph failures,
parse errors, quality failures).  End-to-end = quality_pass AND judge_ok AND
predicted == target.

| Method | Easy | Medium | Hard | Overall | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 30.0% (45/150) | 47.3% (71/150) | 8.0% (12/150) | 28.4% (128/450) | 450 |
| ICL | 22.7% (34/150) | 47.3% (71/150) | 16.0% (24/150) | 28.7% (129/450) | 450 |
| SelfRefine | 18.0% (27/150) | 30.0% (45/150) | 10.7% (16/150) | 19.6% (88/450) | 450 |
| Ours | 26.7% (40/150) | 49.3% (74/150) | 16.0% (24/150) | 30.7% (138/450) | 450 |

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
| Direct | not answerable | 43 |
| Direct | gen error: degenerate output | 37 |
| Direct | wrong answer | 35 |
| Direct | not fluent | 20 |
| Direct | other | 15 |
| Direct | gen error: empty | 13 |
| Direct | gen error: no question mark | 7 |
| Direct | gen error: grammar: word repetition: old | 1 |
| Direct | gen error: grammar: word repetition: find | 1 |
| Direct | gen error: grammar: word repetition: into | 1 |
| Direct | gen error: does not end with ? | 1 |
| ICL | gen error: degenerate output | 46 |
| ICL | wrong answer | 44 |
| ICL | not answerable | 43 |
| ICL | gen error: empty | 13 |
| ICL | gen error: no question mark | 12 |
| ICL | other | 10 |
| ICL | not fluent | 10 |
| ICL | gen error: does not end with ? | 3 |
| ICL | gen error: grammar: word repetition: did | 1 |
| ICL | gen error: grammar: bad start: whywhy | 1 |
| ICL | gen error: grammar: word repetition: why | 1 |
| SelfRefine | gen error: degenerate initial output | 123 |
| SelfRefine | gen error: initial generation failed | 37 |
| SelfRefine | wrong answer | 34 |
| SelfRefine | not answerable | 31 |
| SelfRefine | not fluent | 18 |
| SelfRefine | gen error: no question mark | 16 |
| SelfRefine | other | 10 |
| SelfRefine | gen error: does not end with ? | 2 |
| SelfRefine | gen error: grammar: bad start: who, | 1 |
| SelfRefine | gen error: answer leakage | 1 |
| Ours | gen error: degenerate output | 55 |
| Ours | gen error: graph_invalid | 44 |
| Ours | gen error: self-check failed: answer mismatch; focus mismatch | 17 |
| Ours | gen error: no question mark | 14 |
| Ours | gen error: self-check failed: answer mismatch | 12 |
| Ours | gen error: parse failure | 10 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1-2 sentences; focus mismatch | 3 |
| Ours | gen error: does not end with ? | 3 |
| Ours | not fluent | 2 |
| Ours | gen error: grammar: word repetition: what | 2 |
| Ours | gen error: self-check failed: focus mismatch | 2 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1 sentence | 2 |
| Ours | gen error: empty | 2 |
| Ours | gen error: grammar: word repetition: him | 1 |
| Ours | gen error: question length out of range | 1 |
| Ours | gen error: answer leakage | 1 |
| Ours | gen error: grammar: word repetition: reason | 1 |
| Ours | gen error: self-check failed: answer mismatch; needs only 1 sentence; focus mismatch | 1 |
| Ours | not answerable | 1 |
| Ours | gen error: grammar: word repetition: during | 1 |
| Ours | gen error: grammar: bad start: whywhy | 1 |
| Ours | gen error: grammar: word repetition: order | 1 |

## 13. Examples

### Best Easy examples (quality-pass, correct prediction)

**Direct Example:**
- Story: how-the-river-gods-wedding-was-broken-off
- Question: How did the sorcerer feel when he begged for mercy?
- Target answer: scared .
- Target: Easy, Predicted: Easy

**ICL Example:**
- Story: money-box
- Question: How did they all feel about the comedy?
- Target answer: amused .
- Target: Easy, Predicted: Easy

**SelfRefine Example:**
- Story: princess-glass-mountain
- Question: Who was fond of the prince?
- Target answer: they liked him .
- Target: Easy, Predicted: Easy

**Ours Example:**
- Story: thomas-the-rhymer
- Question: What was Thomas known for after living in his castle for many years?
- Target answer: he was famous for his prophecies .
- Target: Easy, Predicted: Easy


### Best Medium examples (quality-pass, correct prediction)

**Direct Example:**
- Story: the-three-crowns
- Question: Why did the princess give the prince advice to put a heavy thing in the basket instead of himself?
- Target answer: she wanted to see if his brothers would do something bad to them .
- Target: Medium, Predicted: Medium

**ICL Example:**
- Story: the-three-crowns
- Question: Why did the princess give the prince advice to put a heavy thing in the basket instead of himself?
- Target answer: she wanted to see if his brothers would do something bad to them .
- Target: Medium, Predicted: Medium

**SelfRefine Example:**
- Story: the-tale-of-benjamin-bunny
- Question: What did Old Mr. Bunny do after driving the cat into the greenhouse?
- Target answer: he locked the door and whipped his son benjamin .
- Target: Medium, Predicted: Medium

**Ours Example:**
- Story: the-three-crowns
- Question: What was the princess's larger goal in instructing the prince to put a heavy object in the basket instead of himself?
- Target answer: she wanted to see if his brothers would do something bad to them .
- Target: Medium, Predicted: Medium


### Best Hard examples (quality-pass, correct prediction)

**Direct Example:**
- Story: the-believing-husbands
- Question: Why did the wife wait until she heard the funeral passing before telling the man to get up?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

**ICL Example:**
- Story: the-believing-husbands
- Question: Why did the wife wait until she heard the funeral procession before telling the man to get up?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

**SelfRefine Example:**
- Story: three-treasures-of-giants
- Question: Why did Martin and Michael change the contents of their bags multiple times as they moved through the castle?
- Target answer: copper , gold , and silver .
- Target: Hard, Predicted: Hard

**Ours Example:**
- Story: the-believing-husbands
- Question: What development connected the husband's initial intention to stay in bed to his sudden rush to attend the funeral?
- Target answer: she wanted to rush him to go to the funeral .
- Target: Hard, Predicted: Hard

