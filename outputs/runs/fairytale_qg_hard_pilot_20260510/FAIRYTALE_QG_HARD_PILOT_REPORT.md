# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-10 19:54:40

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Total candidates | 15 |
| Total generations | 60 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 14 | 15 | 93.3% |
| ICL | 13 | 15 | 86.7% |
| SelfRefine | 5 | 15 | 33.3% |
| Ours | 13 | 15 | 86.7% |

## 2. Quality Pass by Method

| Method | quality_pass | Total | Pct |
|---|---:|---:|---:|
| Direct | 6 | 15 | 40.0% |
| ICL | 2 | 15 | 13.3% |
| SelfRefine | 2 | 15 | 13.3% |
| Ours | 2 | 15 | 13.3% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| Direct | 6 | 0 | 0 | 6 |
| ICL | 2 | 0 | 0 | 2 |
| SelfRefine | 2 | 0 | 0 | 2 |
| Ours | 2 | 0 | 0 | 2 |

## 4. Hard Hit Rate by Method

predicted Hard / quality-pass

| Method | Hard | quality-pass | Hard hit rate |
|---|---:|---:|---:|
| Direct | 0 | 6 | 0.0% |
| ICL | 0 | 2 | 0.0% |
| SelfRefine | 0 | 2 | 0.0% |
| Ours | 0 | 2 | 0.0% |

## 5. Evidence Dependency by Method (quality-pass only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 0 | 0 | 0 | 6 |
| ICL | 0 | 0 | 0 | 2 |
| SelfRefine | 0 | 0 | 0 | 2 |
| Ours | 0 | 0 | 0 | 2 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | other | 4 |
| Direct | not answerable | 4 |
| Direct | wrong answer | 1 |
| ICL | not answerable | 8 |
| ICL | other | 4 |
| ICL | answer leakage | 1 |
| SelfRefine | not answerable | 11 |
| SelfRefine | other | 2 |
| Ours | not answerable | 8 |
| Ours | other | 2 |
| Ours | wrong answer | 2 |
| Ours | answer leakage | 1 |

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 15 | 0 | 0.0% |
| ICL | 15 | 0 | 0.0% |
| SelfRefine | 15 | 0 | 0.0% |
| Ours | 15 | 0 | 0.0% |

## 8. Examples

### Best Ours examples (quality-pass, predicted Hard)

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: How many times did the young swineherd exchange his pigs for dogs with special abilities?
- Target answer: three times .
- Predicted difficulty: Easy

**ICL Example:**
- Story: three-dogs
- Question: How many times did the youth exchange his pigs for dogs with the old man?
- Target answer: three times .
- Predicted difficulty: Easy

**SelfRefine Example:**
- Story: master-girl
- Question: What transformation did the king's son observe after dipping his hair in the second kettle, and how did he react to this discovery?
- Target answer: the second kettle turned his hair into silver .
- Predicted difficulty: Easy

### Failure cases

**Failure 1 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

**Failure 2 (ICL):**
- Story: three-dogs
- Question: Why did the princes decide to plot against the youth and threaten the princesses?
- Reason: The question can be answered from the story, but the expected answer is not fully consistent with the story. The story mentions jealousy but does not explicitly state that the princes decided to plot against the youth and threaten the princesses due to jealousy alone.

**Failure 3 (SelfRefine):**
- Story: three-dogs
- Question: 
- Reason: empty question

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality-pass >= 60% | FAIL | 13.3% |
| Ours predicted Hard among quality-pass >= 20% | FAIL | 0.0% |
| Ours bridge_required=yes >= 50% | FAIL | 0.0% |
| Ours Hard hit > Direct/ICL/SelfRefine | FAIL | Ours=0.00, Direct=0.00, ICL=0.00, SelfRefine=0.00 |
| No method copies source question | PASS | pass |

**Overall: SOME CRITERIA FAILED**
