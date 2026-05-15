# Hard Rescue Pilot Report (Quality Filter Edition)

**Date:** 2026-05-07 00:39
**Paths:** 8
**Strategies:** hidden_endpoint, relation_composition
**K candidates per path per strategy:** 4
**Total candidates generated:** 64
**API calls:** generation=148, filter=29, judge=87

## 1. Pool Statistics

| Stage | Count |
|-------|------:|
| Selected Hard paths | 8 |
| Total candidates | 64 |
| Generation errors | 35 |
| Grammar pass | 26 |
| Drift check failures | 47 |
| Drift repaired | 21 |
| Too direct answer-type cue | 14 |
| Unsupported answer type (skipped) | 2 |
| **Quality filter pass** | **10** |

### Quality Filter Fail Reasons

| Reason | Count |
|--------|------:|
| answer_consistency=no | 14 |
| double_question | 10 |
| answer_type=invalid | 8 |
| answer_phrase=skipped (early exit) | 3 |
| fec=no | 3 |
| grammar=base: no question mark | 2 |
| alignment_asks=no | 2 |
| target_drift=yes | 2 |
| trigger_leakage | 1 |
| grammar=base: word repetition: of | 1 |

## 2. Quality-Pass Difficulty Distribution

*(Computed over quality_filter_pass candidates only)*

| Metric | Count | Rate |
|--------|------:|-----:|
| Quality-pass (judged) | 10 | — |
| Blind Pred Easy | 10 | 100.0% |
| Blind Pred Medium | 0 | 0.0% |
| Blind Pred Hard | 0 | 0.0% |

### Required Steps (Blind Judge)

| Steps | Count | Rate |
|------:|------:|-----:|
| 1 | 10 | 100.0% |

### Single Sentence Answerable (Blind Judge)

| SSA | Count | Rate |
|-----|------:|-----:|
| yes | 10 | 100.0% |

## 3. Quality-Pass Structural Metrics

### Path Dependency

| Level | Count | Rate |
|-------|------:|-----:|
| strong | 5 | 50.0% |
| none | 5 | 50.0% |

### Shortcut Without Path

| Value | Count | Rate |
|-------|------:|-----:|
| no | 10 | 100.0% |

## 4. Quality Metrics (Among Quality-Pass)

| Metric | Value |
|--------|------:|
| Answerable (yes/partial) | 10/10 (100.0%) |
| Final-Event Consistent (yes/partial) | 10/10 (100.0%) |
| Alignment asks (yes/partial) | 10/10 (100.0%) |
| Target drift != yes | 10/10 (100.0%) |
| Answer consistency != no | 10/10 (100.0%) |

## 5. Per Answer Type (Quality-Pass Only)

| Hard Answer Type | N QP | Blind Easy | Blind Med | Blind Hard | FEC yes/partial | SSA=no | PathDep strong |
|-----------------|-----:|----------:|---------:|----------:|----------------|-------:|---------------:|
| agreement_resolution | 3 | 3 | 0 | 0 | 3 (100%) | 0 | 0 |
| casualty_damage | 5 | 5 | 0 | 0 | 5 (100%) | 0 | 3 |
| investigation_outcome | 2 | 2 | 0 | 0 | 2 (100%) | 0 | 2 |

## 6. Per Strategy

| Strategy | N judged | QP Rate | Blind Hard | Blind Med | Blind Easy | FEC% | SSA=no% | PathDep strong% |
|----------|--------:|--------:|----------:|---------:|----------:|-----:|--------:|----------------:|
| hidden_endpoint | 15 | 19% | 0 (0%) | 4 (27%) | 11 (73%) | 87% | 7% | 80% |
| relation_composition | 14 | 12% | 1 (7%) | 3 (21%) | 10 (71%) | 93% | 7% | 57% |

## 7. Quality-Pass Samples by Predicted Difficulty

### Easy Samples (top 3)

### Easy #1 [hidden_endpoint]

- **Question:** What happened to the case after the Myyrmanni bombing?
- **Answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **Event path:** crowded -> took place -> investigated -> closed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes | **Shortcut:** no
- **Quality Filter Pass:** True
- **Quality Reason:** all checks passed
- **Blind Judge Reason:** The answer can be directly found in sentence S6, which states that the case was closed in January 2003 without any indictments.

### Easy #2 [relation_composition]

- **Question:** What final harm resulted from Aleksander Rogaliński's uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes | **Shortcut:** no
- **Quality Filter Pass:** True
- **Quality Reason:** all checks passed
- **Blind Judge Reason:** The answer can be directly found in sentence S6, which states that the Russian commander was killed after the hand-to-hand fight.

### Easy #3 [hidden_endpoint]

- **Question:** What final harm resulted from the initial clash of the uprising?
- **Answer:** After a short hand-to-hand fight the Russian commander was killed
- **Event path:** uprising -> negotiate -> refused -> killed
- **Blind Pred:** Easy | **PathDep:** strong | **Answerable:** yes | **FEC:** yes | **SSA:** yes | **Shortcut:** no
- **Quality Filter Pass:** True
- **Quality Reason:** all checks passed
- **Blind Judge Reason:** The answer can be directly found in sentence S6, which states that the Russian commander was killed after the hand-to-hand fight.

### Medium Samples (top 3)

*No quality-pass medium samples.*

### Hard Samples (top 3)

*No quality-pass hard samples.*

## 8. Oracle Top-1 Diagnostic

> **NOT USED FOR MAIN METRICS. Diagnostic only.**

| Metric | Count | Rate |
|--------|------:|-----:|
| Total paths | 4 | — |
| Oracle Blind Hard | 1 | 25.0% |
| Oracle SSA=no | 2 | 50.0% |
| Oracle PathDep strong | 4 | 100.0% |
| Oracle quality-filter-pass | 0 | 0.0% |

| # | Strategy | Blind Pred | SSA | PathDep | QP | Question (truncated) |
|--:|----------|-----------:|-----|---------|----|---------------------|
| 1 | hidden_endpoint | Medium | no | strong | N | What led to the French and British forces moving t |
| 2 | hidden_endpoint | Medium | partial | strong | N | What were the consequences of the Dutch beginning  |
| 3 | relation_composition | Hard | no | strong | N | How did the presence of many children at the Myyrm |
| 4 | relation_composition | Medium | partial | strong | N | What outcry resulted when the refusal of Polish fo |

## Success / Readiness Criteria

- Quality filter pass rate: 10/29 (34.5%)
- [PASS] FEC among quality-pass >= 80% (100.0%)
- [PASS] Alignment (asks_expected_answer) among quality-pass >= 80% (100.0%)
- [INFO] Difficulty distribution (quality-pass): Easy=10, Medium=0, Hard=0
- [NOTE] No difficulty metric used to select main evaluation set
