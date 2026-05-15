# Quality Pilot v3 — Filter Report

**Total samples:** 15
**Passed:** 7 (46.7%)

## Error Rates

- **Generation error:** 2/15 (13.3%)
- **Judge error:** 3 (25.0% of eligible)

## Primary Metric: asks_target_event

**Overall:** 88.9%
| Level | Total | Yes | Rate |
|-------|-------|-----|------|
| Easy | 3 | 2 | 66.7% |
| Medium | 3 | 3 | 100.0% |
| Hard | 3 | 3 | 100.0% |

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 5 | 3 | 60.0% |
| Medium | 5 | 2 | 40.0% |
| Hard | 5 | 2 | 40.0% |

## Grammar Failure Distribution

| Reason | Count |
|--------|-------|
| empty | 2 |
| base | 1 |

Examples:

- [Hard] "After NATO operations on December 14 and the operation contributed forces to the operation help on  on December  did officials do?" → base: word repetition: on
- [Hard] "" → empty
- [Easy] "" → empty

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| none | 12 |
| needs_phrase | 3 |

## Answer Phrase Pass Rate

- Pass rate: 80.0%

## Answer Consistency

| Label | Count |
|-------|-------|
| yes | 8 |
| partial | 0 |
| no | 4 |
| judge_error | 3 |

- yes rate: 66.7%
- yes+partial rate: 66.7%

Inconsistency examples:

- [Hard] Q: "After NATO operations on December 14 and the operation contributed forces to the operation help on  on December  did officials do?"
  gold_phrase="by its end on 20 December 1995, NATO pilots had flown 100,420 sorties" expected=""
  reason: skipped (early exit)
- [Easy] Q: "What did the Department of Social Security (DSS) do after the cyclone to assist affected farmers?"
  gold_phrase="sent employees to receive claims" expected=""
  reason: The question asks about assistance to affected farmers, while the target event is about the DSS sending employees to receive claims. The target sentence does not mention assistance to farmers.
- [Hard] Q: ""
  gold_phrase="are perceived as the area's "early" history in fact originated" expected=""
  reason: skipped (early exit)
- [Easy] Q: ""
  gold_phrase="buildings were still seen in the affected areas as late as 1951" expected=""
  reason: skipped (early exit)

## Path Coverage

| Level | Avg Coverage | Pass Count | Pass Rate |
|-------|-------------|------------|-----------|
| Easy | 0.8 | 3 | 60.0% |
| Medium | 1.4 | 2 | 40.0% |
| Hard | 1.2 | 3 | 60.0% |

Path coverage failures:

- [Easy] "What did the Department of Social Security (DSS) do after the cyclone to assist affected farmers?" coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Easy] "" coverage=0 → skipped (early exit)
- [Medium] "What significant event prompted the evacuation of 600,000 people in Bangladesh, leading to the deaths of 30 people in a plane crash in Vietnam?" coverage=1 → covers 1/3 events, need >= 2 [FAIL]
- [Medium] "What happened to Charles Dickens five years after the Staplehurst rail crash that indicated he had not fully recovered from the experience?" coverage=1 → covers 1/3 events, need >= 2 [FAIL]
- [Medium] "What number did the commander provide to make the defenders seem larger than they were during the Battle of Shanghai, and why was this done?" coverage=1 → covers 1/3 events, need >= 2 [FAIL]

## Hard Degraded

- Degraded count: 1
- Degraded ratio: 20.0%

Degraded examples:

- "How did the operations that began on 12 April 1993 and spanned more than two years demonstrate NATO's adaptability in the post-Cold War era during the Bosnian War?" → can_answer_from_single_sentence=yes (sent=S5)
  single=yes, need_intermediate=yes

## Success Criteria

| Criterion | Target | Actual | Met |
|-----------|--------|--------|-----|
| overall_pass_rate | >=50% | 46.7% | NO |
| answer_consistency_yes+partial | >=80% | 66.7% | NO |
| answer_consistency_yes | >=60% | 66.7% | YES |
| medium_path_coverage | >=70% | 40.0% | NO |
| hard_path_coverage | >=80% | 60.0% | NO |
| hard_degraded | <=20% | 20.0% | YES |
| grammar_fail | <=15% | 20.0% | NO |

## Recommendation

NOT READY to scale. Failed criteria: overall_pass_rate_50pct, answer_consistency_yes_partial_80pct, medium_path_coverage_70pct, hard_path_coverage_80pct, grammar_fail_15pct. Analyze failures and fix prompt/filter before scaling.