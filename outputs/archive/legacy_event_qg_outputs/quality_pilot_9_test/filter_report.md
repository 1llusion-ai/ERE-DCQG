# Quality Pilot v3 — Filter Report

**Total samples:** 9
**Passed:** 3 (33.3%)

## Error Rates

- **Generation error:** 0/9 (0.0%)
- **Judge error:** 3 (37.5% of eligible)

## Primary Metric: asks_target_event

**Overall:** 60.0%
| Level | Total | Yes | Rate |
|-------|-------|-----|------|
| Easy | 3 | 2 | 66.7% |
| Medium | 2 | 1 | 50.0% |
| Hard | 0 | 0 | 0.0% |

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 3 | 0 | 0.0% |
| Medium | 3 | 1 | 33.3% |
| Hard | 3 | 2 | 66.7% |

## Grammar Failure Distribution

| Reason | Count |
|--------|-------|
| base | 1 |

Examples:

- [Hard] "By the end of the era, how did the demographic shift and cultural transition affect the influence of Native Americans compared to the beginning of the period? " → base: bad start: by

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| none | 9 |

## Answer Phrase Pass Rate

- Pass rate: 66.7%

## Answer Consistency

| Label | Count |
|-------|-------|
| yes | 3 |
| partial | 0 |
| no | 3 |
| judge_error | 3 |

- yes rate: 50.0%
- yes+partial rate: 50.0%

Inconsistency examples:

- [Medium] Q: "What event prompted the issuance of cyclone watches and warnings by the Australian Bureau of Meteorology before Winifred's approach?"
  gold_phrase="turning toward the coast, southwestward" expected=""
  reason: The question does not ask about the 'turning' event but rather about the issuance of cyclone watches and warnings, which is related to the cyclone's approach.
- [Hard] Q: "By the end of the era, how did the demographic shift and cultural transition affect the influence of Native Americans compared to the beginning of the period? "
  gold_phrase="by the time statehood was achieved" expected=""
  reason: skipped (early exit)
- [Easy] Q: "What was the result of the concert after it was titled?"
  gold_phrase="playing in front of a large audience of" expected=""
  reason: The question is about the title of the concert and its broadcast, not about the commercial success or the audience playing.

## Path Coverage

| Level | Avg Coverage | Pass Count | Pass Rate |
|-------|-------------|------------|-----------|
| Easy | 1.0 | 2 | 66.7% |
| Medium | 1.0 | 1 | 33.3% |
| Hard | 1.33 | 2 | 66.7% |

Path coverage failures:

- [Easy] "What happened to Bouch within the year?" coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Medium] "What event prompted the issuance of cyclone watches and warnings by the Australian Bureau of Meteorology before Winifred's approach?" coverage=0 → covers 0/3 events, need >= 2 [FAIL]
- [Medium] "What significant event led to the evacuation of 600,000 people in Bangladesh and also resulted in the deaths of 30 people in a plane crash in Vietnam?" coverage=1 → covers 1/3 events, need >= 2 [FAIL]
- [Hard] "By the end of the era, how did the demographic shift and cultural transition affect the influence of Native Americans compared to the beginning of the period? " coverage=0 → skipped (early exit)

## Hard Degraded

- Degraded count: 0
- Degraded ratio: 0.0%

## Success Criteria

| Criterion | Target | Actual | Met |
|-----------|--------|--------|-----|
| overall_pass_rate | >=50% | 33.3% | NO |
| answer_consistency_yes+partial | >=80% | 50.0% | NO |
| answer_consistency_yes | >=60% | 50.0% | NO |
| medium_path_coverage | >=70% | 33.3% | NO |
| hard_path_coverage | >=80% | 66.7% | NO |
| hard_degraded | <=20% | 0.0% | YES |
| grammar_fail | <=15% | 11.1% | YES |

## Recommendation

NOT READY to scale. Failed criteria: overall_pass_rate_50pct, answer_consistency_yes_partial_80pct, answer_consistency_yes_60pct, medium_path_coverage_70pct, hard_path_coverage_80pct. Analyze failures and fix prompt/filter before scaling.