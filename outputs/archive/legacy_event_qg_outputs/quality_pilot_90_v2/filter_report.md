# Quality Pilot 90 v2 — Filter Report

**Total samples:** 90
**Passed:** 15 (16.7%)

## Error Rates

- **Generation error:** 30/90 (33.3%)
- **Judge error:** 14 (24.6% of eligible)

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 30 | 9 | 30.0% |
| Medium | 30 | 5 | 16.7% |
| Hard | 30 | 1 | 3.3% |

## Grammar Failure Distribution

| Reason | Count |
|--------|-------|
| empty | 30 |
| base | 3 |

Examples:

- [Medium] "" → empty
- [Easy] "" → empty
- [Easy] "" → empty
- [Easy] "" → empty
- [Easy] "" → empty

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| hard_blacklist | 4 |
| none | 86 |

Failures:

- trigger="occurred" type=hard_blacklist → hard_blacklisted trigger: 'occurred'
- trigger="made" type=hard_blacklist → hard_blacklisted trigger: 'made'
- trigger="occurred" type=hard_blacklist → hard_blacklisted trigger: 'occurred'
- trigger="made" type=hard_blacklist → hard_blacklisted trigger: 'made'

## Answer Phrase Pass Rate

- Pass rate: 54.4%

## Answer Consistency

| Label | Count |
|-------|-------|
| yes | 17 |
| partial | 8 |
| no | 51 |
| judge_error | 14 |

- yes rate: 22.4%
- yes+partial rate: 32.9%

Inconsistency examples:

- [Medium] Q: "What happened to the design of the Tay Rail Bridge that made it more susceptible to collapse during the storm in 1879?"
  gold_phrase="The Tay Bridge Disaster occurred during a violent storm on Sunday 28 December 1879" expected=""
  reason: The question asks about the design flaws that made the Tay Rail Bridge more susceptible to collapse, but the gold answer focuses on the occurrence of the disaster without addressing the design issues.
- [Medium] Q: ""
  gold_phrase="lifted" expected=""
  reason: skipped (early exit)
- [Easy] Q: ""
  gold_phrase="restrain" expected=""
  reason: skipped (early exit)
- [Easy] Q: "Why did the demand for furs in Europe decline during the early 19th century?"
  gold_phrase="demand for furs in Europe diminished" expected=""
  reason: The question asks about the reason for the decline in demand for furs in Europe, while the gold phrase only states that the demand diminished without providing a reason.
- [Easy] Q: ""
  gold_phrase="replacing" expected=""
  reason: skipped (early exit)

## Path Coverage

| Level | Avg Coverage | Pass Count | Pass Rate |
|-------|-------------|------------|-----------|
| Easy | 0.63 | 14 | 46.7% |
| Medium | 0.8 | 9 | 30.0% |
| Hard | 1.2 | 3 | 10.0% |

Path coverage failures:

- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "Why did the demand for furs in Europe decline during the early 19th century?" coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "" coverage=0 → skipped (early exit)

## Hard Degraded

- Degraded count: 11
- Degraded ratio: 36.7%

Degraded examples:

- "After the defenders held out during the Battle of Shanghai and the successful defense marked a morale boost for the Chinese army, how did the Chinese authorities respond to inform the public about the heroic efforts of the 423 defenders?" → can_answer_from_single_sentence=yes (sent=S7)
  single=yes, need_intermediate=no
- "After the Irgun members were disguised and planted the bomb, what action did the hotel staff take in response to the increasing alarm?" → can_answer_from_single_sentence=yes (sent=None)
  single=yes, need_intermediate=no
- "After the government claimed to have foiled an alleged coup and revoked licenses for demonstrations, what did President Arroyo do regarding the state of emergency?" → can_answer_from_single_sentence=yes (sent=S4)
  single=yes, need_intermediate=no
- "What was the Russian commander's initial intention when he ordered a loose formation and tried to negotiate an agreement with Rogaliński?" → can_answer_from_single_sentence=yes (sent=S4)
  single=yes, need_intermediate=no
- "After Mohammed Merah wounded six agents during the siege and was killed by the police, what was the outcome for his brother and an accomplice?" → can_answer_from_single_sentence=yes (sent=S11)
  single=yes, need_intermediate=no

## Success Criteria

| Criterion | Target | Actual | Met |
|-----------|--------|--------|-----|
| overall_pass_rate | >=50% | 16.7% | NO |
| answer_consistency_yes+partial | >=80% | 32.9% | NO |
| answer_consistency_yes | >=60% | 22.4% | NO |
| medium_path_coverage | >=70% | 30.0% | NO |
| hard_path_coverage | >=80% | 10.0% | NO |
| hard_degraded | <=20% | 36.7% | NO |
| grammar_fail | <=15% | 36.7% | NO |

## Recommendation

NOT READY to scale. Failed criteria: overall_pass_rate_50pct, answer_consistency_yes_partial_80pct, answer_consistency_yes_60pct, medium_path_coverage_70pct, hard_path_coverage_80pct, hard_degraded_20pct, grammar_fail_15pct. Analyze failures and fix prompt/filter before scaling.