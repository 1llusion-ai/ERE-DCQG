# Quality Pilot 90 — Filter Report

**Total samples:** 90
**Passed:** 11 (12.2%)

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 30 | 7 | 23.3% |
| Medium | 30 | 3 | 10.0% |
| Hard | 30 | 1 | 3.3% |

## Grammar Failure Distribution

| Reason | Count |
|--------|-------|
| empty | 17 |
| repeat_question_mark | 2 |
| base | 1 |

Examples:

- [Easy] "" → empty
- [Easy] "" → empty
- [Easy] "" → empty
- [Medium] "" → empty
- [Easy] "" → empty

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| hard_blacklist | 5 |
| none | 83 |
| needs_phrase | 2 |

Failures:

- trigger="occurred" type=hard_blacklist → hard_blacklisted trigger: 'occurred'
- trigger="made" type=hard_blacklist → hard_blacklisted trigger: 'made'
- trigger="made" type=hard_blacklist → hard_blacklisted trigger: 'made'
- trigger="took place" type=hard_blacklist → hard_blacklisted trigger: 'took place'
- trigger="made" type=hard_blacklist → hard_blacklisted trigger: 'made'

## Answer Phrase Pass Rate

- Pass rate: 74.4%

## Answer Consistency

| Label | Count |
|-------|-------|
| no | 52 |
| yes | 34 |
| partial | 4 |

- yes rate: 37.8%
- yes+partial rate: 42.2%

Inconsistency examples:

- [Medium] Q: "What of the Tay Bridge Disaster was NOT due to a specific part?"
  gold_phrase="Bridge Disaster occurred during a violent storm" expected="the Tay Bridge Disaster was caused by a violent storm on December 28, 187."
  reason: EXPECTED=the Tay Bridge Disaster was caused by a violent storm on December 28, 187.
TYPE=etermine
- [Medium] Q: "What happened to the concertgoers who fainted due to the extreme heat at the concert? "
  gold_phrase="were lifted overhead to the indoor areas of the stadium" expected="The concert in Dallas experienced extreme heat,"
  reason: The the question question question question does not not "canceled the concert" but did theD onD mention mention mention mention the extreme heat affecting the concertD
- [Easy] Q: ""
  gold_phrase="billed" expected=""
  reason: skipped (early exit)
- [Easy] Q: ""
  gold_phrase="restrain" expected=""
  reason: skipped (early exit)
- [Easy] Q: ""
  gold_phrase="replacing" expected=""
  reason: skipped (early exit)

## Path Coverage

| Level | Avg Coverage | Pass Count | Pass Rate |
|-------|-------------|------------|-----------|
| Easy | 0.77 | 20 | 66.7% |
| Medium | 1.03 | 8 | 26.7% |
| Hard | 1.4 | 3 | 10.0% |

Path coverage failures:

- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "" coverage=0 → skipped (early exit)
- [Easy] "" coverage=0 → skipped (early exit)

## Hard Degraded

- Degraded count: 11
- Degraded ratio: 36.7%

Degraded examples:

- "After Charles Dickens was travelling with Ellen Ternan and her mother and the train crashed, how did this affect him in the long term?" → can_answer_from_single_sentence=yes (sent=S5)
  single=yes, need_intermediate=yes
- "What was the sequence of events that occurred between the government's claim of foiling a coup and the revocation of demonstration permits?" → can_answer_from_single_sentence=yes (sent=S4)
  single=yes, need_intermediate=no
- "After the Polish troops fought and the Russians approached the local manor, how did the Russian unit end up?" → can_answer_from_single_sentence=yes (sent=S6)
  single=yes, need_intermediate=no
- "What event occurred after the hurricane reached its peak intensity of 160 mph but before it dissipated?" → can_answer_from_single_sentence=yes (sent=S8)
  single=yes, need_intermediate=yes
- "After the tour promoted the band's fourth studio album and visited Europe and North America, what did the film do with the concert in Geneva?" → can_answer_from_single_sentence=yes (sent=S9)
  single=yes, need_intermediate=no

## Success Criteria

| Criterion | Target | Actual | Met |
|-----------|--------|--------|-----|
| overall_pass_rate | >=50% | 12.2% | NO |
| answer_consistency_yes+partial | >=80% | 42.2% | NO |
| answer_consistency_yes | >=60% | 37.8% | NO |
| medium_path_coverage | >=70% | 26.7% | NO |
| hard_path_coverage | >=80% | 10.0% | NO |
| hard_degraded | <=20% | 36.7% | NO |
| grammar_fail | <=15% | 22.2% | NO |

## Recommendation

NOT READY to scale. Failed criteria: overall_pass_rate_50pct, answer_consistency_yes_partial_80pct, answer_consistency_yes_60pct, medium_path_coverage_70pct, hard_path_coverage_80pct, hard_degraded_20pct, grammar_fail_15pct. Analyze failures and fix prompt/filter before scaling.