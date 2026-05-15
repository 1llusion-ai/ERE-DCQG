# Quality Pilot v3 — Filter Report

**Total samples:** 90
**Passed:** 35 (38.9%)

## Error Rates

- **Generation error:** 4/90 (4.4%)
- **Judge error:** 21 (26.6% of eligible)

## Primary Metric: asks_target_event

**Overall:** 91.4%
| Level | Total | Yes | Rate |
|-------|-------|-----|------|
| Easy | 18 | 17 | 94.4% |
| Medium | 19 | 18 | 94.7% |
| Hard | 21 | 18 | 85.7% |

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 30 | 19 | 63.3% |
| Medium | 30 | 8 | 26.7% |
| Hard | 30 | 8 | 26.7% |

## Grammar Failure Distribution

| Reason | Count |
|--------|-------|
| empty | 4 |
| base | 3 |
| repeat_question_mark | 2 |
| broken_grammar | 1 |
| too_long_hard | 1 |

Examples:

- [Hard] "" → empty
- [Medium] "What process did east-central Minnesota undergo by the end of the era, shifting its status from northern Minnesota to become the economic center of the area? ?" → repeat_question_mark
- [Hard] "After members of the Irgun disguise themselves and plant one bomb, did they send any any kind of of of warning regarding the bombing itself they they?" → base: word repetition: any
- [Medium] "What impact did the Staplehurst rail crash have on Charles Dickens' ability to speak after the incident? ?" → repeat_question_mark
- [Medium] "What the first significant combat engagement in NATO's history Operation Deny Flight and how did it contribute to the operation's objectives?" → broken_grammar: What the first

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| none | 82 |
| needs_phrase | 8 |

Failures:

- trigger="receiving" type=needs_phrase → needs_phrase trigger without valid phrase: 'receiving'

## Answer Phrase Pass Rate

- Pass rate: 86.7%

## Answer Consistency

| Label | Count |
|-------|-------|
| yes | 50 |
| partial | 3 |
| no | 16 |
| judge_error | 21 |

- yes rate: 72.5%
- yes+partial rate: 76.8%

Inconsistency examples:

- [Hard] Q: "After Winifred originated as a tropical low and prompting the issuance of cyclone watches and warnings, what did officials warn residents about due to power outages?"
  gold_phrase="power outages disrupted electrical service" expected=""
  reason: The question does not specifically ask about power outages disrupting electrical service or the final event of disrupted electrical service. The target event and the question are not consistent with the provided context.
- [Hard] Q: ""
  gold_phrase="aftermath" expected=""
  reason: skipped (early exit)
- [Easy] Q: "What direction did the storm take after reaching its peak intensity near the Bahamas?"
  gold_phrase="The storm was propelled northward" expected=""
  reason: The question does not specifically ask about the storm being propelled northward, but rather about its direction after reaching peak intensity.
- [Medium] Q: "What process did east-central Minnesota undergo by the end of the era, shifting its status from northern Minnesota to become the economic center of the area? ?"
  gold_phrase="replaced" expected=""
  reason: skipped (early exit)
- [Medium] Q: "What was the final outcome of the events that began with extradition and included a raid on CONCACAF headquarters?"
  gold_phrase="The arrests case triggered Australia, Colombia" expected=""
  reason: The question asks about the final outcome of events starting with extradition and a raid on CONCACAF headquarters, which is not the same as the target event about arrests and subsequent investigations.

## Path Coverage

| Level | Avg Coverage | Pass Count | Pass Rate |
|-------|-------------|------------|-----------|
| Easy | 0.87 | 22 | 73.3% |
| Medium | 1.27 | 9 | 30.0% |
| Hard | 1.5 | 17 | 56.7% |

Path coverage failures:

- [Easy] "Why was Big Show used as a storyline replacement for Stone Cold Steve Austin?" coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Easy] "What did the UN Security Council resolutions do in relation to the conflict? " coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Easy] "What event followed the massacre and led to the displacement of the Iroquois from western New York?" coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Easy] "What political impact did Native Americans have due to their involvement in the fur trade?" coverage=0 → covers 0/2 events, need >= 1 [FAIL]
- [Easy] "" coverage=0 → skipped (early exit)

## Hard Degraded

- Degraded count: 10
- Degraded ratio: 33.3%

Degraded examples:

- "After the government claimed to have foiled an alleged coup d'état and allowed for the suspension of public activities, what did President Arroyo do regarding the state of emergency?" → can_answer_from_single_sentence=yes (sent=S9)
  single=yes, need_intermediate=no
- "After she suffered brain damage due to asphyxia and was brought to life support, what did campaigners continue to push for regarding Gardner's death?" → can_answer_from_single_sentence=yes (sent=S7)
  single=yes, need_intermediate=yes
- "After the hurricane tracked westward toward the Sargasso Sea and killed many people, what did it eventually become and do?" → can_answer_from_single_sentence=yes (sent=S9)
  single=yes, need_intermediate=yes
- "After the Irgun disguised themselves as workmen and planted a bomb, what appears to have been the outcome regarding the hoax call at the hotel?" → can_answer_from_single_sentence=yes (sent=S9)
  single=yes, need_intermediate=no
- "After the Irish Catholic gentry tried to seize control and their suspected association with King Charles started the English Civil War, what did the English and Scottish Parliaments do regarding raising an army to put down the rebellion?" → can_answer_from_single_sentence=yes (sent=S5)
  single=yes, need_intermediate=no

## Success Criteria

| Criterion | Target | Actual | Met |
|-----------|--------|--------|-----|
| overall_pass_rate | >=50% | 38.9% | NO |
| answer_consistency_yes+partial | >=80% | 76.8% | NO |
| answer_consistency_yes | >=60% | 72.5% | YES |
| medium_path_coverage | >=70% | 30.0% | NO |
| hard_path_coverage | >=80% | 56.7% | NO |
| hard_degraded | <=20% | 33.3% | NO |
| grammar_fail | <=15% | 12.2% | YES |

## Recommendation

NOT READY to scale. Failed criteria: overall_pass_rate_50pct, answer_consistency_yes_partial_80pct, medium_path_coverage_70pct, hard_path_coverage_80pct, hard_degraded_20pct. Analyze failures and fix prompt/filter before scaling.