# Path Prefilter Report

**Total paths:** 363
**Passed:** 292 (80.4%)

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 125 | 91 | 72.8% |
| Medium | 122 | 102 | 83.6% |
| Hard | 116 | 99 | 85.3% |

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| none | 323 |
| needs_phrase | 31 |
| hard_blacklist | 9 |

Examples:

- [Easy] trigger="control" -> weak trigger but valid phrase: 'In line with the agreement with Johor in 1606, the Dutch took control of Malacca'
- [Medium] trigger="control" -> weak trigger but valid phrase: 'In line with the agreement with Johor in 1606, the Dutch took control of Malacca'
- [Hard] trigger="control" -> weak trigger but valid phrase: 'In line with the agreement with Johor in 1606, the Dutch took control of Malacca'
- [Easy] trigger="occurred" -> hard_blacklisted trigger: 'occurred'
- [Hard] trigger="occurred" -> hard_blacklisted trigger: 'occurred'
- [Easy] trigger="took place" -> hard_blacklisted trigger: 'took place'

## Answer Phrase

- Pass rate: 85.1%

Failures:

- [Easy] trigger="given" phrase="given" -> phrase equals trigger
- [Easy] trigger="given" phrase="given" -> phrase equals trigger
- [Medium] trigger="given" phrase="given" -> phrase equals trigger
- [Easy] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" -> partial extraction (no clause boundary found)
- [Easy] trigger="designed" phrase="American military interventions in Nicaragua were designed to stop any other nation except the United States of America from building a Nicaraguan Canal" -> partial extraction (no clause boundary found)

## Relation Group by Difficulty

| Level | TEMPORAL | CAUSE | SUBEVENT | MIXED | NONE |
|-------|----------|-------|----------|-------|------|
| Easy | 102 | 23 | 0 | 0 | 0 |
| Medium | 91 | 30 | 1 | 0 | 0 |
| Hard | 59 | 54 | 3 | 0 | 0 |

- Temporal-only Hard: 59 (50.9%)

## Single-Sentence Risk

| Level | Low | Medium | High |
|-------|-----|--------|------|
| Easy | 92 | 13 | 20 |
| Medium | 116 | 6 | 0 |
| Hard | 112 | 3 | 1 |

## Failure Reasons

| Reason | Count |
|--------|-------|
| answer_phrase_fail: partial extraction (no clause boundary found) | 31 |
| hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found) | 7 |
| answer_phrase_fail: partial extraction: phrase ends with dangling word 'over' | 5 |
| hard_weak_trigger='took place' | 4 |
| answer_phrase_fail: phrase equals trigger | 3 |
| hard_weak_trigger='took' | 3 |
| answer_phrase_fail: partial extraction: unclosed bracket or quote | 3 |
| hard_weak_trigger='made' | 3 |
| hard_weak_trigger='occurred' | 2 |
| hard_weak_trigger='seen' | 2 |
| answer_phrase_fail: partial extraction (no clause boundary found); soft_weak_trigger='held' with no valid phrase | 2 |
| hard_weak_trigger='began' | 1 |
| hard_weak_trigger='become' | 1 |
| answer_phrase_fail: partial extraction (no clause boundary found); soft_weak_trigger='Battle' with no valid phrase | 1 |
| hard_weak_trigger='ended' | 1 |
| answer_phrase_fail: partial extraction: phrase ends with dangling word 'in' | 1 |
| hard_weak_trigger='called'; answer_phrase_fail: partial extraction: unclosed bracket or quote | 1 |

Examples:

- [Easy] trigger="given" -> answer_phrase_fail: phrase equals trigger
- [Easy] trigger="given" -> answer_phrase_fail: phrase equals trigger
- [Medium] trigger="given" -> answer_phrase_fail: phrase equals trigger
- [Easy] trigger="ended" -> hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="ended" -> hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="ended" -> hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="designed" -> answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="combined" -> answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="destroyed" -> answer_phrase_fail: partial extraction (no clause boundary found)
- [Medium] trigger="took" -> hard_weak_trigger='took'
- [Hard] trigger="took" -> hard_weak_trigger='took' [risk: temporal_only_hard]
- [Easy] trigger="took" -> hard_weak_trigger='took'
- [Easy] trigger="began" -> hard_weak_trigger='began'
- [Easy] trigger="occurred" -> hard_weak_trigger='occurred'
- [Hard] trigger="occurred" -> hard_weak_trigger='occurred'
- [Easy] trigger="become" -> hard_weak_trigger='become'
- [Easy] trigger="Battle" -> answer_phrase_fail: partial extraction (no clause boundary found); soft_weak_trigger='Battle' with no valid phrase
- [Easy] trigger="ended" -> hard_weak_trigger='ended'
- [Easy] trigger="took place" -> hard_weak_trigger='took place'
- [Medium] trigger="took place" -> hard_weak_trigger='took place'
- [Hard] trigger="took place" -> hard_weak_trigger='took place'
- [Easy] trigger="seen" -> hard_weak_trigger='seen'
- [Medium] trigger="seen" -> hard_weak_trigger='seen'
- [Medium] trigger="approached" -> answer_phrase_fail: partial extraction: phrase ends with dangling word 'in'
- [Easy] trigger="called" -> hard_weak_trigger='called'; answer_phrase_fail: partial extraction: unclosed bracket or quote
- [Easy] trigger="brought" -> answer_phrase_fail: partial extraction: unclosed bracket or quote
- [Easy] trigger="brought" -> answer_phrase_fail: partial extraction: unclosed bracket or quote
- [Hard] trigger="orders" -> answer_phrase_fail: partial extraction: unclosed bracket or quote [risk: temporal_only_hard]
- [Easy] trigger="controversy" -> answer_phrase_fail: partial extraction: phrase ends with dangling word 'over'
- [Medium] trigger="controversy" -> answer_phrase_fail: partial extraction: phrase ends with dangling word 'over'
- [Medium] trigger="controversy" -> answer_phrase_fail: partial extraction: phrase ends with dangling word 'over'
- [Easy] trigger="made" -> hard_weak_trigger='made'
- [Hard] trigger="made" -> hard_weak_trigger='made'
- [Medium] trigger="made" -> hard_weak_trigger='made'
- [Easy] trigger="held" -> answer_phrase_fail: partial extraction (no clause boundary found); soft_weak_trigger='held' with no valid phrase
- [Medium] trigger="held" -> answer_phrase_fail: partial extraction (no clause boundary found); soft_weak_trigger='held' with no valid phrase