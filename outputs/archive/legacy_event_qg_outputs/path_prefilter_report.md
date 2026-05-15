# Path Prefilter Report

**Total paths:** 145
**Passed:** 116 (80.0%)

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 50 | 38 | 76.0% |
| Medium | 49 | 39 | 79.6% |
| Hard | 46 | 39 | 84.8% |

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| none | 131 |
| needs_phrase | 13 |
| hard_blacklist | 1 |

Examples:

- [Hard] trigger="control" → weak trigger but valid phrase: 'In line with the agreement with Johor in 1606, the Dutch took control of Malacca'
- [Medium] trigger="operate" → weak trigger but valid phrase: 'could operate in environments other than a major force on force engagement on the plains of Central Europe'
- [Hard] trigger="operate" → weak trigger but valid phrase: 'could operate in environments other than a major force on force engagement on the plains of Central Europe'
- [Easy] trigger="took place" → hard_blacklisted trigger: 'took place'

## Answer Phrase

- Pass rate: 82.8%

Failures:

- [Easy] trigger="given" phrase="given" → phrase equals trigger
- [Easy] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)
- [Easy] trigger="designed" phrase="American military interventions in Nicaragua were designed to stop any other nation except the United States of America from building a Nicaraguan Canal" → partial extraction (no clause boundary found)
- [Easy] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)
- [Medium] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)

## Relation Group by Difficulty

| Level | TEMPORAL | CAUSE | SUBEVENT | MIXED | NONE |
|-------|----------|-------|----------|-------|------|
| Easy | 44 | 6 | 0 | 0 | 0 |
| Medium | 38 | 11 | 0 | 0 | 0 |
| Hard | 36 | 10 | 0 | 0 | 0 |

- Temporal-only Hard: 36 (78.3%)

## Single-Sentence Risk

| Level | Low | Medium | High |
|-------|-----|--------|------|
| Easy | 42 | 2 | 6 |
| Medium | 47 | 1 | 1 |
| Hard | 45 | 0 | 1 |

## Failure Reasons

| Reason | Count |
|--------|-------|
| answer_phrase_fail: partial extraction (no clause boundary found) | 16 |
| hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found) | 6 |
| hard_weak_trigger='become' | 2 |
| hard_weak_trigger='ended' | 2 |
| answer_phrase_fail: phrase equals trigger | 1 |
| hard_weak_trigger='began'; answer_phrase_fail: partial extraction (no clause boundary found) | 1 |
| hard_weak_trigger='took place'; answer_phrase_fail: partial extraction (no clause boundary found) | 1 |

Examples:

- [Easy] trigger="given" → answer_phrase_fail: phrase equals trigger
- [Easy] trigger="ended" → hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="ended" → hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Medium] trigger="ended" → hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="designed" → answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="destroyed" → answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="combined" → answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="began" → hard_weak_trigger='began'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Easy] trigger="took place" → hard_weak_trigger='took place'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Medium] trigger="become" → hard_weak_trigger='become'
- [Hard] trigger="become" → hard_weak_trigger='become' [risk: temporal_only_hard]
- [Easy] trigger="ended" → hard_weak_trigger='ended'
- [Hard] trigger="ended" → hard_weak_trigger='ended'