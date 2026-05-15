# Path Prefilter Report

**Total paths:** 11
**Passed:** 7 (63.6%)

## Per-Level Pass Rate

| Level | Total | Passed | Pass Rate |
|-------|-------|--------|-----------|
| Easy | 4 | 3 | 75.0% |
| Medium | 4 | 2 | 50.0% |
| Hard | 3 | 2 | 66.7% |

## Weak Trigger Distribution

| Type | Count |
|------|-------|
| none | 11 |

## Answer Phrase

- Pass rate: 63.6%

Failures:

- [Easy] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)
- [Medium] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)
- [Medium] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)
- [Hard] trigger="ended" phrase="Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention" → partial extraction (no clause boundary found)

## Relation Group by Difficulty

| Level | TEMPORAL | CAUSE | SUBEVENT | MIXED | NONE |
|-------|----------|-------|----------|-------|------|
| Easy | 4 | 0 | 0 | 0 | 0 |
| Medium | 3 | 1 | 0 | 0 | 0 |
| Hard | 1 | 2 | 0 | 0 | 0 |

- Temporal-only Hard: 1 (33.3%)

## Single-Sentence Risk

| Level | Low | Medium | High |
|-------|-----|--------|------|
| Easy | 3 | 1 | 0 |
| Medium | 4 | 0 | 0 |
| Hard | 3 | 0 | 0 |

## Failure Reasons

| Reason | Count |
|--------|-------|
| hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found) | 4 |

Examples:

- [Easy] trigger="ended" → hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Medium] trigger="ended" → hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)
- [Medium] trigger="ended" → hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)