# LLM Path Judge Pilot Report

**Model:** gpt-4o-mini
**Judged:** 15 / input 3890
**Parse OK:** 15 (100.0%)
**Kept:** 14 (93.3%)

## Overall Distributions

### path_questionable_distribution

| Label | Count |
|---|---:|
| yes | 13 |
| no | 1 |
| partial | 1 |

### expected_required_steps_distribution

| Label | Count |
|---|---:|
| 1 | 15 |

### single_sentence_risk_distribution

| Label | Count |
|---|---:|
| high | 4 |
| low | 10 |
| medium | 1 |

### recommended_difficulty_distribution

| Label | Count |
|---|---:|
| easy | 14 |
| medium | 1 |

## Per-Level Summary

| Level | Total | Kept | Kept% | Diff Agree% | High Single-Sent Risk% | Recommended Difficulty | Expected Steps |
|---|---:|---:|---:|---:|---:|---|---|
| Easy | 15 | 14 | 93.3% | 93.3% | 26.7% | easy:14, medium:1 | 1:15 |
| Medium | 0 | 0 | 0.0% | 0.0% | 0.0% | - | - |
| Hard | 0 | 0 | 0.0% | 0.0% | 0.0% | - | - |

## Examples: not_kept

- [Easy] Cherry Valley massacre
  - Path: ordered/Arranging -> took place/Process_start
  - Answer phrase: atrocities that took place at Cherry Valley
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The event path includes a weak trigger 'took place' which does not support a natural question about the final event."}
  - Keep reason: path_questionable=no

## Examples: disagreement

- [Easy] Cherry Valley massacre
  - Path: permitted/Preventing_or_letting -> took place/Process_start
  - Answer phrase: atrocities that took place at Cherry Valley
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "medium", "recommended_difficulty": "medium", "reason": "The path can support a question about the final event, but the presence of the phrase 'took place' introduces some ambiguity that may require clarification."}
  - Keep reason: keep

## Examples: hard_high_risk

(none)
