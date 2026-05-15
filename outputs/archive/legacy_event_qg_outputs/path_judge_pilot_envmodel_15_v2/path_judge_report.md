# LLM Path Judge Pilot Report

**Model:** gpt-4o-free
**Judged:** 15 / input 3890
**Parse OK:** 0 (0.0%)
**Kept:** 0 (0.0%)

## Overall Distributions

### path_questionable_distribution

| Label | Count |
|---|---:|
| no | 15 |

### expected_required_steps_distribution

| Label | Count |
|---|---:|
| 1 | 15 |

### single_sentence_risk_distribution

| Label | Count |
|---|---:|
| high | 15 |

### recommended_difficulty_distribution

| Label | Count |
|---|---:|
| easy | 15 |

## Per-Level Summary

| Level | Total | Kept | Kept% | Diff Agree% | High Single-Sent Risk% | Recommended Difficulty | Expected Steps |
|---|---:|---:|---:|---:|---:|---|---|
| Easy | 5 | 0 | 0.0% | 100.0% | 100.0% | easy:5 | 1:5 |
| Medium | 5 | 0 | 0.0% | 0.0% | 100.0% | easy:5 | 1:5 |
| Hard | 5 | 0 | 0.0% | 0.0% | 100.0% | easy:5 | 1:5 |

## Examples: not_kept

- [Medium] Cherry Valley massacre
  - Path: restrain/Hindering -> took place/Process_start -> described/Statement
  - Answer phrase: has been described as one of the
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 400: Bad Request"}
  - Keep reason: path_questionable=no
- [Easy] 2008 IIHF World Championship
  - Path: formalize/Institutionalization -> changes/Change
  - Answer phrase: were two changes in the format compared
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 400: Bad Request"}
  - Keep reason: path_questionable=no
- [Medium] Photographer of Dreams
  - Path: given/Giving -> decided/Deciding -> flashed/Motion
  - Answer phrase: Santa Barbara", flashed the endless waters of
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no
- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high

## Examples: disagreement

- [Medium] Cherry Valley massacre
  - Path: restrain/Hindering -> took place/Process_start -> described/Statement
  - Answer phrase: has been described as one of the
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 400: Bad Request"}
  - Keep reason: path_questionable=no
- [Medium] Photographer of Dreams
  - Path: given/Giving -> decided/Deciding -> flashed/Motion
  - Answer phrase: Santa Barbara", flashed the endless waters of
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no
- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high

## Examples: hard_high_risk

- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> completing/Process_end -> started/Process_start -> interrupted/Hindering
  - Answer phrase: but was interrupted by John's and Billy
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high
- [Hard] Cherry Valley massacre
  - Path: committed/Commitment -> took place/Process_start -> minimize/Cause_change_of_position_on_a_scale -> drove/Motion
  - Answer phrase: Expedition which drove the Iroquois out of
  - Judge: {"path_questionable": "no", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "judge_error: HTTPError: HTTP Error 429: Too Many Requests"}
  - Keep reason: path_questionable=no; hard_single_sentence_risk=high