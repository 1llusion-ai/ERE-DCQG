# LLM Path Judge Pilot Report

**Model:** gpt-4o-mini
**Judged:** 15 / input 3890
**Parse OK:** 15 (100.0%)
**Kept:** 10 (66.7%)

## Overall Distributions

### path_questionable_distribution

| Label | Count |
|---|---:|
| partial | 7 |
| yes | 8 |

### expected_required_steps_distribution

| Label | Count |
|---|---:|
| 2 | 5 |
| 1 | 5 |
| 3+ | 5 |

### single_sentence_risk_distribution

| Label | Count |
|---|---:|
| high | 15 |

### recommended_difficulty_distribution

| Label | Count |
|---|---:|
| medium | 8 |
| easy | 3 |
| hard | 4 |

## Per-Level Summary

| Level | Total | Kept | Kept% | Diff Agree% | High Single-Sent Risk% | Recommended Difficulty | Expected Steps |
|---|---:|---:|---:|---:|---:|---|---|
| Easy | 5 | 5 | 100.0% | 60.0% | 100.0% | easy:3, medium:2 | 1:5 |
| Medium | 5 | 5 | 100.0% | 100.0% | 100.0% | medium:5 | 2:5 |
| Hard | 5 | 0 | 0.0% | 80.0% | 100.0% | hard:4, medium:1 | 3+:5 |

## Examples: not_kept

- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The proposed answer phrase is truncated and does not provide a complete answer."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The proposed answer phrase is truncated and does not form a complete answer, making it less suitable for clear question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The proposed answer phrase is somewhat truncated and the context requires multiple steps to fully understand the final event."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> completing/Process_end -> started/Process_start -> interrupted/Hindering
  - Answer phrase: but was interrupted by John's and Billy
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The proposed answer phrase is truncated and does not provide a complete answer, making it less clear for question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Cherry Valley massacre
  - Path: committed/Commitment -> took place/Process_start -> minimize/Cause_change_of_position_on_a_scale -> drove/Motion
  - Answer phrase: Expedition which drove the Iroquois out of
  - Judge: {"path_questionable": "yes", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The final event can be clearly questioned and is supported by the context, but it requires understanding multiple steps."}
  - Keep reason: hard_single_sentence_risk=high

## Examples: disagreement

- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The proposed answer phrase is truncated and does not form a complete answer, making it less suitable for clear question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Easy] Borrowed Heaven tour
  - Path: promoted/Creating -> announced/Expressing_publicly
  - Answer phrase: Caroline Corr announced her pregnancy and was
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The proposed answer phrase is truncated and does not provide a complete answer."}
  - Keep reason: keep
- [Easy] Operation Upshot–Knothole
  - Path: exercise/Practice -> employed/Employment
  - Answer phrase: test) was employed as primary for the
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The proposed answer phrase is truncated and does not form a complete answer."}
  - Keep reason: keep

## Examples: hard_high_risk

- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The proposed answer phrase is truncated and does not provide a complete answer."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The proposed answer phrase is truncated and does not form a complete answer, making it less suitable for clear question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The proposed answer phrase is somewhat truncated and the context requires multiple steps to fully understand the final event."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> completing/Process_end -> started/Process_start -> interrupted/Hindering
  - Answer phrase: but was interrupted by John's and Billy
  - Judge: {"path_questionable": "partial", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The proposed answer phrase is truncated and does not provide a complete answer, making it less clear for question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Cherry Valley massacre
  - Path: committed/Commitment -> took place/Process_start -> minimize/Cause_change_of_position_on_a_scale -> drove/Motion
  - Answer phrase: Expedition which drove the Iroquois out of
  - Judge: {"path_questionable": "yes", "expected_required_steps": "3+", "single_sentence_risk": "high", "recommended_difficulty": "hard", "reason": "The final event can be clearly questioned and is supported by the context, but it requires understanding multiple steps."}
  - Keep reason: hard_single_sentence_risk=high