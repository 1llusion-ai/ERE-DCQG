# LLM Path Judge Pilot Report

**Model:** gpt-4o
**Judged:** 15 / input 3890
**Parse OK:** 15 (100.0%)
**Kept:** 10 (66.7%)

## Overall Distributions

### path_questionable_distribution

| Label | Count |
|---|---:|
| partial | 15 |

### expected_required_steps_distribution

| Label | Count |
|---|---:|
| 1 | 12 |
| 2 | 3 |

### single_sentence_risk_distribution

| Label | Count |
|---|---:|
| high | 15 |

### recommended_difficulty_distribution

| Label | Count |
|---|---:|
| easy | 11 |
| medium | 4 |

## Per-Level Summary

| Level | Total | Kept | Kept% | Diff Agree% | High Single-Sent Risk% | Recommended Difficulty | Expected Steps |
|---|---:|---:|---:|---:|---:|---|---|
| Easy | 5 | 5 | 100.0% | 80.0% | 100.0% | easy:4, medium:1 | 1:5 |
| Medium | 5 | 5 | 100.0% | 0.0% | 100.0% | easy:5 | 1:5 |
| Hard | 5 | 0 | 0.0% | 0.0% | 100.0% | easy:2, medium:3 | 1:2, 2:3 |

## Examples: not_kept

- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event's answer phrase is clear but can be directly obtained from the answer sentence alone, making the path weak for multi-step question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event ('rise of the ghazis') is answerable from the answer sentence alone, but the path provides some additional context. The proposed answer phrase is truncated and unnatural, reducing clarity."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event is answerable from the answer sentence alone, making the single_sentence_risk high. While the path connects multiple events, the phrasing of the proposed answer phrase is somewhat noisy and could be clearer. The question would likely require connecting two events but does not fully justify a 'hard' difficulty level."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> completing/Process_end -> started/Process_start -> interrupted/Hindering
  - Answer phrase: but was interrupted by John's and Billy
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event ('interrupted') and its explanation are fully contained within a single sentence (S5). While the path includes multiple events, they do not contribute significantly to forming a natural, multi-step question. The proposed answer phrase is also slightly truncated and awkward."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Cherry Valley massacre
  - Path: committed/Commitment -> took place/Process_start -> minimize/Cause_change_of_position_on_a_scale -> drove/Motion
  - Answer phrase: Expedition which drove the Iroquois out of
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event is answerable from the answer sentence alone, but the path provides some additional context that could support a slightly more complex question."}
  - Keep reason: hard_single_sentence_risk=high

## Examples: disagreement

- [Medium] Cherry Valley massacre
  - Path: restrain/Hindering -> took place/Process_start -> described/Statement
  - Answer phrase: has been described as one of the
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event's answer phrase is truncated and unnatural ('has been described as one of the'), and the answer can be obtained directly from the answer sentence without requiring the full event path."}
  - Keep reason: keep
- [Medium] Photographer of Dreams
  - Path: given/Giving -> decided/Deciding -> flashed/Motion
  - Answer phrase: Santa Barbara", flashed the endless waters of
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event is answerable directly from the answer sentence alone, but the proposed answer phrase is truncated and unnatural, making the path weak for question generation."}
  - Keep reason: keep
- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event's answer phrase is clear but can be directly obtained from the answer sentence alone, making the path weak for multi-step question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event ('rise of the ghazis') is answerable from the answer sentence alone, but the path provides some additional context. The proposed answer phrase is truncated and unnatural, reducing clarity."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event is answerable from the answer sentence alone, making the single_sentence_risk high. While the path connects multiple events, the phrasing of the proposed answer phrase is somewhat noisy and could be clearer. The question would likely require connecting two events but does not fully justify a 'hard' difficulty level."}
  - Keep reason: hard_single_sentence_risk=high

## Examples: hard_high_risk

- [Hard] Operation Vengeance
  - Path: Vengeance/Revenge -> killed/Killing -> blamed/Judgment_communication -> claimed/Statement
  - Answer phrase: U.S. pilots claimed to have shot down
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event's answer phrase is clear but can be directly obtained from the answer sentence alone, making the path weak for multi-step question generation."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Byzantine–Seljuq wars
  - Path: practiced/Practice -> control/Control -> taken over/Conquering -> rise/Cause_change_of_position_on_a_scale
  - Answer phrase: to the rise of the ghazis and
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event ('rise of the ghazis') is answerable from the answer sentence alone, but the path provides some additional context. The proposed answer phrase is truncated and unnatural, reducing clarity."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> ended/Process_end -> returned/Arriving -> ending/Process_end
  - Answer phrase: returning and ending the tour in Europe
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event is answerable from the answer sentence alone, making the single_sentence_risk high. While the path connects multiple events, the phrasing of the proposed answer phrase is somewhat noisy and could be clearer. The question would likely require connecting two events but does not fully justify a 'hard' difficulty level."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Big Picture Tour
  - Path: started/Process_start -> completing/Process_end -> started/Process_start -> interrupted/Hindering
  - Answer phrase: but was interrupted by John's and Billy
  - Judge: {"path_questionable": "partial", "expected_required_steps": "1", "single_sentence_risk": "high", "recommended_difficulty": "easy", "reason": "The final event ('interrupted') and its explanation are fully contained within a single sentence (S5). While the path includes multiple events, they do not contribute significantly to forming a natural, multi-step question. The proposed answer phrase is also slightly truncated and awkward."}
  - Keep reason: hard_single_sentence_risk=high
- [Hard] Cherry Valley massacre
  - Path: committed/Commitment -> took place/Process_start -> minimize/Cause_change_of_position_on_a_scale -> drove/Motion
  - Answer phrase: Expedition which drove the Iroquois out of
  - Judge: {"path_questionable": "partial", "expected_required_steps": "2", "single_sentence_risk": "high", "recommended_difficulty": "medium", "reason": "The final event is answerable from the answer sentence alone, but the path provides some additional context that could support a slightly more complex question."}
  - Keep reason: hard_single_sentence_risk=high