# Evidence Necessity Audit Report

Generated: 2026-05-10 11:36:15

## 1. Overview

| Metric | Count |
|---|---:|
| Total documents processed | 5 |
| Total candidates | 25 |
| Assessment OK | 25 |
| Assessment LLM error | 0 |

## 2. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 0 |
| Invalid spans | 0 |
| Contradictions fixed | 0 |
| Status = ok | 25 |
| Status = llm_error | 0 |

## 3. Potential vs Verified Difficulty

| Level | Potential | Verified |
|---|---:|---:|
| Easy | 3 (12.0%) | 3 (12.0%) |
| Medium | 15 (60.0%) | 20 (80.0%) |
| Hard | 7 (28.0%) | 2 (8.0%) |

**Potential Hard -> Verified Hard conversion:** 28.6% (2/7)

## 4. Ablation: answer_only_can_identify_answer

| Value | All | Potential Hard only |
|---|---:|---:|
| yes | 3 | 0 |
| partial | 0 | 0 |
| no | 22 | 7 |

## 5. Ablation: bridge_removal_effect

| Value | All | Potential Hard only |
|---|---:|---:|
| none | 14 | 0 |
| harder | 9 | 5 |
| ambiguous | 2 | 2 |
| unanswerable | 0 | 0 |

## 6. Ablation: necessity_type

| Type | All | Potential Hard only |
|---|---:|---:|
| answer_identification | 1 | 0 |
| background_context | 22 | 5 |
| disambiguation | 2 | 2 |

## 7. Hard Candidate Detail

| Metric | Count |
|---|---:|
| Potential Hard total | 7 |
| Potential Hard + alone_sufficient=no | 7 |
| Verified Hard total | 2 |
| Verified Hard + answer_only=no | 2 |

## 8. Distribution by Reasoning Operation

| Operation | All | Verified Hard |
|---|---:|---:|
| bridge | 25 | 2 |

## 9. Distribution by Answer Event Type

| Event Type | All | Verified Hard |
|---|---:|---:|
| Achieve | 1 | 0 |
| Agree_or_refuse_to_act | 1 | 0 |
| Assistance | 2 | 0 |
| Attack | 2 | 0 |
| Catastrophe | 1 | 0 |
| Cause_change_of_position_on_a_scale | 1 | 0 |
| Cause_to_amalgamate | 1 | 0 |
| Choosing | 1 | 0 |
| Conquering | 1 | 0 |
| Create_artwork | 1 | 0 |
| Criminal_investigation | 1 | 0 |
| Giving | 1 | 0 |
| Hindering | 1 | 1 |
| Hostile_encounter | 1 | 0 |
| Imposing_obligation | 1 | 0 |
| Process_end | 2 | 1 |
| Process_start | 1 | 0 |
| Releasing | 1 | 0 |
| Removing | 1 | 0 |
| Response | 1 | 0 |
| Self_motion | 1 | 0 |
| Statement | 1 | 0 |

## 10. Answer Locality / num_required / evidence_necessity

| Locality | Count | Pct |
|---|---:|---:|
| single_sentence | 3 | 12.0% |
| two_sentence | 15 | 60.0% |
| multi_sentence | 7 | 28.0% |

| # Required | Count | Pct |
|---|---:|---:|
| 1 | 3 | 12.0% |
| 2 | 15 | 60.0% |
| 3 | 2 | 8.0% |
| 4 | 3 | 12.0% |
| 5 | 1 | 4.0% |
| 6 | 1 | 4.0% |

| Necessity | Count | Pct |
|---|---:|---:|
| weak | 1 | 4.0% |
| partial | 0 | 0.0% |
| strong | 24 | 96.0% |

## 11. Verified Hard Trace Examples (up to 10)

### Example 1

**Document:** Cherry Valley massacre (doc_id=c0c67db40cd5e2e03645ff1116fafcfc)
**Answer trigger:** "restrain" (Hindering)
**Answer phrase:** "he was powerless to restrain the Seneca"
**Answer sentence [S8]:** Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.

**Ablation assessment:**
- potential_evidence_difficulty: Hard
- verified_evidence_difficulty: Hard
- answer_only_can_identify_answer: no
- anchor_answer_can_identify_answer: partial
- full_evidence_can_identify_answer: yes
- bridge_removal_effect: ambiguous
- necessity_type: disambiguation
- ablation_reason: Without S7, the context of Butler's authority over the Indians is unclear, making it ambiguous who 'he' refers to.

**Evidence structure:**
- anchor_sentence_ids: [4]
- bridge_sentence_ids: [7]
- evidence_span: [4, 7, 8]
- num_required_sentences: 3
- reasoning_operation: bridge

**Anchor sentences:**
- [S4] The raiders were under the overall command of Walter Butler, who exercised little authority over the Indians on the expedition.

**Bridge sentences:**
- [S7] Butler's authority with the Indians was undermined by his poor treatment of Joseph Brant, the leader of the Mohawks.

---

### Example 2

**Document:** United States occupation of Nicaragua (doc_id=f28bce270df5a122c09365002d247e76)
**Answer trigger:** "ended" (Process_end)
**Answer phrase:** "invoking his new Good Neighbor policy ended American intervention"
**Answer sentence [S5]:** Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention.

**Ablation assessment:**
- potential_evidence_difficulty: Hard
- verified_evidence_difficulty: Hard
- answer_only_can_identify_answer: no
- anchor_answer_can_identify_answer: partial
- full_evidence_can_identify_answer: yes
- bridge_removal_effect: ambiguous
- necessity_type: disambiguation
- ablation_reason: Without S4, it is unclear who ended the American intervention.

**Evidence structure:**
- anchor_sentence_ids: [0]
- bridge_sentence_ids: [4]
- evidence_span: [0, 4, 5]
- num_required_sentences: 3
- reasoning_operation: bridge

**Anchor sentences:**
- [S0] The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.

**Bridge sentences:**
- [S4] President Herbert Hoover (1929–1933) opposed the relationship.

---

## 12. Success Criteria Check

- Potential Hard >= 100: **FAIL** (7)
- Verified Hard >= 100: **FAIL** (2)
- Verified Hard + answer_only=no: **NEEDS CHECK** (2)
- Potential->Verified conversion: 28.6%

**Conclusion:** Too few Hard evidence candidates. Need to change data construction method or answer types.
