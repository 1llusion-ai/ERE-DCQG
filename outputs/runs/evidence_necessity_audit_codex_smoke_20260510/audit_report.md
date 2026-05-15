# Evidence Necessity Audit Report

Generated: 2026-05-10 02:24:02

## 1. Overview

| Metric | Count |
|---|---:|
| Total documents processed | 1 |
| Total candidates | 3 |
| Assessment OK | 3 |
| Assessment LLM error | 0 |

## 2. Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 1 | 33.3% |
| Medium | 1 | 33.3% |
| Hard | 1 | 33.3% |

## 3. Hard Candidate Detail

| Metric | Count |
|---|---:|
| Hard candidates total | 1 |
| Hard with answer_sentence_alone_sufficient=no | 1 |

## 4. Distribution by Reasoning Operation

| Operation | All | Hard-only |
|---|---:|---:|
| bridge | 2 | 0 |
| disambiguation | 1 | 1 |

## 5. Distribution by Answer Event Type

| Event Type | All | Hard-only |
|---|---:|---:|
| Cause_change_of_position_on_a_scale | 1 | 0 |
| Hindering | 1 | 1 |
| Statement | 1 | 0 |

## 6. Distribution by Answer Locality

| Locality | Count | Pct |
|---|---:|---:|
| single_sentence | 1 | 33.3% |
| two_sentence | 1 | 33.3% |
| multi_sentence | 1 | 33.3% |

## 7. Distribution by num_required_sentences

| # Required | Count | Pct |
|---|---:|---:|
| 1 | 1 | 33.3% |
| 2 | 1 | 33.3% |
| 3 | 1 | 33.3% |

## 8. Distribution by Evidence Necessity

| Necessity | Count | Pct |
|---|---:|---:|
| weak | 0 | 0.0% |
| partial | 0 | 0.0% |
| strong | 3 | 100.0% |

## 9. Hard Candidate Trace Examples (up to 10)

### Example 1

**Document:** Cherry Valley massacre (doc_id=c0c67db40cd5e2e03645ff1116fafcfc)
**Answer trigger:** "restrain" (Hindering)
**Answer phrase:** "he was powerless to restrain the Seneca"
**Answer sentence [S8]:** Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [4]
- bridge_sentence_ids: [7]
- evidence_span: [4, 7, 8]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
The answer sentence alone does not provide context about Butler's command or his relationship with the Seneca. Sentence S4 introduces Butler's role and lack of authority, while S7 explains the reason for his weakened authority, both necessary to understand the context of the answer.

**Anchor sentences:**
- [S4] The raiders were under the overall command of Walter Butler, who exercised little authority over the Indians on the expedition.

**Bridge sentences:**
- [S7] Butler's authority with the Indians was undermined by his poor treatment of Joseph Brant, the leader of the Mohawks.

---

## 10. Success Criteria Check

- Hard evidence candidates >= 100: **FAIL** (1)
- Hard with answer_sentence_alone_sufficient=no: **NEEDS CHECK** (1)

**Conclusion:** Too few Hard evidence candidates. Need to change data construction method or answer types, rather than continuing to adjust prompts.
