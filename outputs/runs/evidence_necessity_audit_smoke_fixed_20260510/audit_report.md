# Evidence Necessity Audit Report

Generated: 2026-05-10 02:29:34

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
| Parse failures (evidence_assessment_parse_ok=false) | 0 |
| Invalid spans (answer_sent_id not in evidence_span) | 0 |
| Contradictions fixed by validator | 2 |
| Assessment status = ok | 25 |
| Assessment status = llm_error | 0 |

## 3. Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 3 | 12.0% |
| Medium | 14 | 56.0% |
| Hard | 8 | 32.0% |

## 4. Hard Candidate Detail

| Metric | Count |
|---|---:|
| Hard candidates total | 8 |
| Hard with answer_sentence_alone_sufficient=no | 8 |

## 5. Distribution by Reasoning Operation

| Operation | All | Hard-only |
|---|---:|---:|
| bridge | 8 | 1 |
| disambiguation | 16 | 7 |
| temporal_order | 1 | 0 |

## 6. Distribution by Answer Event Type

| Event Type | All | Hard-only |
|---|---:|---:|
| Achieve | 1 | 0 |
| Agree_or_refuse_to_act | 1 | 0 |
| Assistance | 2 | 0 |
| Attack | 2 | 1 |
| Catastrophe | 1 | 1 |
| Cause_change_of_position_on_a_scale | 1 | 0 |
| Cause_to_amalgamate | 1 | 0 |
| Choosing | 1 | 0 |
| Conquering | 1 | 0 |
| Create_artwork | 1 | 0 |
| Criminal_investigation | 1 | 1 |
| Giving | 1 | 0 |
| Hindering | 1 | 1 |
| Hostile_encounter | 1 | 0 |
| Imposing_obligation | 1 | 1 |
| Process_end | 2 | 1 |
| Process_start | 1 | 0 |
| Releasing | 1 | 1 |
| Removing | 1 | 0 |
| Response | 1 | 0 |
| Self_motion | 1 | 1 |
| Statement | 1 | 0 |

## 7. Distribution by Answer Locality

| Locality | Count | Pct |
|---|---:|---:|
| single_sentence | 3 | 12.0% |
| two_sentence | 14 | 56.0% |
| multi_sentence | 8 | 32.0% |

## 8. Distribution by num_required_sentences

| # Required | Count | Pct |
|---|---:|---:|
| 1 | 3 | 12.0% |
| 2 | 14 | 56.0% |
| 3 | 8 | 32.0% |

## 9. Distribution by Evidence Necessity

| Necessity | Count | Pct |
|---|---:|---:|
| weak | 0 | 0.0% |
| partial | 1 | 4.0% |
| strong | 24 | 96.0% |

## 10. Hard Candidate Trace Examples (up to 10)

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
The answer sentence mentions Butler's inability to restrain the Seneca, but without S4 and S7, the reader cannot understand the context of Butler's authority and his relationship with the Indians.

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

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [0]
- bridge_sentence_ids: [4]
- evidence_span: [0, 4, 5]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
Sentence S0 provides the context of the occupation period, and S4 introduces the relevant president, Franklin D. Roosevelt, making the answer sentence more understandable.

**Anchor sentences:**
- [S0] The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.

**Bridge sentences:**
- [S4] President Herbert Hoover (1929–1933) opposed the relationship.

---

### Example 3

**Document:** Battle of Malacca (1641) (doc_id=3dcfd60153822a6a8f6a516f161fc506)
**Answer trigger:** "assaulted" (Attack)
**Answer phrase:** "The Dutch with their local allies assaulted"
**Answer sentence [S6]:** The Dutch with their local allies assaulted and wrested Malacca from the Portuguese in January 1641.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [1]
- bridge_sentence_ids: [5]
- evidence_span: [1, 5, 6]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
The answer sentence alone does not provide context about the Dutch and their allies. Sentence S1 introduces the Dutch campaign against the Portuguese, and S5 provides the context of the alliance with local forces, which is necessary to understand the answer.

**Anchor sentences:**
- [S1] In the early 17th century, the Dutch East India Company ("Verenigde Oostindische Compagnie", "VOC") began the campaign to destroy Portuguese power in the East.

**Bridge sentences:**
- [S5] Although the Dutch were routed, the Portuguese fleet of Martim Afonso de Castro, the Viceroy of Portuguese India, suffered heavier casualties and the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch.

---

### Example 4

**Document:** Battle of Malacca (1641) (doc_id=3dcfd60153822a6a8f6a516f161fc506)
**Answer trigger:** "casualties" (Catastrophe)
**Answer phrase:** "suffered heavier casualties and the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch"
**Answer sentence [S5]:** Although the Dutch were routed, the Portuguese fleet of Martim Afonso de Castro, the Viceroy of Portuguese India, suffered heavier casualties and the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: bridge
- evidence_necessity: strong
- anchor_sentence_ids: [1]
- bridge_sentence_ids: [4]
- evidence_span: [1, 4, 5]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
Sentence S1 introduces the Dutch campaign against the Portuguese, and S4 provides the context of the siege and naval battle, which is necessary to understand the casualties and alliance mentioned in the answer sentence.

**Anchor sentences:**
- [S1] In the early 17th century, the Dutch East India Company ("Verenigde Oostindische Compagnie", "VOC") began the campaign to destroy Portuguese power in the East.

**Bridge sentences:**
- [S4] The first serious attempt was the siege of Malacca in 1606 by the third VOC fleet from Holland with eleven ships, led by Admiral Cornelis Matelief de Jonge that lead to the naval battle of Cape Rachado.

---

### Example 5

**Document:** Myyrmanni bombing (doc_id=e253b7fd1109bd5f87966022eea7762f)
**Answer trigger:** "closed" (Self_motion)
**Answer phrase:** "closed in January 2003 without any indictments as Gerdt was the sole suspect"
**Answer sentence [S6]:** The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [1]
- bridge_sentence_ids: [5]
- evidence_span: [1, 5, 6]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
The answer sentence alone does not provide context about the incident or the suspect. Sentence S1 introduces the incident and the suspect, and S5 provides context about the crowded shopping center, which helps to understand the scale of the incident.

**Anchor sentences:**
- [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.

**Bridge sentences:**
- [S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.

---

### Example 6

**Document:** Myyrmanni bombing (doc_id=e253b7fd1109bd5f87966022eea7762f)
**Answer trigger:** "investigated" (Criminal_investigation)
**Answer phrase:** "The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect"
**Answer sentence [S6]:** The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [1]
- bridge_sentence_ids: [5]
- evidence_span: [1, 5, 6]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
The answer sentence alone does not provide context about the incident or the suspect. Sentence S1 introduces the incident and the suspect, and S5 provides context about the crowded shopping center, which helps to understand the scale of the incident.

**Anchor sentences:**
- [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.

**Bridge sentences:**
- [S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.

---

### Example 7

**Document:** Myyrmanni bombing (doc_id=e253b7fd1109bd5f87966022eea7762f)
**Answer trigger:** "released" (Releasing)
**Answer phrase:** "released at the scene"
**Answer sentence [S4]:** 66 victims required hospitalization with the remainder treated and released at the scene.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [0]
- bridge_sentence_ids: [3]
- evidence_span: [0, 3, 4]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
The answer sentence alone does not provide context about the incident or the number of injured. Sentence S0 provides the location and date of the incident, and S3 provides the number of injured, which helps to understand the scale of the incident.

**Anchor sentences:**
- [S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.

**Bridge sentences:**
- [S3] 166 people were injured, including 10 children.

---

### Example 8

**Document:** Myyrmanni bombing (doc_id=e253b7fd1109bd5f87966022eea7762f)
**Answer trigger:** "required" (Imposing_obligation)
**Answer phrase:** "66 victims required hospitalization with the remainder treated"
**Answer sentence [S4]:** 66 victims required hospitalization with the remainder treated and released at the scene.

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- answer_locality: multi_sentence
- reasoning_operation: disambiguation
- evidence_necessity: strong
- anchor_sentence_ids: [0]
- bridge_sentence_ids: [3]
- evidence_span: [0, 3, 4]
- num_required_sentences: 3

**Why answer sentence alone is insufficient:**
The answer sentence alone does not provide context about the incident or the number of injured. Sentence S0 provides the location and date of the incident, and S3 provides the number of injured, which helps to understand the scale of the incident.

**Anchor sentences:**
- [S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.

**Bridge sentences:**
- [S3] 166 people were injured, including 10 children.

---

## 11. Success Criteria Check

- Hard evidence candidates >= 100: **FAIL** (8)
- Hard with answer_sentence_alone_sufficient=no: **NEEDS CHECK** (8)

**Conclusion:** Too few Hard evidence candidates. Need to change data construction method or answer types, rather than continuing to adjust prompts.
