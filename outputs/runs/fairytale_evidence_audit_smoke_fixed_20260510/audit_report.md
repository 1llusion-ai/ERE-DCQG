# FairytaleQA Evidence Audit Report

Generated: 2026-05-10 12:01:25

## 1. Dataset Loading Summary

| Field | Value |
|---|---|
| Split | validation |
| Total QA pairs loaded | 50 |
| QA pairs assessed | 50 |
| Fields available | story_name, story_section, question, answer1, answer2, local_or_sum, attribute, ex_or_im, ex_or_im2, split |
| Source | HuggingFace |

## 2. Evidence Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 35 | 70.0% |
| Medium | 12 | 24.0% |
| Hard | 3 | 6.0% |

## 3. Fairytale Labels vs Evidence Difficulty

### 3a. local-or-sum x difficulty

| local-or-sum | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| local | 33 | 9 | 1 | 43 |
| summary | 2 | 3 | 2 | 7 |

### 3b. ex-or-im x difficulty

| ex-or-im | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| explicit | 33 | 9 | 2 | 44 |
| implicit | 2 | 3 | 1 | 6 |

### 3c. attribute x difficulty

| attribute | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| action | 10 | 4 | 0 | 14 |
| causal relationship | 10 | 2 | 1 | 13 |
| character | 3 | 0 | 0 | 3 |
| feeling | 2 | 3 | 0 | 5 |
| outcome resolution | 4 | 1 | 0 | 5 |
| prediction | 2 | 2 | 2 | 6 |
| setting | 4 | 0 | 0 | 4 |

## 4. Verified Hard Detail

| Metric | Count |
|---|---:|
| Hard count | 3 |
| Hard rate | 6.0% |

### Hard by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 1 |
| motivation | 1 |
| summary_inference | 1 |

### Hard by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 1 |
| motivation_bridge | 1 |
| summary_synthesis | 1 |

## 5. Answer Sufficiency Diagnostics

### answer_sentence_alone_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 33 | 66.0% |
| partial | 2 | 4.0% |
| no | 15 | 30.0% |

### section_evidence_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 46 | 92.0% |
| partial | 4 | 8.0% |
| no | 0 | 0.0% |

### bridge_removal_effect

| Value | Count | Pct |
|---|---:|---:|
| none | 29 | 58.0% |
| harder | 11 | 22.0% |
| ambiguous | 9 | 18.0% |
| unanswerable | 1 | 2.0% |

## 6. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 2 |
| Contradictions fixed | 2 |
| Missing trace fields | 0 |
| Invalid required sentence IDs | 0 |
| Hard validation violations | 0 |
| Status = ok | 48 |
| Status = llm_error | 2 |

## 7. Verified Hard Examples (up to 10)

### Example 1

**Story:** little-boy-blue
**Question:** what will the boy do to help his hurt mother ?
**Answer:** take the boat and fetch the doctor from the village .
**Answer2:** ran to the cottage for water and bathed the poor woman 's face , and raised her head that she might drink .
**Labels:** local-or-sum=summary, attribute=prediction, ex-or-im=explicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 5
- reasoning_operation: motivation
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S4 and S5 provide the actions taken by the boy, but S7, S8, and S9 explain his motivation and decision-making process.

**Required evidence sentences:**
- [S4] little boy blue ran to the cottage for water and bathed the poor woman 's face , and raised her head that she might drink .
- [S7] then little boy blue began to think what he should do next . [BRIDGE]
- [S8] " can i leave you alone while i go for the doctor , mamma ? [BRIDGE]
- [S9] " he asked , anxiously , as he held her clasped hands tightly in his two little ones . [BRIDGE]
- [S10] his mother drew him towards her and kissed him .

---

### Example 2

**Story:** old-dschang
**Question:** how will sir we feel when the match-maker recommends old dschang to marry his daughter ?
**Answer:** angry .
**Answer2:** angry .
**Labels:** local-or-sum=summary, attribute=prediction, ex-or-im=explicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: partial
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S7, S8, and S10 together provide the full context of Sir We's reaction, but S7 is necessary to understand the sequence of events.

**Required evidence sentences:**
- [S7] " the old match - maker had taken his money , so she could not well refuse , and though she feared being scolded , she mentioned him to sir we . [BRIDGE]
- [S8] he grew angry and wanted to throw her out of the house .
- [S10] " " tell the old man that if this very day he brings me two white jade - stones , and four hundred ounces of yellow gold , then i will give him my daughter 's hand in marriage .

---

### Example 3

**Story:** old-dschang
**Question:** why did sir we tell old dschang and his wife what was in his mind ?
**Answer:** an aristocratic relative told him that old dschang and his wife should leave this part of the country .
**Answer2:** a relative made him sir we feel ashamed and told him " it would be bettwe if both of them left this part of the country " .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S1 and S3 provide the reason for Sir We's actions, while S0 introduces the context of the aristocratic relative's visit.

**Required evidence sentences:**
- [S0] once an aristocratic relative visited sir we and said : " if you had really been poor , were there not enough young gentlemen in the neighborhood for your daughter ?
- [S1] why did you have to marry her to such a wrinkled old gardener ? [BRIDGE]
- [S3] " then sir we prepared a banquet and invited his daughter and old dschang to visit him .

---

## 8. Comparison Note Against MAVEN-ERE

**MAVEN-ERE baseline (current):**
- QG-eligible Hard rescue: 0/112 blind Hard among judged candidates
- Quality-pass candidates: 0/21 blind Hard
- Root cause: answer sentences are locally identifiable; event-path hop count does not translate to answering difficulty

**FairytaleQA evidence audit (50 QA pairs):**
- Easy: 35 (70.0%)
- Medium: 12 (24.0%)
- Hard: 3 (6.0%)

**Assessment:** FairytaleQA produces 3 verified Hard candidates (6.0%) from 50 QA pairs. This is a meaningful improvement over MAVEN-ERE's 0% blind Hard rate. Narrative QA appears more promising for difficulty-controlled QG because answers often require understanding character motivation, causal chains, and multi-sentence inference rather than local event-phrase extraction.
