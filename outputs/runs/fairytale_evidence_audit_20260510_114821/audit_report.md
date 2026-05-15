# FairytaleQA Evidence Audit Report

Generated: 2026-05-10 11:51:00

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

### 3a. local-or-sum × difficulty

| local-or-sum | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| local | 34 | 7 | 2 | 43 |
| summary | 1 | 5 | 1 | 7 |

### 3b. ex-or-im × difficulty

| ex-or-im | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| explicit | 35 | 8 | 1 | 44 |
| implicit | 0 | 4 | 2 | 6 |

### 3c. attribute × difficulty

| attribute | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| action | 11 | 3 | 0 | 14 |
| causal relationship | 9 | 2 | 2 | 13 |
| character | 3 | 0 | 0 | 3 |
| feeling | 3 | 2 | 0 | 5 |
| outcome resolution | 4 | 1 | 0 | 5 |
| prediction | 1 | 4 | 1 | 6 |
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
| motivation | 2 |

### Hard by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 1 |
| motivation_bridge | 2 |

## 5. Answer Sufficiency Diagnostics

### answer_sentence_alone_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 35 | 70.0% |
| partial | 0 | 0.0% |
| no | 15 | 30.0% |

### section_evidence_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 50 | 100.0% |
| partial | 0 | 0.0% |
| no | 0 | 0.0% |

### bridge_removal_effect

| Value | Count | Pct |
|---|---:|---:|
| none | 34 | 68.0% |
| harder | 3 | 6.0% |
| ambiguous | 12 | 24.0% |
| unanswerable | 1 | 2.0% |

## 6. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 0 |
| Contradictions fixed | 1 |
| Status = ok | 50 |
| Status = llm_error | 0 |

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
- evidence_necessity_reason: S4 and S5 provide the actions, but S7, S8, and S9 explain the motivation and decision-making process of the boy.

**Required evidence sentences:**
- [S4] there were no neighbors , for the cottage stood all alone by the river , so the child was obliged to support his mother in his arms as best he could while she crawled painfully back to the cottage
- [S7] " can i leave you alone while i go for the doctor , mamma ? " he asked , anxiously , as he held her clasped hands tightly in his two little ones [BRIDGE]
- [S8] his mother drew him towards her and kissed him . [BRIDGE]
- [S9] (out of range) [BRIDGE]
- [S10] (out of range)

---

### Example 2

**Story:** old-dschang
**Question:** why didn't the match-maker want old dschang to marry sir we's daughter ?
**Answer:** he was a poor old gardener .
**Answer2:** he is too old for a beautiful daughter .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: motivation
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S1 introduces the match-maker's response, S2 and S3 provide the reasons.

**Required evidence sentences:**
- [S1] then the old match - maker said : " you do not know what you wish ! why should a gentleman 's beautiful daughter condescend to marry a poor old gardener like yourself ? even though you had money to burn , your white hair would not match her black locks [BRIDGE]
- [S2] such a marriage is out of the question ! " but old dschang did not cease to entreat her : " make an attempt , just one attempt , to mention me ! if they will not listen to you , then i must resign myself to my fate ! "
- [S3] (out of range)

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
- num_required_sentences: 4
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S0-S2 provide the context for Sir We's decision, and S3 shows the outcome.

**Required evidence sentences:**
- [S0] once an aristocratic relative visited sir we and said : " if you had really been poor , were there not enough young gentlemen in the neighborhood for your daughter ? why did you have to marry her to such a wrinkled old gardener ? now that you have thrown her away , so to speak , it would be better if both of them left this part of the country
- [S1] " then sir we prepared a banquet and invited his daughter and old dschang to visit him [BRIDGE]
- [S2] when they had had sufficient to eat and drink he allowed them to get an inkling of what was in his mind . [BRIDGE]
- [S3] (out of range)

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
