# FairytaleQA Evidence Audit Report

Generated: 2026-05-10 14:24:22

## 1. Dataset Loading Summary

| Field | Value |
|---|---|
| Split | validation |
| Total QA pairs loaded | 1025 |
| Pool before filter | 1025 |
| Pool after filter | 281 |
| Filter criteria | ex_or_im=implicit |
| QA pairs assessed | 200 |
| Fields available | story_name, story_section, question, answer1, answer2, local_or_sum, attribute, ex_or_im, ex_or_im2, split |
| Source | HuggingFace |

## 2. Evidence Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 56 | 28.0% |
| Medium | 115 | 57.5% |
| Hard | 29 | 14.5% |

## 3. Fairytale Labels vs Evidence Difficulty

### 3a. local-or-sum x difficulty

| local-or-sum | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| local | 53 | 84 | 15 | 152 |
| summary | 3 | 31 | 14 | 48 |

### 3b. ex-or-im x difficulty

| ex-or-im | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| implicit | 56 | 115 | 29 | 200 |

### 3c. attribute x difficulty

| attribute | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| action | 9 | 8 | 3 | 20 |
| causal relationship | 21 | 59 | 17 | 97 |
| character | 0 | 3 | 1 | 4 |
| feeling | 15 | 24 | 5 | 44 |
| outcome resolution | 5 | 5 | 0 | 10 |
| prediction | 6 | 16 | 2 | 24 |
| setting | 0 | 0 | 1 | 1 |

## 4. Verified Hard Detail

| Metric | Count |
|---|---:|
| Hard count | 29 |
| Hard rate | 14.5% |

### Hard by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 6 |
| character_state | 5 |
| motivation | 12 |
| summary_inference | 6 |

### Hard by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 6 |
| disambiguation | 1 |
| motivation_bridge | 14 |
| summary_synthesis | 8 |

### Hard by attribute

| Attribute | Count |
|---|---:|
| action | 3 |
| causal relationship | 17 |
| character | 1 |
| feeling | 5 |
| prediction | 2 |
| setting | 1 |

### Hard by question prefix

| Prefix | Count |
|---|---:|
| how did | 6 |
| what | 1 |
| what did | 2 |
| what will | 2 |
| where did | 1 |
| why did | 14 |
| why was | 2 |
| why were | 1 |

## 5. Answer Sufficiency Diagnostics

### answer_sentence_alone_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 56 | 28.0% |
| partial | 0 | 0.0% |
| no | 144 | 72.0% |

### section_evidence_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 184 | 92.0% |
| partial | 5 | 2.5% |
| no | 11 | 5.5% |

### bridge_removal_effect

| Value | Count | Pct |
|---|---:|---:|
| none | 33 | 16.5% |
| harder | 5 | 2.5% |
| ambiguous | 141 | 70.5% |
| unanswerable | 21 | 10.5% |

## 6. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 0 |
| Contradictions fixed | 36 |
| Missing trace fields | 0 |
| Invalid required sentence IDs | 0 |
| Hard validation violations | 0 |
| Status = ok | 200 |
| Status = llm_error | 0 |

## 7. Verified Hard Examples (up to 10)

### Example 1

**Story:** why-dog-and-cat-are-enemies
**Question:** why did the dog and the cat go hungry as well ?
**Answer:** they did not have money to buy food .
**Answer2:** the man and wife did not have any food to give them .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S2 explains the selling of the ring, S3 shows the consequence of poverty, and S4 directly states the animals' hunger, which is necessary to understand the causal relationship.

**Required evidence sentences:**
- [S2] but this they did not know , and hence sold the ring for a small sum .
- [S3] but no sooner was the ring gone than they began to grow poorer and poorer , and at last did not know when they would get their next meal . [BRIDGE]
- [S4] they had a dog and a cat , and these had to go hungry as well .

---

### Example 2

**Story:** why-dog-and-cat-are-enemies
**Question:** why were the cat and dog enemies ?
**Answer:** the cat took the credit for retrieving the ring .
**Answer2:** the cat took all the credit for returning the ring .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S11 states the outcome but S9 and S10 explain the cause. Without S9 and S10, the reader cannot determine why the cat and dog became enemies.

**Required evidence sentences:**
- [S9] " but when the dog came home they beat him and scolded him , because he had not helped to bring home the ring again . [BRIDGE]
- [S10] and the cat sat by the fireplace , purred and said never a word . [BRIDGE]
- [S11] then the dog grew angry at the cat , because she had robbed him of his reward , and when he saw her he chased her and tried to seize her .

---

### Example 3

**Story:** assipattle-and-the-mester-stoorworm
**Question:** how did assipattle feel when his brothers and mother mistreated him ?
**Answer:** upset .
**Answer2:** sad .
**Labels:** local-or-sum=local, attribute=feeling, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: character_state
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S2 summarizes Assipattle's feelings but S0 and S1 provide the context. Without S0 and S1, the reader cannot determine Assipattle's feelings.

**Required evidence sentences:**
- [S0] and his brothers , working hard in the fields , would point to him with mocking fingers , and laugh , and say to each other how well the name suited him , and of how little use he was in the world .
- [S1] and when they came home from their work , they would push him about and tease him , and even his mother would make him sweep the floor , and draw water from the well , and fetch peats from the peat - stack , and do all the little odd jobs that nobody else would do .
- [S2] so poor assipattle had rather a hard life of it , and he would often have been very miserable had it not been for his sister , who loved him dearly , and who would listen quite patiently to all the stories that he had to tell ; who never laughed at him or told him that he was telling lies , as his brothers did . [BRIDGE]

---

### Example 4

**Story:** assipattle-and-the-mester-stoorworm
**Question:** how did everyone feel about the mester stoorworm ?
**Answer:** scared .
**Answer2:** almost paralyzed with terror .
**Labels:** local-or-sum=local, attribute=feeling, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: character_state
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S0 states the outcome but S1 and S2 provide the context. Without S1 and S2, the reader cannot determine the feelings.

**Required evidence sentences:**
- [S0] as you may imagine , everyone was almost paralysed with terror at this awful calamity which threatened them ; and the king called a solemn meeting of all his counsellors , and asked them if they could devise any way of warding off the danger . [BRIDGE]
- [S1] and for three whole days they sat in council , these grave , bearded men , and many were the suggestions which were made , and many the words of wisdom which were spoken ; but , alas !
- [S2] no one was wise enough to think of a way by which the mester stoorworm might be driven back .

---

### Example 5

**Story:** assipattle-and-the-mester-stoorworm
**Question:** how did assipattle get out of the stoorworm's body ?
**Answer:** made the stoorworm sick .
**Answer2:** he cut a hole in the stoorworm 's liver and made it sick .
**Labels:** local-or-sum=summary, attribute=action, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 4
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S5 and S8 provide the answer, but S6 and S7 are needed to establish the causal chain of events leading to Assipattle's escape.

**Required evidence sentences:**
- [S5] presently he came to the huge creature 's liver , and having heard that the liver of a fish is full of oil , he made a hole in it and put in the live peat .
- [S6] woe 's me ! [BRIDGE]
- [S7] but there was a conflagration ! [BRIDGE]
- [S8] and assipattle just got back to his boat in time ; for the mester stoorworm , in its convulsions , threw the boat right out of its mouth again , and it was flung up , high and dry , on the bare land .

---

### Example 6

**Story:** the-corpse-watchers
**Question:** what kind of person was the eldest daughter ?
**Answer:** selfish .
**Answer2:** unkind .
**Labels:** local-or-sum=local, attribute=character, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: character_state
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S0 and S6 provide actions that indicate her character, but S2 is needed to understand her selfishness.

**Required evidence sentences:**
- [S0] there was once a poor woman that had three daughters , and one day the eldest said , " mother , bake my cake and kill my cock till i go seek my fortune .
- [S2] " " curse or no curse , " says she , " the whole is little enough . [BRIDGE]
- [S6] " the dickens a bit you 'll get from me , " says she ; " it 's all too little for myself .

---

### Example 7

**Story:** the-corpse-watchers
**Question:** what did the youngest do different from her two other sisters ?
**Answer:** the youngest took her mother 's blessing .
**Answer2:** she took her mom 's blessing and shared her food with the old woman .
**Labels:** local-or-sum=summary, attribute=action, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S14 and S15 provide the actions of the youngest, but S1 is needed to understand the contrast with her sisters.

**Required evidence sentences:**
- [S1] " so she did , and when all was ready , says her mother to her , " which will you have -- half of these with my blessing , or the whole with my curse ? [BRIDGE]
- [S14] at last the youngest went off in search of the other two , and she took care to carry her mother 's blessing with her .
- [S15] she shared her dinner with the poor woman on the road , and she told her that she would watch over her .

---

### Example 8

**Story:** the-corpse-watchers
**Question:** why did the first two daughters have misfortune ?
**Answer:** they did n't care for their mother 's blessing .
**Answer2:** they did not take their mother 's blessing or feed the old woman .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: motivation
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S1 and S2 show the eldest daughter's choice, S12 explains the second daughter's similar fate, and the answer requires understanding the motivation behind their misfortune.

**Required evidence sentences:**
- [S1] " so she did , and when all was ready , says her mother to her , " which will you have -- half of these with my blessing , or the whole with my curse ?
- [S2] " " curse or no curse , " says she , " the whole is little enough .
- [S12] about a week after , the second daughter went to seek her fortune , and she did n't care for her mother 's blessing no more nor her sister , and the very same thing happened to her . [BRIDGE]

---

### Example 9

**Story:** magic-apples
**Question:** why did the king's daughter ask the lad to fetch her apples ?
**Answer:** to trick him .
**Answer2:** she wanted to take his table - cloth , purse and cap and wish herself back home .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 4
- reasoning_operation: motivation
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S0 introduces the apples, S1 shows the request, S3 and S4 show the theft and escape, all necessary to understand the motivation behind the request.

**Required evidence sentences:**
- [S0] after they had eaten , the king 's daughter said : " o , do look at the handsome apples up there on the tree !
- [S1] if you were really kind , you would fetch me down a couple of them ! [BRIDGE]
- [S3] but he had forgotten his table - cloth and his purse , and these she took . [BRIDGE]
- [S4] and while he was shaking down the apples his cap fell off . [BRIDGE]

---

### Example 10

**Story:** kari-woodencoat
**Question:** why did the bull and the king's daughter go slowly ?
**Answer:** the bull was still healing from the fight .
**Answer2:** the bull was still weak .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 4
- reasoning_operation: summary_inference
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S10, S11, S12, and S13 together provide the information about the bull's condition and recovery, which explains why they went slowly.

**Required evidence sentences:**
- [S10] but then he was so wretched and so weak that he could not move a bit . [BRIDGE]
- [S11] his whole body was covered with wounds ; and he could not even tell the king 's daughter to take the horn of ointment from the troll 's girdle and anoint him with the salve . [BRIDGE]
- [S12] but she did so of her own accord , and then he recovered again . [BRIDGE]
- [S13] yet they had to stay where they were for three whole weeks , until he was able to go on again .

---

## 8. Comparison Note Against MAVEN-ERE

**MAVEN-ERE baseline (current):**
- QG-eligible Hard rescue: 0/112 blind Hard among judged candidates
- Quality-pass candidates: 0/21 blind Hard
- Root cause: answer sentences are locally identifiable; event-path hop count does not translate to answering difficulty

**FairytaleQA evidence audit (200 QA pairs):**
- Easy: 56 (28.0%)
- Medium: 115 (57.5%)
- Hard: 29 (14.5%)

**Assessment:** FairytaleQA produces 29 verified Hard candidates (14.5%) from 200 QA pairs. This is a meaningful improvement over MAVEN-ERE's 0% blind Hard rate. Narrative QA appears more promising for difficulty-controlled QG because answers often require understanding character motivation, causal chains, and multi-sentence inference rather than local event-phrase extraction.
