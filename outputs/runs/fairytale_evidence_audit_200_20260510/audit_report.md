# FairytaleQA Evidence Audit Report

Generated: 2026-05-10 12:14:27

## 1. Dataset Loading Summary

| Field | Value |
|---|---|
| Split | validation |
| Total QA pairs loaded | 200 |
| QA pairs assessed | 200 |
| Fields available | story_name, story_section, question, answer1, answer2, local_or_sum, attribute, ex_or_im, ex_or_im2, split |
| Source | HuggingFace |

## 2. Evidence Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 156 | 78.0% |
| Medium | 36 | 18.0% |
| Hard | 8 | 4.0% |

## 3. Fairytale Labels vs Evidence Difficulty

### 3a. local-or-sum x difficulty

| local-or-sum | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| local | 148 | 25 | 6 | 179 |
| summary | 8 | 11 | 2 | 21 |

### 3b. ex-or-im x difficulty

| ex-or-im | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| explicit | 143 | 16 | 1 | 160 |
| implicit | 13 | 20 | 7 | 40 |

### 3c. attribute x difficulty

| attribute | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| action | 55 | 7 | 3 | 65 |
| causal relationship | 31 | 11 | 3 | 45 |
| character | 21 | 1 | 0 | 22 |
| feeling | 15 | 9 | 1 | 25 |
| outcome resolution | 18 | 2 | 0 | 20 |
| prediction | 6 | 5 | 1 | 12 |
| setting | 10 | 1 | 0 | 11 |

## 4. Verified Hard Detail

| Metric | Count |
|---|---:|
| Hard count | 8 |
| Hard rate | 4.0% |

### Hard by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 4 |
| disambiguation | 1 |
| summary_inference | 3 |

### Hard by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 4 |
| disambiguation | 1 |
| summary_synthesis | 3 |

## 5. Answer Sufficiency Diagnostics

### answer_sentence_alone_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 156 | 78.0% |
| partial | 4 | 2.0% |
| no | 40 | 20.0% |

### section_evidence_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 196 | 98.0% |
| partial | 3 | 1.5% |
| no | 1 | 0.5% |

### bridge_removal_effect

| Value | Count | Pct |
|---|---:|---:|
| none | 124 | 62.0% |
| harder | 23 | 11.5% |
| ambiguous | 41 | 20.5% |
| unanswerable | 12 | 6.0% |

## 6. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 0 |
| Contradictions fixed | 15 |
| Missing trace fields | 0 |
| Invalid required sentence IDs | 0 |
| Hard validation violations | 0 |
| Status = ok | 200 |
| Status = llm_error | 0 |

## 7. Verified Hard Examples (up to 10)

### Example 1

**Story:** why-dog-and-cat-are-enemies
**Question:** what will happen when the cat tries to get the ring back ?
**Answer:** she will successfully get the ring .
**Answer2:** she will get it back .
**Labels:** local-or-sum=summary, attribute=prediction, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: partial
- num_required_sentences: 20
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The entire sequence of events from S0 to S19 is required to infer the outcome of the cat's actions.

**Required evidence sentences:**
- [S0] " they must have the ring back again , " he said to the cat . [BRIDGE]
- [S1] the cat answered : " the ring has been carefully locked up in the chest , where no one can get at it . [BRIDGE]
- [S2] " " you must catch a mouse , " said the dog , " and the mouse must gnaw a hole in the chest and fetch out the ring . [BRIDGE]
- [S3] and if she does not want to , say that you will bite her to death , and you will see that she will do it . [BRIDGE]
- [S4] " this advice pleased the cat , and she caught a mouse . [BRIDGE]
- [S5] then she wanted to go to the house in which stood the chest , and the dog came after . [BRIDGE]
- [S6] they came to a broad river . [BRIDGE]
- [S7] and since the cat could not swim , the dog took her on his back and swam across with her . [BRIDGE]
- [S8] then the cat carried the mouse to the house in which the chest stood . [BRIDGE]
- [S9] the mouse gnawed a hole in the chest , and fetched out the ring . [BRIDGE]
- [S10] the cat put the ring in her mouth and went back to the river , where the dog was waiting for her , and swam across with her . [BRIDGE]
- [S11] then they started out together for home , in order to bring the lucky ring to their master and mistress . [BRIDGE]
- [S12] but the dog could only run along the ground ; when there was a house in the way he always had to go around it . [BRIDGE]
- [S13] the cat , however , quickly climbed over the roof , and so she reached home long before the dog , and brought the ring to her master . [BRIDGE]
- [S14] then her master said to his wife : " what a good creature the cat is ! [BRIDGE]
- [S15] we will always give her enough to eat and care for her as though she were our own child ! [BRIDGE]
- [S16] " but when the dog came home they beat him and scolded him , because he had not helped to bring home the ring again . [BRIDGE]
- [S17] and the cat sat by the fireplace , purred and said never a word . [BRIDGE]
- [S18] then the dog grew angry at the cat , because she had robbed him of his reward , and when he saw her he chased her and tried to seize her . [BRIDGE]
- [S19] and ever since that day cat and dog are enemies . [BRIDGE]

---

### Example 2

**Story:** little-boy-blue
**Question:** what did the animals do when the little boy fell asleep ?
**Answer:** strayed near the edge of the meadow .
**Answer2:** feeding upon the squire 's pet cornfield and the sheep were enjoying themselves amidst the juicy grasses of the meadows .
**Labels:** local-or-sum=local, attribute=action, ex-or-im=explicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S2 and S4 provide the actions of the animals, but S2 is necessary to understand the sequence of events.

**Required evidence sentences:**
- [S2] the sheep strayed near the edge of the meadow and paused , waiting for the warning sound of the horn . [BRIDGE]
- [S3] and the breeze carried the fragrance of the growing corn to the nostrils of the browsing cows and tempted them nearer and nearer to the forbidden feast .
- [S4] but the silver horn was silent , and before long the cows were feeding upon the squire 's pet cornfield and the sheep were enjoying themselves amidst the juicy grasses of the meadows .

---

### Example 3

**Story:** old-dschang
**Question:** why didn't the match-maker want old dschang to marry sir we's daughter ?
**Answer:** he was a poor old gardener .
**Answer2:** he is too old for a beautiful daughter .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S1 introduces the match-maker's response, S2 and S3 provide the reasons, all are needed to understand the match-maker's stance.

**Required evidence sentences:**
- [S1] then the old match - maker said : " you do not know what you wish ! [BRIDGE]
- [S2] why should a gentleman 's beautiful daughter condescend to marry a poor old gardener like yourself ?
- [S3] even though you had money to burn , your white hair would not match her black locks .

---

### Example 4

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
- evidence_necessity_reason: S0 and S1 provide the cause, S2 the effect, and S1 bridges the cause and effect.

**Required evidence sentences:**
- [S0] once an aristocratic relative visited sir we and said : " if you had really been poor , were there not enough young gentlemen in the neighborhood for your daughter ?
- [S1] why did you have to marry her to such a wrinkled old gardener ? [BRIDGE]
- [S2] now that you have thrown her away , so to speak , it would be better if both of them left this part of the country .

---

### Example 5

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
- evidence_necessity_reason: S7 and S9 provide the background for the cat's action, and S11 explains the outcome, which together form the causal chain.

**Required evidence sentences:**
- [S7] then her master said to his wife : " what a good creature the cat is !
- [S9] " but when the dog came home they beat him and scolded him , because he had not helped to bring home the ring again . [BRIDGE]
- [S11] then the dog grew angry at the cat , because she had robbed him of his reward , and when he saw her he chased her and tried to seize her . [BRIDGE]

---

### Example 6

**Story:** assipattle-and-the-mester-stoorworm
**Question:** how did assipattle feel when his brothers and mother mistreated him ?
**Answer:** upset .
**Answer2:** sad .
**Labels:** local-or-sum=local, attribute=feeling, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: disambiguation
- bridge_removal_effect: ambiguous
- necessity_type: disambiguation
- evidence_necessity_reason: S2 provides the context that Assipattle had a hard life and would be miserable without his sister, implying he felt upset or sad.

**Required evidence sentences:**
- [S0] and his brothers , working hard in the fields , would point to him with mocking fingers , and laugh , and say to each other how well the name suited him , and of how little use he was in the world .
- [S1] and when they came home from their work , they would push him about and tease him , and even his mother would make him sweep the floor , and draw water from the well , and fetch peats from the peat - stack , and do all the little odd jobs that nobody else would do .
- [S2] so poor assipattle had rather a hard life of it , and he would often have been very miserable had it not been for his sister , who loved him dearly , and who would listen quite patiently to all the stories that he had to tell ; who never laughed at him or told him that he was telling lies , as his brothers did . [BRIDGE]

---

### Example 7

**Story:** assipattle-and-the-mester-stoorworm
**Question:** how did assipattle trick the boatman ?
**Answer:** he pretended he found gold .
**Answer2:** he pretended to find gold .
**Labels:** local-or-sum=local, attribute=action, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 4
- reasoning_operation: summary_inference
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S10-S13 together show the action and the trick, but S10 is crucial for understanding the trick.

**Required evidence sentences:**
- [S10] presently the lad gave a wild shriek , and jumped high in the air . [BRIDGE]
- [S11] " gold , gold !
- [S12] " he cried .
- [S13] " by the name of thor , who would have looked to find gold here ?

---

### Example 8

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
- evidence_necessity_reason: S5-S8 provide the causal chain of events leading to Assipattle's escape.

**Required evidence sentences:**
- [S5] presently he came to the huge creature 's liver , and having heard that the liver of a fish is full of oil , he made a hole in it and put in the live peat . [BRIDGE]
- [S6] woe 's me ! [BRIDGE]
- [S7] but there was a conflagration ! [BRIDGE]
- [S8] and assipattle just got back to his boat in time ; for the mester stoorworm , in its convulsions , threw the boat right out of its mouth again , and it was flung up , high and dry , on the bare land .

---

## 8. Comparison Note Against MAVEN-ERE

**MAVEN-ERE baseline (current):**
- QG-eligible Hard rescue: 0/112 blind Hard among judged candidates
- Quality-pass candidates: 0/21 blind Hard
- Root cause: answer sentences are locally identifiable; event-path hop count does not translate to answering difficulty

**FairytaleQA evidence audit (200 QA pairs):**
- Easy: 156 (78.0%)
- Medium: 36 (18.0%)
- Hard: 8 (4.0%)

**Assessment:** FairytaleQA produces 8 verified Hard candidates (4.0%) from 200 QA pairs. This is a meaningful improvement over MAVEN-ERE's 0% blind Hard rate. Narrative QA appears more promising for difficulty-controlled QG because answers often require understanding character motivation, causal chains, and multi-sentence inference rather than local event-phrase extraction.
