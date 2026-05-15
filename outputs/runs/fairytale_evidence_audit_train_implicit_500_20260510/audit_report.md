# FairytaleQA Evidence Audit Report

Generated: 2026-05-10 14:53:35

## 1. Dataset Loading Summary

| Field | Value |
|---|---|
| Split | train |
| Total QA pairs loaded | 8548 |
| Pool before filter | 8548 |
| Pool after filter | 2166 |
| Filter criteria | ex_or_im=implicit |
| QA pairs assessed | 500 |
| Fields available | story_name, story_section, question, answer1, local_or_sum, attribute, ex_or_im, split |
| Source | HuggingFace |

## 2. Evidence Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 155 | 31.0% |
| Medium | 269 | 53.8% |
| Hard | 76 | 15.2% |

## 3. Fairytale Labels vs Evidence Difficulty

### 3a. local-or-sum x difficulty

| local-or-sum | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| local | 140 | 206 | 46 | 392 |
| summary | 15 | 63 | 30 | 108 |

### 3b. ex-or-im x difficulty

| ex-or-im | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| implicit | 155 | 269 | 76 | 500 |

### 3c. attribute x difficulty

| attribute | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| action | 35 | 37 | 13 | 85 |
| causal relationship | 53 | 134 | 41 | 228 |
| character | 0 | 3 | 2 | 5 |
| feeling | 42 | 60 | 12 | 114 |
| outcome resolution | 7 | 10 | 1 | 18 |
| prediction | 18 | 25 | 7 | 50 |

## 4. Verified Hard Detail

| Metric | Count |
|---|---:|
| Hard count | 76 |
| Hard rate | 15.2% |

### Hard by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 18 |
| character_state | 8 |
| disambiguation | 6 |
| explicit_lookup | 1 |
| motivation | 26 |
| summary_inference | 17 |

### Hard by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 18 |
| disambiguation | 7 |
| motivation_bridge | 34 |
| summary_synthesis | 17 |

### Hard by attribute

| Attribute | Count |
|---|---:|
| action | 13 |
| causal relationship | 41 |
| character | 2 |
| feeling | 12 |
| outcome resolution | 1 |
| prediction | 7 |

### Hard by question prefix

| Prefix | Count |
|---|---:|
| how | 2 |
| how did | 16 |
| how will | 4 |
| what | 1 |
| what did | 2 |
| what was | 1 |
| what will | 5 |
| who | 1 |
| who was | 1 |
| why | 3 |
| why did | 34 |
| why was | 6 |

## 5. Answer Sufficiency Diagnostics

### answer_sentence_alone_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 154 | 30.8% |
| partial | 1 | 0.2% |
| no | 345 | 69.0% |

### section_evidence_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 452 | 90.4% |
| partial | 11 | 2.2% |
| no | 37 | 7.4% |

### bridge_removal_effect

| Value | Count | Pct |
|---|---:|---:|
| none | 64 | 12.8% |
| harder | 33 | 6.6% |
| ambiguous | 332 | 66.4% |
| unanswerable | 71 | 14.2% |

## 6. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 0 |
| Contradictions fixed | 113 |
| Missing trace fields | 0 |
| Invalid required sentence IDs | 0 |
| Hard validation violations | 0 |
| Status = ok | 500 |
| Status = llm_error | 0 |

## 7. Verified Hard Examples (up to 10)

### Example 1

**Story:** three-dogs
**Question:** how many times did the boy make an exchange with the old gray-beard ?
**Answer:** three times .
**Labels:** local-or-sum=summary, attribute=action, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: no
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The answer requires summarizing multiple exchanges across different sections of the story.

**Required evidence sentences:**
- [S4] " and when the old man noticed this , he began : " that is why i have come , for i want to exchange my dog for one of your pigs . [BRIDGE]
- [S16] " when the old man noticed this , he began : " that is why i have come , for i want to exchange my dog for one of your pigs . [BRIDGE]
- [S31] " the youth was at once willing and agreed to close the bargain . [BRIDGE]

---

### Example 2

**Story:** the-crane-that-crossed-the-river
**Question:** who was the skull and headless body ?
**Answer:** the mother .
**Labels:** local-or-sum=summary, attribute=character, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: S6 and S7 describe the skull and headless body, and S8 connects these to the mother, requiring synthesis of these sentences to answer the question.

**Required evidence sentences:**
- [S6] they had no sooner come in sight of this fall of water , than they heard a rolling sound behind them , and looking back , they beheld the skull of a woman rolling along the beach . [BRIDGE]
- [S7] it seemed to be pursuing them , and it came on with great speed ; when , behold , from out of the woods hard by , appeared a headless body , which made for the beach with the utmost dispatch . [BRIDGE]
- [S8] the skull too advanced toward it , and when they looked again , lo !

---

### Example 3

**Story:** leelinau-the-lost-daughter
**Question:** why did leelinau isolate herself ?
**Answer:** was not entertained with the girls ' activity .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 17
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The answer requires synthesizing multiple sentences to understand Leelinau's disinterest in the activities of her peers.

**Required evidence sentences:**
- [S0] day by day , these strange communings with unseen beings drew away the heart of leelinau more and more from the simple duties of the lodge , and she walked among her people , melancholy and silent , like a spirit who had visited them from another land .
- [S1] the pastimes which engaged the frolic moments of her young companions , passed by her as little trivial pageants in which she had no concern . [BRIDGE]
- [S2] when the girls of the neighboring lodges assembled to play at the favorite female game of pappus - e - ko - waun , or the block and string , before the lodge - door , leelinau would sit vacantly by , or enter so feebly into the spirit of the play as to show that it was irksome to her . [BRIDGE]
- [S3] again , in the evening , when the young people formed a ring around the lodge , and the piepeend - jigun , or leather and bone , passed rapidly from one to the other , she either handed it along without attempting to play , or if she took a part , it was with no effort to succeed . [BRIDGE]
- [S4] the time of the corn - gathering had come , and the young people of the tribe were assembled in the field , busy in plucking the ripened maize . [BRIDGE]
- [S5] one of the girls , noted for her beauty , had found a red ear , and every one congratulated her that a brave admirer was on his way to her father 's lodge . [BRIDGE]
- [S6] she blushed , and hiding the trophy in her bosom , she thanked the good spirit that it was a red ear , and not a crooked , that she had found . [BRIDGE]
- [S7] presently it chanced that one who was there among the young men , espied in the hands of leelinau , who had plucked it indifferently , one of the crooked kind , and at once the word " wa - ge - min ! [BRIDGE]
- [S8] " was shouted aloud through the field , and the whole circle was set in a roar . [BRIDGE]
- [S9] " the thief is in the corn - field ! [BRIDGE]
- [S10] " exclaimed the young man , iagoo by name , and famous in the tribe for his mirthful powers of story - telling ; " see you not the old man stooping as he enters the field ? [BRIDGE]
- [S11] see you not signs that he crouched as he crept in the dark ? [BRIDGE]
- [S12] is it not plain by this mark on the stalk that he was heavily bent in his back ? [BRIDGE]
- [S13] old man ! [BRIDGE]
- [S14] be nimble , or some one will take thee while thou art taking the ear . [BRIDGE]
- [S15] " these questions iagoo accompanied with the action of one bowed with age stealthily entering the corn - field . [BRIDGE]
- [S16] he went on : [BRIDGE]

---

### Example 4

**Story:** the-black-bull-of-norroway
**Question:** what will happened after the mystic spell falls upon the youngest princess ?
**Answer:** she will not see the black bull .
**Labels:** local-or-sum=summary, attribute=outcome resolution, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: no
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: S17 introduces the mystic spell, S18 explains its effect, and S19 confirms the outcome. Without S18, the reader cannot determine the full outcome.

**Required evidence sentences:**
- [S17] oh , woe - a - day !
- [S18] in a moment a mystic spell fell upon her , which caused her to become invisible to the eyes of the prince of norroway , who , having vanquished the evil spirit , was loosed from the spell which had lain over him , and had transformed him into the likeness of a great black bull , and who returned in haste down the glen to present himself , in his rightful form , to the maiden whom he loved , and whom he hoped to win for his bride . [BRIDGE]
- [S19] long , long he sought , but he could not find her , while all the time she was sitting patiently waiting on the stone ; but the spell was on her eyes also , and hindered her seeing him , as it hindered him seeing her .

---

### Example 5

**Story:** three-dogs
**Question:** why did the two princes seize the youth by the throat and strangle him ?
**Answer:** they were jealous of the youth .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: motivation
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S3 states the action but S0 and S1 explain the motivation. Without S0 and S1, the reader cannot determine why the princes acted.

**Required evidence sentences:**
- [S0] now when the princes learned that the youth had delivered the king 's three daughters , a great jealousy took possession of them , and they thought of how badly they had fared in their own venture .
- [S1] and they took counsel together as to how they might get the better of the youth , and win power and glory for themselves . [BRIDGE]
- [S3] then they suddenly threw themselves on their comrade , seized him by the throat and strangled him .

---

### Example 6

**Story:** youth-who-wanted-to-win-daughter-of-mother-in-corner
**Question:** how did the woman feel about her son singing and dancing ?
**Answer:** unhappy .
**Labels:** local-or-sum=local, attribute=feeling, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: character_state
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S6 indicates the mother's reaction but S2 and S3 provide the context for her feelings.

**Required evidence sentences:**
- [S2] but he liked to sing and to dance , and that is what he did all day long , and far into the night as well .
- [S3] the longer this went on , the worse off his mother was . [BRIDGE]
- [S6] at length it was too much for his mother .

---

### Example 7

**Story:** youth-who-wanted-to-win-daughter-of-mother-in-corner
**Question:** why did the mother think the youth's idea was not a bad idea ?
**Answer:** she could not afford to care for him anymore .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 4
- reasoning_operation: causal_chain
- bridge_removal_effect: ambiguous
- necessity_type: causal_bridge
- evidence_necessity_reason: S10 indicates the mother's change of mind but S3, S4, and S6 provide the context for her decision.

**Required evidence sentences:**
- [S3] the longer this went on , the worse off his mother was . [BRIDGE]
- [S4] the youth was growing , and he wanted so much to eat that it was barely possible to find it . [BRIDGE]
- [S6] at length it was too much for his mother .
- [S10] when the mother heard that she thought it might not be such a bad idea after all .

---

### Example 8

**Story:** werewolf
**Question:** what will the tiny old man do for the king's daughter ?
**Answer:** help her .
**Labels:** local-or-sum=local, attribute=prediction, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: disambiguation
- bridge_removal_effect: ambiguous
- necessity_type: disambiguation
- evidence_necessity_reason: S13 introduces the old man but S10 and S11 provide the context of the princess's situation. Without S10 and S11, the reader cannot determine the old man's role.

**Required evidence sentences:**
- [S10] and while she sat there , lost in her thoughts , she heard a voice say : " good evening , lovely maiden !
- [S11] why do you sit here so sad and lonely ? [BRIDGE]
- [S13] when she looked around there was nothing to be seen but a tiny old man , who nodded to her and seemed to be very humble .

---

### Example 9

**Story:** werewolf
**Question:** how did the queen treat the princess after the king left ?
**Answer:** poorly .
**Labels:** local-or-sum=summary, attribute=action, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: ambiguous
- necessity_type: summary_synthesis
- evidence_necessity_reason: S3 and S4 describe the queen's behavior but S1 provides the context of the king's departure. Without S1, the reader cannot infer the queen's treatment of the princess.

**Required evidence sentences:**
- [S1] news came that the enemy had entered the land , and the king was compelled to go to war .
- [S3] for no sooner had the king departed than the queen showed her true nature , and was just as harsh and unkind as she formerly had pretended to be friendly and obliging . [BRIDGE]
- [S4] not a day went by without her scolding and threatening the princess and the queen 's daughters were every bit as malicious as their mother .

---

### Example 10

**Story:** swan-maiden
**Question:** why did the prince bring his gun to the fields ?
**Answer:** to shoot birds and collect their feathers .
**Labels:** local-or-sum=local, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 3
- reasoning_operation: motivation
- bridge_removal_effect: ambiguous
- necessity_type: motivation_bridge
- evidence_necessity_reason: S2 states the task but S1 and S5 explain the motivation.

**Required evidence sentences:**
- [S1] " no , " said she , " there is more to be done yet before you can have what you ask for .
- [S2] if you can thatch the roof of the stable with bird feathers , no two of which shall be of the same color , and can do it between the rise and the set of sun to - morrow , then you shall have your sweetheart and welcome . [BRIDGE]
- [S5] so at sunrise he arose and went into the fields with his gun ; but if there were birds to be shot , it was few of them that he saw ; for at noontide he had but two , and they were both of a color .

---

## 8. Comparison Note Against MAVEN-ERE

**MAVEN-ERE baseline (current):**
- QG-eligible Hard rescue: 0/112 blind Hard among judged candidates
- Quality-pass candidates: 0/21 blind Hard
- Root cause: answer sentences are locally identifiable; event-path hop count does not translate to answering difficulty

**FairytaleQA evidence audit (500 QA pairs):**
- Easy: 155 (31.0%)
- Medium: 269 (53.8%)
- Hard: 76 (15.2%)

**Assessment:** FairytaleQA produces 76 verified Hard candidates (15.2%) from 500 QA pairs. This is a meaningful improvement over MAVEN-ERE's 0% blind Hard rate. Narrative QA appears more promising for difficulty-controlled QG because answers often require understanding character motivation, causal chains, and multi-sentence inference rather than local event-phrase extraction.
