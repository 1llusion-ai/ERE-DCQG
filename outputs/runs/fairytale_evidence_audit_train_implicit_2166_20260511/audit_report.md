# FairytaleQA Evidence Audit Report

Generated: 2026-05-11 17:26:00

## 1. Dataset Loading Summary

| Field | Value |
|---|---|
| Split | train |
| Total QA pairs loaded | 8548 |
| Pool before filter | 8548 |
| Pool after filter | 2166 |
| Filter criteria | ex_or_im=implicit |
| QA pairs assessed | 2166 |
| Fields available | story_name, story_section, question, answer1, local_or_sum, attribute, ex_or_im, split |
| Source | HuggingFace |

## 2. Evidence Difficulty Distribution

| Difficulty | Count | Pct |
|---|---:|---:|
| Easy | 659 | 30.4% |
| Medium | 1153 | 53.2% |
| Hard | 354 | 16.3% |

## 3. Fairytale Labels vs Evidence Difficulty

### 3a. local-or-sum x difficulty

| local-or-sum | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| local | 608 | 861 | 211 | 1680 |
| summary | 51 | 292 | 143 | 486 |

### 3b. ex-or-im x difficulty

| ex-or-im | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| implicit | 659 | 1153 | 354 | 2166 |

### 3c. attribute x difficulty

| attribute | Easy | Medium | Hard | Total |
|---|---:|---:|---:|---:|
| action | 131 | 137 | 34 | 302 |
| causal relationship | 267 | 590 | 182 | 1039 |
| character | 22 | 24 | 5 | 51 |
| feeling | 158 | 244 | 72 | 474 |
| outcome resolution | 32 | 51 | 17 | 100 |
| prediction | 47 | 105 | 43 | 195 |
| setting | 2 | 2 | 1 | 5 |

## 4. Verified Hard Detail

| Metric | Count |
|---|---:|
| Hard count | 354 |
| Hard rate | 16.3% |

### Hard by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 91 |
| character_state | 34 |
| contrast | 1 |
| disambiguation | 16 |
| explicit_lookup | 9 |
| motivation | 121 |
| summary_inference | 78 |
| temporal_order | 4 |

### Hard by necessity_type

| Type | Count |
|---|---:|
| answer_identification | 4 |
| causal_bridge | 93 |
| disambiguation | 17 |
| motivation_bridge | 153 |
| summary_synthesis | 83 |
| temporal_bridge | 4 |

### Hard by attribute

| Attribute | Count |
|---|---:|
| action | 34 |
| causal relationship | 182 |
| character | 5 |
| feeling | 72 |
| outcome resolution | 17 |
| prediction | 43 |
| setting | 1 |

### Hard by question prefix

| Prefix | Count |
|---|---:|
| how | 6 |
| how did | 66 |
| how will | 32 |
| what | 8 |
| what did | 10 |
| what happened after | 6 |
| what was | 1 |
| what will | 37 |
| where did | 1 |
| who | 2 |
| who was | 3 |
| why | 8 |
| why did | 146 |
| why does | 3 |
| why was | 24 |
| why were | 1 |

## 5. Answer Sufficiency Diagnostics

### answer_sentence_alone_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 646 | 29.8% |
| partial | 11 | 0.5% |
| no | 1509 | 69.7% |

### section_evidence_sufficient

| Value | Count | Pct |
|---|---:|---:|
| yes | 1975 | 91.2% |
| partial | 60 | 2.8% |
| no | 131 | 6.0% |

### bridge_removal_effect

| Value | Count | Pct |
|---|---:|---:|
| none | 385 | 17.8% |
| harder | 155 | 7.2% |
| ambiguous | 1403 | 64.8% |
| unanswerable | 223 | 10.3% |

## 6. Consistency Diagnostics

| Diagnostic | Count |
|---|---:|
| Parse failures | 8 |
| Contradictions fixed | 441 |
| Missing trace fields | 0 |
| Invalid required sentence IDs | 0 |
| Hard validation violations | 0 |
| Status = ok | 2158 |
| Status = llm_error | 8 |

## 7. Verified Hard Examples (up to 10)

### Example 1

**Story:** the-crane-that-crossed-the-river
**Question:** who was the skull and headless body ?
**Answer:** the mother .
**Labels:** local-or-sum=summary, attribute=character, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 14
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The answer requires synthesizing multiple sentences to understand the sequence of events and the identity of the skull and headless body.

**Required evidence sentences:**
- [S6] they had no sooner come in sight of this fall of water , than they heard a rolling sound behind them , and looking back , they beheld the skull of a woman rolling along the beach . [BRIDGE]
- [S7] it seemed to be pursuing them , and it came on with great speed ; when , behold , from out of the woods hard by , appeared a headless body , which made for the beach with the utmost dispatch . [BRIDGE]
- [S8] the skull too advanced toward it , and when they looked again , lo ! [BRIDGE]
- [S9] they had united , and were making all haste to come up with the hunter and his two sons . [BRIDGE]
- [S10] they now might well be in extreme fear , for they knew not how to escape her . [BRIDGE]
- [S11] at this moment , one of them looked out and saw a stately crane sitting on a rock in the middle of the rapids . [BRIDGE]
- [S12] they called out to the bird , " see , grandfather , we are persecuted . [BRIDGE]
- [S13] come and take us across the falls that we may escape her . [BRIDGE]
- [S14] " the crane so addressed was of extraordinary size , and had arrived at a great old age , and , as might be expected , he sat , when first descried by the two sons , in a state of profound thought , revolving his long experience of life there in the midst of the most violent eddies . [BRIDGE]
- [S15] when he heard himself appealed to , the crane stretched forth his neck with great deliberation , and lifting himself slowly by his wings , he flew across to their assistance . [BRIDGE]
- [S16] " be careful , " said the old crane , " that you do not touch the crown of my head . [BRIDGE]
- [S17] i am bald from age and long service , and very tender at that spot . [BRIDGE]
- [S18] should you be so unlucky as to lay a hand upon it , i shall not be able to avoid throwing you both in the rapids . [BRIDGE]
- [S19] " [BRIDGE]

---

### Example 2

**Story:** the-crane-that-crossed-the-river
**Question:** why did the mother want to keep the young man a secret ?
**Answer:** the young man was her lover .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 15
- reasoning_operation: causal_chain
- bridge_removal_effect: unanswerable
- necessity_type: causal_bridge
- evidence_necessity_reason: The answer requires understanding the sequence of events and the mother's actions to infer her motive for keeping the young man a secret.

**Required evidence sentences:**
- [S5] at length the elder of the two said to his mother : " my mother , who is this tall young man that comes here so often during our father 's absence ? [BRIDGE]
- [S6] does he wish to see him ? [BRIDGE]
- [S7] shall i tell him when he comes back this evening ? [BRIDGE]
- [S8] " " naubesah , you little fool , " said the mother , " mind your bow and arrows , and do not be afraid to enter the forest in search of birds and squirrels , with your little brother . [BRIDGE]
- [S9] it is not manly to be ever about the lodge . [BRIDGE]
- [S10] nor will you become a warrior if you tell all the little things that you see and hear to your father . [BRIDGE]
- [S11] say not a word to him . [BRIDGE]
- [S12] " the boys obeyed , but as they grew older and still noticed the visits of the stranger , they resolved to speak again to their mother . [BRIDGE]
- [S13] they now told her that they meant to make known to their father all that they had witnessed , for they frequently saw this young man passing through the woods , and he did not walk in the path , nor did he carry any thing to eat . [BRIDGE]
- [S14] if he had any message to deliver at their lodge , why did he not give it to their father ? [BRIDGE]
- [S15] for they had observed that messages were always addressed to men , and not to women . [BRIDGE]
- [S16] when her sons spoke thus to her , the mother was greatly vexed . [BRIDGE]
- [S17] " i will kill you , " she said , " if you speak of it . [BRIDGE]
- [S18] " in fear they for a time held their peace , but still taking note that the stranger came so often and by stealth to the lodge , they resolved at last to speak with their father . [BRIDGE]
- [S19] accordingly one day , when they were out in the woods , learning to follow the chase , they told him all that they had seen . [BRIDGE]

---

### Example 3

**Story:** the-crane-that-crossed-the-river
**Question:** why did the mother seek revenge ?
**Answer:** her sons told her husband about the young man .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 19
- reasoning_operation: causal_chain
- bridge_removal_effect: unanswerable
- necessity_type: causal_bridge
- evidence_necessity_reason: The answer requires understanding the sequence of events and the mother's actions to infer her motive for seeking revenge.

**Required evidence sentences:**
- [S1] accordingly one day , when they were out in the woods , learning to follow the chase , they told him all that they had seen . [BRIDGE]
- [S2] the face of the father grew dark . [BRIDGE]
- [S3] he was still for a while , and when at length he looked up- " it is done ! [BRIDGE]
- [S4] " he said . [BRIDGE]
- [S5] " do you , my children , tarry here until the hour of the falling of the sun , then come to the lodge and you will find me . [BRIDGE]
- [S6] " the father left them at a slow pace , and they remained sporting away their time till the hour for their return had come . [BRIDGE]
- [S7] when they reached the lodge the mother was not there . [BRIDGE]
- [S8] they dared not to ask their father whither she had gone , and from that day forth her name was never spoken again in the lodge . [BRIDGE]
- [S9] in course of time the two boys had grown to be men , and although the mother was never more seen in the lodge , in charge of her household tasks , nor on the path in the forest , nor by the river side , she still lingered , ever and ever , near the lodge . [BRIDGE]
- [S10] changed , but the same , with ghastly looks and arms that were withered , she appeared to her sons as they returned from the hunt , in the twilight , in the close of the day . [BRIDGE]
- [S11] at night she darkly unlatched the lodge - door and glided in , and bent over them as they sought to sleep . [BRIDGE]
- [S12] oftenest it was her bare brow , white , and bony , and bodyless , that they saw floating in the air , and making a mock of them in the wild paths of the forest , or in the midnight darkness of the lodge . [BRIDGE]
- [S13] she was a terror to all their lives , and she made every spot where they had seen her , hideous to the living eye ; so that after being long buffeted and beset , they at last resolved , together with their father , now stricken in years , to leave the country . [BRIDGE]
- [S14] they began a journey toward the south . [BRIDGE]
- [S15] after traveling many days along the shore of a great lake , they passed around a craggy bluff , and came upon a scene where there was a rough fall of waters , and a river issuing forth from the lake . [BRIDGE]
- [S16] they had no sooner come in sight of this fall of water , than they heard a rolling sound behind them , and looking back , they beheld the skull of a woman rolling along the beach . [BRIDGE]
- [S17] it seemed to be pursuing them , and it came on with great speed ; when , behold , from out of the woods hard by , appeared a headless body , which made for the beach with the utmost dispatch . [BRIDGE]
- [S18] the skull too advanced toward it , and when they looked again , lo ! [BRIDGE]
- [S19] they had united , and were making all haste to come up with the hunter and his two sons . [BRIDGE]

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
- evidence_necessity_reason: S17 introduces the mystic spell, S18 explains its effect, and S19 confirms the outcome. Without S17, the sequence of events leading to the outcome is unclear.

**Required evidence sentences:**
- [S17] oh , woe - a - day ! [BRIDGE]
- [S18] in a moment a mystic spell fell upon her , which caused her to become invisible to the eyes of the prince of norroway , who , having vanquished the evil spirit , was loosed from the spell which had lain over him , and had transformed him into the likeness of a great black bull , and who returned in haste down the glen to present himself , in his rightful form , to the maiden whom he loved , and whom he hoped to win for his bride .
- [S19] long , long he sought , but he could not find her , while all the time she was sitting patiently waiting on the stone ; but the spell was on her eyes also , and hindered her seeing him , as it hindered him seeing her .

---

### Example 5

**Story:** the-three-crowns
**Question:** what did the king decide to do after his sons failed ?
**Answer:** picked up the man by himself .
**Labels:** local-or-sum=local, attribute=action, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: no
- num_required_sentences: 3
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: S0 and S1 provide context about the prince's actions, and S6 shows the outcome, but the king's decision to pick up the man by himself is not directly stated in the provided section.

**Required evidence sentences:**
- [S0] so he changed his clothes , and washed himself , and out he set to the prince 's forge and asked him to sit along with himself . [BRIDGE]
- [S1] the prince begged to be allowed to sit in the other carriage , and when they were half - way he opened his snuff - box . [BRIDGE]
- [S6] ' so seven inches vanished ; and when the carriage door was opened in the yard , out walks the prince as fine as hands could make him , and the first thing he did was to run over to his bride and embrace her .

---

### Example 6

**Story:** the-seal-catcher-and-the-merman
**Question:** what will the stranger do after the seal catcher heals his father ?
**Answer:** release him .
**Labels:** local-or-sum=summary, attribute=prediction, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: partial
- num_required_sentences: 6
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: S8-S13 provide the necessary context and outcome, but S8 and S9 are crucial for understanding the sequence of events leading to the answer.

**Required evidence sentences:**
- [S8] but presently , to his great joy , his guide approached him , and said , " now you are at liberty to return home to your wife and children . [BRIDGE]
- [S9] i will take you to them , but only on one condition . [BRIDGE]
- [S10] " " and what is that ?
- [S11] " asked the seal catcher eagerly , overjoyed at the prospect of being restored safely to the upper world , and to his family .
- [S12] " that you will take a solemn oath never to wound a seal again .
- [S13] " " that will i do right gladly , " he replied .

---

### Example 7

**Story:** the-witch-of-fife
**Question:** why did the husband leave his wife alone and never tried to find out her secrets again ?
**Answer:** he got himself in trouble .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: no
- num_required_sentences: 11
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The entire section is needed to understand the sequence of events leading to the husband's decision to leave his wife alone.

**Required evidence sentences:**
- [S0] as soon as they heard that , the men servants cried out that he was a warlock , and they dragged him before the bishop , and , as bishops in those days had a holy horror of warlocks and witches , he ordered him to be burned alive . [BRIDGE]
- [S1] when the sentence was pronounced , you may be very sure that the poor old man wished with all his heart that he had stayed quietly at home in bed , and never hankered after the bishop 's wine . [BRIDGE]
- [S2] but it was too late to wish that now , for the servants dragged him out into the courtyard , and put a chain round his waist , and fastened it to a great iron stake , and they piled faggots of wood round his feet and set them alight . [BRIDGE]
- [S3] as the first tiny little tongue of flame crept up , the poor old man thought that his last hour had come . [BRIDGE]
- [S4] but when he thought that , he forgot completely that his wife was a witch . [BRIDGE]
- [S5] for , just as the little tongue of flame began to singe his breeches , there was a swish and a flutter in the air , and a great grey bird , with outstretched wings , appeared in the sky , and swooped down suddenly , and perched for a moment on the old man 's shoulder . [BRIDGE]
- [S6] and in this grey bird 's mouth was a little red pirnie , which , to everyone 's amazement , it popped on to the prisoner 's head . [BRIDGE]
- [S7] then it gave one fierce croak , and flew away again , but to the old man 's ears that croak was the sweetest music that he had ever heard . [BRIDGE]
- [S8] for to him it was the croak of no earthly bird , but the voice of his wife whispering words of magic to him . [BRIDGE]
- [S9] and when he heard them he jumped for joy , for he knew that they were words of deliverance , and he shouted them aloud , and his chains fell off , and he mounted in the air -- up and up -- while the onlookers watched him in awestruck silence . [BRIDGE]
- [S10] he flew right away to the kingdom of fife , without as much as saying good - bye to them ; and when he found himself once more safely at home , you may be very sure that he never tried to find out his wife 's secrets again , but left her alone to her own devices .

---

### Example 8

**Story:** the-elfin-knight
**Question:** what will happen after earl gregory follows the horseman ?
**Answer:** he will meet goblins .
**Labels:** local-or-sum=summary, attribute=prediction, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: no
- num_required_sentences: 10
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The entire section is needed to infer the prediction about what will happen after Earl Gregory follows the horseman.

**Required evidence sentences:**
- [S0] but earl st . [BRIDGE]
- [S1] clair forgot that he carried a talisman which his companion lacked , that enabled him to see things as they really were , while the other 's eyes were holden , and he was startled and amazed when earl gregory said sharply , " thy mind hath gone mad over this elfin king . [BRIDGE]
- [S2] i tell thee he who passed was a goodly knight , clad in a green vesture , and riding on a great black jennet . [BRIDGE]
- [S3] and because i love a gallant horseman , and would fain learn his name and degree , i will follow him till i find him , even if it be at the world 's end . [BRIDGE]
- [S4] " and without another word he put spurs to his horse and galloped off in the direction which the mysterious stranger had taken , leaving earl st . [BRIDGE]
- [S5] clair alone upon the moorland , his fingers touching the sacred sign and his trembling lips muttering prayers for protection . [BRIDGE]
- [S6] for he knew that his friend had been bewitched , and he made up his mind , brave gentleman that he was , that he would follow him to the world 's end , if need be , and try to deliver him from the spell that had been cast over him . [BRIDGE]
- [S7] meanwhile earl gregory rode on and on , ever following in the wake of the knight in green , over moor , and burn , and moss , till he came to the most desolate region that he had ever been in in his life ; where the wind blew cold , as if from snow - fields , and where the hoar - frost lay thick and white on the withered grass at his feet . [BRIDGE]
- [S8] and there , in front of him , was a sight from which mortal man might well shrink back in awe and dread . [BRIDGE]
- [S9] for he saw an enormous ring marked out on the ground , inside of which the grass , instead of being withered and frozen , was lush , and rank , and green , where hundreds of shadowy elfin figures were dancing , clad in loose transparent robes of dull blue , which seemed to curl and twist round their wearers like snaky wreaths of smoke .

---

### Example 9

**Story:** the-brown-bear-of-norway
**Question:** why couldn't the prince wake up at night ?
**Answer:** the witch 's daughter gave him sleepy posset .
**Labels:** local-or-sum=summary, attribute=causal relationship, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: no
- num_required_sentences: 4
- reasoning_operation: causal_chain
- bridge_removal_effect: unanswerable
- necessity_type: causal_bridge
- evidence_necessity_reason: S0 indicates the prince's deep sleep, S9-S11 provide the context of the sleepy posset and its effect.

**Required evidence sentences:**
- [S0] when night came on she was let into the palace and lay down till the prince was in such a dead sleep that all she did could n't awake him .
- [S9] says she , ' did you drink any sleepy posset either of these evenings before you went to bed ? [BRIDGE]
- [S10] ' ' i did , ' said he . [BRIDGE]
- [S11] ' the two evenings my wife gave me something to drink , but i do n't know whether it was a sleepy posset or not .

---

### Example 10

**Story:** jamie-freel-and-the-young-lady
**Question:** how will the lady feel when neither the servant nor her father recognizes her ?
**Answer:** upset .
**Labels:** local-or-sum=summary, attribute=feeling, ex-or-im=implicit

**Evidence assessment:**
- answer_sentence_alone_sufficient: no
- section_evidence_sufficient: yes
- num_required_sentences: 38
- reasoning_operation: summary_inference
- bridge_removal_effect: unanswerable
- necessity_type: summary_synthesis
- evidence_necessity_reason: The lady's feelings are inferred from the sequence of events and her reaction to the lack of recognition. Without the full sequence, the reader cannot determine the lady's emotional state.

**Required evidence sentences:**
- [S10] " the gentleman that lives here has no daughter , my girl . [BRIDGE]
- [S11] he had one , but she died better nor a year ago . [BRIDGE]
- [S12] " " do you not know me , sullivan ? [BRIDGE]
- [S13] " " no , poor girl , i do not . [BRIDGE]
- [S14] " " let me see the gentleman . [BRIDGE]
- [S15] i only ask to see him . [BRIDGE]
- [S16] " " well , that 's not much to ax . [BRIDGE]
- [S17] we 'll see what can be done . [BRIDGE]
- [S18] " in a few moments the lady 's father came to the door . [BRIDGE]
- [S19] " how dare you call me your father ? [BRIDGE]
- [S20] " cried the old gentleman angrily . [BRIDGE]
- [S21] " you are an impostor . [BRIDGE]
- [S22] i have no daughter . [BRIDGE]
- [S23] " " look in my face , father , and surely you 'll remember me . [BRIDGE]
- [S24] " " my daughter is dead and buried . [BRIDGE]
- [S25] she died a long , long time ago . [BRIDGE]
- [S26] " the old gentleman 's voice changed from anger to sorrow . [BRIDGE]
- [S27] " you can go , " he concluded . [BRIDGE]
- [S28] " stop , dear father , till you look at this ring on my finger . [BRIDGE]
- [S29] look at your name and mine engraved on it . [BRIDGE]
- [S30] " " it certainly is my daughter 's ring , but i do not know how you came by it . [BRIDGE]
- [S31] i fear in no honest way . [BRIDGE]
- [S32] " " call my mother -- she will be sure to know me , " said the poor girl , who by this time was weeping bitterly . [BRIDGE]
- [S33] " my poor wife is beginning to forget her sorrow . [BRIDGE]
- [S34] she seldom speaks of her daughter now . [BRIDGE]
- [S35] why should i renew her grief by reminding her of her loss ? [BRIDGE]
- [S36] " but the young lady persevered till at last the mother was sent for . [BRIDGE]
- [S37] " mother , " she began , when the old lady came to the door , " do n't you know your daughter ? [BRIDGE]
- [S38] " " i have no daughter . [BRIDGE]
- [S39] my daughter died , and was buried a long , long time ago . [BRIDGE]
- [S40] " " only look in my face and surely you 'll know me . [BRIDGE]
- [S41] " the old lady shook her head . [BRIDGE]
- [S42] " you have all forgotten me . [BRIDGE]
- [S43] but look at this mole on my neck . [BRIDGE]
- [S44] surely , mother , you know me now ? [BRIDGE]
- [S45] " " yes , yes , " said her mother , " my gracie had a mole on her neck like that . [BRIDGE]
- [S46] but then i saw her in the coffin , and saw the lid shut down upon her . [BRIDGE]
- [S47] " [BRIDGE]

---

## 8. Comparison Note Against MAVEN-ERE

**MAVEN-ERE baseline (current):**
- QG-eligible Hard rescue: 0/112 blind Hard among judged candidates
- Quality-pass candidates: 0/21 blind Hard
- Root cause: answer sentences are locally identifiable; event-path hop count does not translate to answering difficulty

**FairytaleQA evidence audit (2166 QA pairs):**
- Easy: 659 (30.4%)
- Medium: 1153 (53.2%)
- Hard: 354 (16.3%)

**Assessment:** FairytaleQA produces 354 verified Hard candidates (16.3%) from 2166 QA pairs. This is a meaningful improvement over MAVEN-ERE's 0% blind Hard rate. Narrative QA appears more promising for difficulty-controlled QG because answers often require understanding character motivation, causal chains, and multi-sentence inference rather than local event-phrase extraction.
