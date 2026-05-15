# Path Pilot Review

Run dir: `outputs/runs/path_pilot_20260502_181500`

## Summary

- Raw paths: 59
- Prefilter passed: 50/59
- LLM judged sample: 15 paths, 5 per level
- LLM kept: 10/15
- Kept answer phrase status: 10 complete
- Note: at least one kept phrase still looks incomplete, so do not batch yet without manual spot check.

## LLM-Judged Paths

### 1. Easy | keep=True | doc=f28bce270df5a122c09365002d247e76

- Title: United States occupation of Nicaragua
- Path: assumed [Choosing] S3 -> opposed [Agree_or_refuse_to_act] S4
- Relations: CAUSE/PRECONDITION
- Gold answer phrase: `President Herbert Hoover (1929鈥?933) opposed the relationship`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=1, single_sentence_risk=high, recommended=easy
- LLM reason: The final event can be clearly questioned and answered from the provided sentence.

Supporting sentences:
- [S2] American military interventions in Nicaragua were designed to stop any other nation except the United States of America from building a Nicaraguan Canal.
- [S3] Nicaragua assumed a quasi-protectorate status under the 1916 Bryan鈥揅hamorro Treaty.
- [S4] President Herbert Hoover (1929鈥?933) opposed the relationship.
- [S5] Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention.

### 2. Easy | keep=True | doc=c0c67db40cd5e2e03645ff1116fafcfc

- Title: Cherry Valley massacre
- Path: restrain [Hindering] S8 -> drove [Motion] S12
- Relations: TEMPORAL/BEFORE
- Gold answer phrase: `drove the Iroquois out of western New York`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=1, single_sentence_risk=high, recommended=easy
- LLM reason: The final event can be clearly questioned and answered from the provided sentence.

Supporting sentences:
- [S7] Butler's authority with the Indians was undermined by his poor treatment of Joseph Brant, the leader of the Mohawks.
- [S8] Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.
- [S9] During the campaigns of 1778, Brant achieved an undeserved reputation for brutality.
- [S11] Diaries belonging to British soldiers during the campaign state the regiment as being the "butchers" and given that Butler was the overall commander of the expedition, there is controversy as to who actually ordered or failed to restrain the killings.
- [S12] The massacre contributed to calls for reprisals, leading to the 1779 Sullivan Expedition which drove the Iroquois out of western New York.

### 3. Easy | keep=True | doc=f46091471f38006751fcdcda15d5775b

- Title: King David Hotel bombing
- Path: planted [Placing] S5 -> injuries [Bodily_harm] S7
- Relations: CAUSE/PRECONDITION
- Gold answer phrase: `injuries occurred in the road outside the hotel`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=1, single_sentence_risk=high, recommended=easy
- LLM reason: The final event can be clearly questioned and answered from the provided sentence.

Supporting sentences:
- [S4] It was conceived as part of a response to Operation Agatha (a series of widespread raids, including one on the Jewish Agency, conducted by the British authorities) and was the deadliest directed at the British during the Mandate era (1920鈥?948).
- [S5] Disguised as Arab workmen and as hotel waiters, members of the Irgun planted a bomb in the basement of the main building of the hotel, whose southern wing housed the Mandate Secretariat and a few offices of the British military headquarters.
- [S6] The resulting explosion caused the collapse of the western half of the southern wing of the hotel.
- [S7] Some of the inflicted deaths and injuries occurred in the road outside the hotel and in adjacent buildings.
- [S8] The Irgun sent warnings by telephone, including one to the hotel's own switchboard, which, possibly because hoax bomb warnings were rife at the time, the staff decided to ignore, but none directly to the British authorities.

### 4. Easy | keep=True | doc=6dabade56742b6040cda6a5838176f6c

- Title: Who's That Girl World Tour
- Path: titled [Name_conferral] S13 -> released [Publishing] S13
- Relations: TEMPORAL/BEFORE
- Gold answer phrase: `was released in VHS titled `
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=partial, steps=1, single_sentence_risk=high, recommended=easy
- LLM reason: The proposed answer phrase is truncated and does not provide a complete answer.

Supporting sentences:
- [S12] According to Pollstar, it was the second highest-grossing female concert tour of 1987, behind Tina Turner's Break Every Rule Tour.
- [S13] Who's That Girl was broadcast in a number of international television channels and was released in VHS titled "".
- [S14] Biographer J. Randy Taraborrelli commented that "Many female artists behave like a diva for a period when they reach superstar status, and the 'Who's That Girl?'

### 5. Easy | keep=True | doc=db50381e7d1dd4a41fb4ac60eaebe3a4

- Title: Battle of Orthez
- Path: attack [Attack] S0 -> Battle [Hostile_encounter] S9
- Relations: TEMPORAL/BEFORE
- Gold answer phrase: `The next action was the Battle of Toulouse`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=1, single_sentence_risk=high, recommended=easy
- LLM reason: one short sentence

Supporting sentences:
- [S0] The Battle of Orthez (27 February 1814) saw the Anglo-Portuguese Army under Field Marshal Arthur Wellesley, Marquess of Wellington attack an Imperial French army led by Marshal Nicolas Soult in southern France.
- [S1] The outnumbered French repelled several Allied assaults on their right flank, but their center and left flank were overcome and Soult was compelled to retreat.
- [S8] In subsequent operations, Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse.
- [S9] The next action was the Battle of Toulouse.

### 6. Medium | keep=True | doc=e253b7fd1109bd5f87966022eea7762f

- Title: Myyrmanni bombing
- Path: crowded [Come_together] S5 -> exploded [Attack] S1 -> died [Death] S2
- Relations: TEMPORAL/BEFORE -> CAUSE/CAUSE
- Gold answer phrase: `In total seven died, including two teenagers`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=2, single_sentence_risk=high, recommended=medium
- LLM reason: The path supports a clear question about the final event, and the answer can be derived from the answer sentence.

Supporting sentences:
- [S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrm盲ki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.
- [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
- [S2] In total seven died, including two teenagers and a 7-year-old.
- [S3] 166 people were injured, including 10 children.
- [S4] 66 victims required hospitalization with the remainder treated and released at the scene.
- [S5] The shopping center was especially crowded, with 1,000鈥?,000 people, including many children who had come to see a clown performance.
- [S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.

### 7. Medium | keep=True | doc=28a13a10cb57f8245b1f98270bad9860

- Title: Territorial era of Minnesota
- Path: asserted [Statement] S16 -> declined [Agree_or_refuse_to_act] S9 -> formed [Coming_to_be] S1
- Relations: TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- Gold answer phrase: `The Minnesota Territory itself was formed only in 1849`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=2, single_sentence_risk=high, recommended=medium
- LLM reason: The final event can be clearly questioned and answered based on the supporting context.

Supporting sentences:
- [S0] The territorial era of Minnesota lasted from the Louisiana Purchase in 1803 to Minnesota's achieving statehood in 1858.
- [S1] The Minnesota Territory itself was formed only in 1849 but the area had a rich history well before this.
- [S2] Though there was a long history of European presence in the area before 19th century, it was during the 19th century that the United States began to establish a firm presence in what would become Minnesota.
- [S8] The Dakota Sioux, and later the Ojibwe, tribes hunted and gathered pelts trading with French, British, and later American traders at Grand Portage, Mendota, and other sites.
- [S9] This trade gradually declined during the early 19th century as demand for furs in Europe diminished.
- [S10] The lumber industry grew rapidly, replacing furs as the key economic resource.
- [S15] At the time the U.S. took possession of the region, Native Americans were by far the largest ethnic groups.
- [S16] Their role in the fur trade gave them a steady stream of income and significant political influence even as the French, British, and Americans asserted territorial claims on the area.
- [S17] French and British traders had mixed with native society in the area for many decades peacefully contributing to the society and creating new ethnic groups consisting of mixed-race peoples.

### 8. Medium | keep=True | doc=f28bce270df5a122c09365002d247e76

- Title: United States occupation of Nicaragua
- Path: began [Process_start] S1 -> assumed [Choosing] S3 -> opposed [Agree_or_refuse_to_act] S4
- Relations: TEMPORAL/BEFORE -> CAUSE/PRECONDITION
- Gold answer phrase: `President Herbert Hoover (1929鈥?933) opposed the relationship`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=2, single_sentence_risk=high, recommended=medium
- LLM reason: The path supports a clear question about Hoover's opposition, and the answer can be derived from the final event sentence.

Supporting sentences:
- [S0] The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.
- [S1] The formal occupation began in 1912, even though there were various other assaults by the U.S. in Nicaragua throughout this period.
- [S2] American military interventions in Nicaragua were designed to stop any other nation except the United States of America from building a Nicaraguan Canal.
- [S3] Nicaragua assumed a quasi-protectorate status under the 1916 Bryan鈥揅hamorro Treaty.
- [S4] President Herbert Hoover (1929鈥?933) opposed the relationship.
- [S5] Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention.

### 9. Medium | keep=True | doc=28a13a10cb57f8245b1f98270bad9860

- Title: Territorial era of Minnesota
- Path: transition [Change] S14 -> replacing [Change_of_leadership] S10 -> changed [Exchange] S20
- Relations: TEMPORAL/CONTAINS -> TEMPORAL/BEFORE
- Gold answer phrase: `Large waves of immigration in the 1850s very suddenly changed the demographics`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=2, single_sentence_risk=high, recommended=medium
- LLM reason: The path supports a clear question about how immigration changed demographics, and the final event can be answered from the provided answer sentence.

Supporting sentences:
- [S9] This trade gradually declined during the early 19th century as demand for furs in Europe diminished.
- [S10] The lumber industry grew rapidly, replacing furs as the key economic resource.
- [S11] Grain production began to develop late during this time as an emerging economic basis as well.
- [S13] By the end of the era east-central Minnesota had replaced northern Minnesota as the economic center of the area.
- [S14] This era was also as a period of cultural transition.
- [S15] At the time the U.S. took possession of the region, Native Americans were by far the largest ethnic groups.
- [S19] The economic influence of the Native Americans diminished and American territorial ideology increasingly sought to limit their influence.
- [S20] Large waves of immigration in the 1850s very suddenly changed the demographics so that within a few years the population shifted from predominantly native to predominantly people of European descent.
- [S21] The native and mixed-race populations continued to influence the territory's culture and politics, even at the end of the territorial era, though by the time statehood was achieved that influence was in steep decline.

### 10. Medium | keep=True | doc=6dabade56742b6040cda6a5838176f6c

- Title: Who's That Girl World Tour
- Path: broadcast [Expressing_publicly] S13 -> reaching [Arriving] S2 -> make [Manufacturing] S3
- Relations: TEMPORAL/BEFORE -> TEMPORAL/CONTAINS
- Gold answer phrase: `Girl Tour incorporated multimedia components to make the show more appealing`
- Answer phrase status: `complete`
- Prefilter: `True` / pass
- LLM judge: questionable=yes, steps=2, single_sentence_risk=high, recommended=medium
- LLM reason: The final event can be clearly questioned based on the context provided, and the answer can be derived from the answer sentence alone.

Supporting sentences:
- [S1] The tour supported her 1986 third studio album "True Blue", as well as the 1987 soundtrack "Who's That Girl".
- [S2] It was Madonna's first world tour, reaching Asia, North America and Europe.
- [S3] Musically and technically superior to her previous initiative, the Who's That Girl Tour incorporated multimedia components to make the show more appealing.
- [S4] Madonna trained physically doing aerobics, jogging and weight-lifting, to cope with the choreography and the dance routines.
- [S12] According to Pollstar, it was the second highest-grossing female concert tour of 1987, behind Tina Turner's Break Every Rule Tour.
- [S13] Who's That Girl was broadcast in a number of international television channels and was released in VHS titled "".
- [S14] Biographer J. Randy Taraborrelli commented that "Many female artists behave like a diva for a period when they reach superstar status, and the 'Who's That Girl?'

### 11. Hard | keep=False | doc=3dcfd60153822a6a8f6a516f161fc506

- Title: Battle of Malacca (1641)
- Path: Battle [Hostile_encounter] S0 -> began [Process_start] S1 -> siege [Besieging] S4 -> led [Conquering] S4
- Relations: TEMPORAL/BEFORE -> TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- Gold answer phrase: `The first serious attempt was the siege of Malacca in 1606 by the third VOC fleet from Holland with eleven ships, led by Admiral Cornelis Matelief de Jonge`
- Answer phrase status: `complete`
- Prefilter: `True` / pass [risk: temporal_only_hard]
- LLM judge: questionable=yes, steps=3+, single_sentence_risk=high, recommended=hard
- LLM reason: The path supports a clear question about the siege led by Admiral Cornelis Matelief de Jonge, requiring multiple events for context.

Supporting sentences:
- [S0] The Battle of Malacca (2 August 1640 鈥?14 January 1641) was a successful attempt by the Dutch to capture Malacca from the Portuguese.
- [S1] In the early 17th century, the Dutch East India Company ("Verenigde Oostindische Compagnie", "VOC") began the campaign to destroy Portuguese power in the East.
- [S2] At that time, the Portuguese had transformed Malacca into an impregnable fortress (the "Fortaleza de Malaca"), controlling access to the sea lanes of the Straits of Malacca and the spice trade there.
- [S3] The Dutch started by launching small incursions and skirmishes against the Portuguese.
- [S4] The first serious attempt was the siege of Malacca in 1606 by the third VOC fleet from Holland with eleven ships, led by Admiral Cornelis Matelief de Jonge that lead to the naval battle of Cape Rachado.
- [S5] Although the Dutch were routed, the Portuguese fleet of Martim Afonso de Castro, the Viceroy of Portuguese India, suffered heavier casualties and the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch.

### 12. Hard | keep=False | doc=f46091471f38006751fcdcda15d5775b

- Title: King David Hotel bombing
- Path: planned [Arranging] S3 -> planted [Placing] S5 -> carried out [Attack] S3 -> killed [Killing] S1
- Relations: TEMPORAL/BEFORE -> TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- Gold answer phrase: `91 people of various nationalities were killed`
- Answer phrase status: `complete`
- Prefilter: `True` / pass [risk: temporal_only_hard]
- LLM judge: questionable=yes, steps=3+, single_sentence_risk=high, recommended=hard
- LLM reason: The path supports a clear question about the final event of 'killed' with sufficient context from multiple events leading to it.

Supporting sentences:
- [S0] The King David Hotel bombing was a terrorist attack carried out on Monday, July 22, 1946, by the militant right-wing Zionist underground organization the Irgun on the British administrative headquarters for Palestine, which was housed in the southern wing of the King David Hotel in Jerusalem during the Jewish insurgency in Mandatory Palestine.
- [S1] 91 people of various nationalities were killed, and 46 were injured.
- [S2] The hotel was the site of the central offices of the British Mandatory authorities of Palestine, principally the Secretariat of the Government of Palestine and the Headquarters of the British Armed Forces in Palestine and Transjordan.
- [S3] When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this had been cancelled by the time the operation was carried out.
- [S4] It was conceived as part of a response to Operation Agatha (a series of widespread raids, including one on the Jewish Agency, conducted by the British authorities) and was the deadliest directed at the British during the Mandate era (1920鈥?948).
- [S5] Disguised as Arab workmen and as hotel waiters, members of the Irgun planted a bomb in the basement of the main building of the hotel, whose southern wing housed the Mandate Secretariat and a few offices of the British military headquarters.
- [S6] The resulting explosion caused the collapse of the western half of the southern wing of the hotel.

### 13. Hard | keep=False | doc=e253b7fd1109bd5f87966022eea7762f

- Title: Myyrmanni bombing
- Path: crowded [Come_together] S5 -> come to [Motion] S5 -> died [Death] S2 -> investigated [Criminal_investigation] S6
- Relations: TEMPORAL/OVERLAP -> TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- Gold answer phrase: `The incident was investigated primarily as six accounts of murder`
- Answer phrase status: `complete`
- Prefilter: `True` / pass [risk: temporal_only_hard]
- LLM judge: questionable=partial, steps=3+, single_sentence_risk=high, recommended=hard
- LLM reason: The final event can be questioned, but the answer is directly found in the answer sentence, making it less complex than intended.

Supporting sentences:
- [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
- [S2] In total seven died, including two teenagers and a 7-year-old.
- [S3] 166 people were injured, including 10 children.
- [S4] 66 victims required hospitalization with the remainder treated and released at the scene.
- [S5] The shopping center was especially crowded, with 1,000鈥?,000 people, including many children who had come to see a clown performance.
- [S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.
- [S7] His motive was not determined.

### 14. Hard | keep=False | doc=c0c67db40cd5e2e03645ff1116fafcfc

- Title: Cherry Valley massacre
- Path: permitted [Preventing_or_letting] S8 -> took place [Process_start] S10 -> descended on [Motion_directional] S2 -> drove [Motion] S12
- Relations: TEMPORAL/BEFORE -> TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- Gold answer phrase: `drove the Iroquois out of western New York`
- Answer phrase status: `complete`
- Prefilter: `True` / pass [risk: temporal_only_hard]
- LLM judge: questionable=yes, steps=3+, single_sentence_risk=high, recommended=hard
- LLM reason: The final event can be clearly questioned and is supported by the context, but it requires understanding multiple steps leading to it.

Supporting sentences:
- [S1] It has been described as one of the most horrific frontier massacres of the war.
- [S2] A mixed force of Loyalists, British soldiers, Seneca and Mohawks descended on Cherry Valley, whose defenders, despite warnings, were unprepared for the attack.
- [S3] During the raid, the Seneca in particular targeted non-combatants, and reports state that 30 such individuals were slain, in addition to a number of armed defenders.
- [S7] Butler's authority with the Indians was undermined by his poor treatment of Joseph Brant, the leader of the Mohawks.
- [S8] Butler repeatedly maintained, against accusations that he permitted the atrocities to take place, that he was powerless to restrain the Seneca.
- [S9] During the campaigns of 1778, Brant achieved an undeserved reputation for brutality.
- [S10] He was not present at Wyoming 鈥?although many thought he was 鈥?and he actively sought to minimize the atrocities that took place at Cherry Valley.
- [S11] Diaries belonging to British soldiers during the campaign state the regiment as being the "butchers" and given that Butler was the overall commander of the expedition, there is controversy as to who actually ordered or failed to restrain the killings.
- [S12] The massacre contributed to calls for reprisals, leading to the 1779 Sullivan Expedition which drove the Iroquois out of western New York.

### 15. Hard | keep=False | doc=db50381e7d1dd4a41fb4ac60eaebe3a4

- Title: Battle of Orthez
- Path: end [Process_end] S3 -> surrounded [Surrounding] S6 -> repelled [Defending] S1 -> conducted [Action] S2
- Relations: TEMPORAL/BEFORE -> TEMPORAL/BEFORE -> TEMPORAL/BEFORE
- Gold answer phrase: `At first the withdrawal was conducted in good order`
- Answer phrase status: `complete`
- Prefilter: `True` / pass [risk: temporal_only_hard]
- LLM judge: questionable=yes, steps=3+, single_sentence_risk=high, recommended=hard
- LLM reason: The final event can be clearly questioned and is supported by the preceding events, but the answer is directly found in the answer sentence.

Supporting sentences:
- [S0] The Battle of Orthez (27 February 1814) saw the Anglo-Portuguese Army under Field Marshal Arthur Wellesley, Marquess of Wellington attack an Imperial French army led by Marshal Nicolas Soult in southern France.
- [S1] The outnumbered French repelled several Allied assaults on their right flank, but their center and left flank were overcome and Soult was compelled to retreat.
- [S2] At first the withdrawal was conducted in good order, but it eventually ended in a scramble for safety and many French soldiers became prisoners.
- [S3] The engagement occurred near the end of the Peninsular War.
- [S4] In mid-February, Wellington's army broke out of its small area of conquered territory near Bayonne.
- [S5] Moving east, the Allies drove the French back from several river lines.
- [S6] After a pause in the campaign, the western-most Allied corps surrounded and isolated Bayonne.
- [S7] Resuming their eastward drive, the remaining two Allied corps pushed Soult's army back to Orthez where the French marshal offered battle.

## Prefilter Rejected Examples

### R1. Medium | doc=c0c67db40cd5e2e03645ff1116fafcfc
- Path: ordered [Arranging] S11 -> took place [Process_start] S10 -> given [Giving] S11
- Gold answer phrase: `given`
- Answer phrase status: `invalid`
- Reject reason: answer_phrase_fail: phrase equals trigger

### R2. Medium | doc=f28bce270df5a122c09365002d247e76
- Path: began [Process_start] S1 -> opposed [Agree_or_refuse_to_act] S4 -> ended [Process_end] S5
- Gold answer phrase: `Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention`
- Answer phrase status: `partial`
- Reject reason: hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)

### R3. Hard | doc=f28bce270df5a122c09365002d247e76
- Path: began [Process_start] S1 -> assumed [Choosing] S3 -> opposed [Agree_or_refuse_to_act] S4 -> ended [Process_end] S5
- Gold answer phrase: `Finally in 1933 President Franklin D Roosevelt, invoking his new Good Neighbor policy ended American intervention`
- Answer phrase status: `partial`
- Reject reason: hard_weak_trigger='ended'; answer_phrase_fail: partial extraction (no clause boundary found)

### R4. Easy | doc=3dcfd60153822a6a8f6a516f161fc506
- Path: assaulted [Attack] S6 -> destroyed [Destroying] S7
- Gold answer phrase: `This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese power, removing their influence in the Malay archipelago`
- Answer phrase status: `partial`
- Reject reason: answer_phrase_fail: partial extraction (no clause boundary found)

### R5. Medium | doc=3dcfd60153822a6a8f6a516f161fc506
- Path: capture [Conquering] S0 -> destroyed [Destroying] S7 -> removing [Removing] S7
- Gold answer phrase: `This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese power, removing their influence in the Malay archipelago`
- Answer phrase status: `partial`
- Reject reason: answer_phrase_fail: partial extraction (no clause boundary found)

### R6. Medium | doc=3dcfd60153822a6a8f6a516f161fc506
- Path: rallied [Filling] S5 -> control [Control] S8 -> took [Conquering] S8
- Gold answer phrase: `In line with the agreement with Johor in 1606, the Dutch took control of Malacca`
- Answer phrase status: `complete`
- Reject reason: hard_weak_trigger='took'

### R7. Easy | doc=e253b7fd1109bd5f87966022eea7762f
- Path: crowded [Come_together] S5 -> took place [Process_start] S0
- Gold answer phrase: `The Myyrmanni bombing took place on October 11, 2002, in Myyrm盲ki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall`
- Answer phrase status: `partial`
- Reject reason: hard_weak_trigger='took place'; answer_phrase_fail: partial extraction (no clause boundary found)

### R8. Hard | doc=28a13a10cb57f8245b1f98270bad9860
- Path: diminished [Cause_change_of_position_on_a_scale] S9 -> declined [Agree_or_refuse_to_act] S9 -> became [Becoming] S12 -> replaced [Change_of_leadership] S13
- Gold answer phrase: `By the end of the era east-central Minnesota had replaced northern Minnesota as the economic center of the area`
- Answer phrase status: `partial`
- Reject reason: answer_phrase_fail: partial extraction (no clause boundary found)

### R9. Hard | doc=28a13a10cb57f8245b1f98270bad9860
- Path: diminished [Cause_change_of_position_on_a_scale] S9 -> declined [Agree_or_refuse_to_act] S9 -> transition [Change] S14 -> become [Becoming] S2
- Gold answer phrase: `the United States began to establish a firm presence in what would become Minnesota`
- Answer phrase status: `complete`
- Reject reason: hard_weak_trigger='become'
