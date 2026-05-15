# Manual Audit Sample

**Source:** `outputs\runs\path_filter_strict_pilot`

**Strict total:** 76 (Easy 32, Medium 35, Hard 9)
**Relaxed-only Hard:** 12
**Rejected Hard:** 7

## Easy Strict Kept (5 samples)

### Easy #1

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **title:** Death and state funeral of Raúl Alfonsín
- **difficulty:** Easy
- **event path:** arranged/Arranging -> mourning/Rite
- **relation sequence:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `arranged three days of national mourning`
- **answer_sentence:** Vice president Julio Cobos, the acting president at the time, arranged three days of national mourning and a state funeral at the Palace of the Argent
- **supporting_sentences:**
  - [S2] He had lung cancer and died at his home; a massive candlelight vigil took place in the vicinity of it.
  - [S3] Vice president Julio Cobos, the acting president at the time, arranged three days of national mourning and a state funeral at the Palace of the Argent
  - [S4] Alfonsín was seen by 40,000 people and the senior politicians of the country; people from other countries also voiced their respect for him.
- **LLM judge:** pq=yes risk=high steps=1 rec=easy path_dep=partial
- **judge reason:** The final event can be clearly questioned based on the provided answer sentence, but the context of the earlier event is not strictly necessary for understanding the question.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Easy #2

- **doc_id:** `06f91ced00b41867979f3d5dc6996da2`
- **title:** Operation Vengeance
- **difficulty:** Easy
- **event path:** attack/Attack -> blamed/Judgment_communication
- **relation sequence:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `blamed Yamamoto for the attack on Pearl Harbor`
- **answer_sentence:** The death of Yamamoto reportedly damaged the morale of Japanese naval personnel, raised the morale of the Allied forces, and was intended as revenge b
- **supporting_sentences:**
  - [S2] The mission of the U.S. aircraft was specifically to kill Yamamoto and was based on United States Navy intelligence on Yamamoto's itinerary in the Sol
  - [S3] The death of Yamamoto reportedly damaged the morale of Japanese naval personnel, raised the morale of the Allied forces, and was intended as revenge b
  - [S4] The U.S. pilots claimed to have shot down three twin-engined bombers and two fighters during the mission, but Japanese sources show only two bombers w
- **LLM judge:** pq=yes risk=high steps=1 rec=easy path_dep=partial
- **judge reason:** The final event can be questioned directly, but context about the attack enhances understanding.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Easy #3

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Easy
- **event path:** Disguised/Wearing -> operation/Military_operation
- **relation sequence:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `this had been cancelled by the time the operation was carried out`
- **answer_sentence:** When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this
- **supporting_sentences:**
  - [S2] The hotel was the site of the central offices of the British Mandatory authorities of Palestine, principally the Secretariat of the Government of Pale
  - [S3] When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this
  - [S4] It was conceived as part of a response to Operation Agatha (a series of widespread raids, including one on the Jewish Agency, conducted by the British
  - [S5] Disguised as Arab workmen and as hotel waiters, members of the Irgun planted a bomb in the basement of the main building of the hotel, whose southern 
  - [S6] The resulting explosion caused the collapse of the western half of the southern wing of the hotel.
- **LLM judge:** pq=partial risk=high steps=1 rec=medium path_dep=partial
- **judge reason:** The final event can be inferred from the answer sentence, but the context of the earlier event adds nuance that could enhance understanding.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Easy #4

- **doc_id:** `f83fc49c020ec542b16d463b3f7c2c14`
- **title:** Sinjar massacre
- **difficulty:** Easy
- **event path:** started/Process_start -> killing/Killing
- **relation sequence:** CAUSE/PRECONDITION
- **gold_answer_phrase:** `The Sinjar massacre was the genocidal killing`
- **answer_sentence:** The Sinjar massacre was the genocidal killing and abduction of thousands of Yazidi men in Sinjar ( "Şingal") city and Sinjar District in Iraq's Nineve
- **supporting_sentences:**
  - [S0] The Sinjar massacre was the genocidal killing and abduction of thousands of Yazidi men in Sinjar ( "Şingal") city and Sinjar District in Iraq's Nineve
  - [S1] This event started with ISIL attacking and capturing Sinjar and neighboring towns on 3 August, during ISIL's offensive in early August 2014.
  - [S2] Dr Noori Abdulrahman, head of the Department of Coordination and Follow-up of the Iraqi Kurdistan Regional Government, stated that ISIL's 3 August cam
- **LLM judge:** pq=yes risk=high steps=1 rec=easy path_dep=yes
- **judge reason:** The final event can be clearly questioned based on the supporting context, and the answer is directly found in the answer sentence.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Easy #5

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Easy
- **event path:** crowded/Come_together -> exploded/Attack
- **relation sequence:** TEMPORAL/BEFORE
- **gold_answer_phrase:** `A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five`
- **answer_sentence:** A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
- **supporting_sentences:**
  - [S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.
  - [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
  - [S2] In total seven died, including two teenagers and a 7-year-old.
  - [S4] 66 victims required hospitalization with the remainder treated and released at the scene.
  - [S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.
- **LLM judge:** pq=yes risk=high steps=1 rec=easy path_dep=partial
- **judge reason:** The final event can be clearly questioned based on the provided answer sentence, but the context of the crowded shopping center adds some depth that could be referenced in a question.
- **strict_reason:** keep
- **relaxed_reason:** keep

## Medium Strict Kept (5 samples)

### Medium #1

- **doc_id:** `f83fc49c020ec542b16d463b3f7c2c14`
- **title:** Sinjar massacre
- **difficulty:** Medium
- **event path:** started/Process_start -> enabled/Preventing_or_letting -> evacuated/Emptying
- **relation sequence:** TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `fled into the Sinjar Mountains to be evacuated`
- **answer_sentence:** The assistance of PKK and YPG enabled the majority of the 50,000 Yazidis who fled into the Sinjar Mountains to be evacuated.
- **supporting_sentences:**
  - [S0] The Sinjar massacre was the genocidal killing and abduction of thousands of Yazidi men in Sinjar ( "Şingal") city and Sinjar District in Iraq's Nineve
  - [S1] This event started with ISIL attacking and capturing Sinjar and neighboring towns on 3 August, during ISIL's offensive in early August 2014.
  - [S2] Dr Noori Abdulrahman, head of the Department of Coordination and Follow-up of the Iraqi Kurdistan Regional Government, stated that ISIL's 3 August cam
  - [S4] On 8 August 2014, the United States reacted with airstrikes on ISIL units and convoys in northern Iraq, which led to a war of several countries agains
  - [S5] The assistance of PKK and YPG enabled the majority of the 50,000 Yazidis who fled into the Sinjar Mountains to be evacuated.
- **LLM judge:** pq=yes risk=high steps=2 rec=medium path_dep=yes
- **judge reason:** The final event can be questioned in relation to the earlier events, specifically how the assistance of PKK and YPG enabled the evacuation after the Yazidis fled.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Medium #2

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Medium
- **event path:** allowed/Preventing_or_letting -> detain/Preventing_or_letting -> lifted/Process_end
- **relation sequence:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `President Arroyo lifted the state of emergency on March`
- **answer_sentence:** President Arroyo lifted the state of emergency on March 3, 2006 by the virtue of Proclamation No.
- **supporting_sentences:**
  - [S3] State security services also claimed that it had arrested a general who was involved in the coup attempt.
  - [S4] President Arroyo lifted the state of emergency on March 3, 2006 by the virtue of Proclamation No.
  - [S5] 1021.
  - [S7] The government, informally known as "Malacañang", after the presidential palace, also suspended all public activities on the same day and even on succ
  - [S8] Under the provisions of the 1987 Constitution, the government was allowed at the moment to detain anyone indefinitely without the privilege of the wri
- **LLM judge:** pq=yes risk=high steps=2 rec=medium path_dep=yes
- **judge reason:** The path supports a clear question about the final event, and the answer can be derived from the context provided.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Medium #3

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Medium
- **event path:** claimed/Statement -> occurred/Presence -> revocation/Process_end
- **relation sequence:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `an immediate revocation on all licenses and permits to hold demonstrations and protests`
- **answer_sentence:** The state of national emergency also led to a temporary suspension of lower-level education classes and an immediate revocation on all licenses and pe
- **supporting_sentences:**
  - [S1] 1017.
  - [S2] This occurred after the government claimed that it foiled an alleged coup d'état attempt against the rule of President Gloria Macapagal-Arroyo earlier
  - [S3] State security services also claimed that it had arrested a general who was involved in the coup attempt.
  - [S5] 1021.
  - [S6] The state of national emergency also led to a temporary suspension of lower-level education classes and an immediate revocation on all licenses and pe
- **LLM judge:** pq=yes risk=high steps=2 rec=medium path_dep=yes
- **judge reason:** The final event can be questioned in relation to the preceding events, making it clear and answerable.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Medium #4

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **title:** Death and state funeral of Raúl Alfonsín
- **difficulty:** Medium
- **event path:** died/Death -> took/Bringing -> left/Placing
- **relation sequence:** CAUSE/PRECONDITION, CAUSE/PRECONDITION
- **gold_answer_phrase:** `left him at the pantheon for the veterans of the Revolution of the Park`
- **answer_sentence:** A military escort took his coffin to the La Recoleta Cemetery, and left him at the pantheon for the veterans of the Revolution of the Park.
- **supporting_sentences:**
  - [S0] Raúl Alfonsín was the president of Argentina from 1983 to 1989.
  - [S1] He died on March 31, 2009, aged 82.
  - [S2] He had lung cancer and died at his home; a massive candlelight vigil took place in the vicinity of it.
  - [S4] Alfonsín was seen by 40,000 people and the senior politicians of the country; people from other countries also voiced their respect for him.
  - [S5] A military escort took his coffin to the La Recoleta Cemetery, and left him at the pantheon for the veterans of the Revolution of the Park.
- **LLM judge:** pq=yes risk=high steps=2 rec=medium path_dep=yes
- **judge reason:** The final event can be questioned in relation to the preceding events, providing context for the placement of the coffin.
- **strict_reason:** keep
- **relaxed_reason:** keep

### Medium #5

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Medium
- **event path:** changes/Change -> quipped/Statement -> proved/Convincing
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `The Crimean War proved to be the moment of truth for Nikolaevan Russia`
- **answer_sentence:** The Crimean War proved to be the moment of truth for Nikolaevan Russia.
- **supporting_sentences:**
  - [S7] Britain attempted to mediate and arranged a compromise that Nicholas agreed to.
  - [S8] When the Ottomans demanded changes, Nicholas refused and prepared for war.
  - [S9] Having obtained promises of support from France and Britain, the Ottomans declared war on Russia in October 1853.
  - [S15] Aside from a minor skirmish at Köstence (today Constanța), there was little for the allies to do.
  - [S16] Karl Marx quipped, "there they are, the French doing nothing and the British helping them as fast as possible".
- **LLM judge:** pq=yes risk=high steps=2 rec=medium path_dep=yes
- **judge reason:** The final event can be questioned in relation to earlier events, providing context for its significance.
- **strict_reason:** keep
- **relaxed_reason:** keep

## Hard Strict Kept (10 samples)

### Hard Strict #1

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event path:** stopped/Preventing_or_letting -> destroyed/Destroying -> rushed/Motion -> forbade/Preventing_or_letting
- **relation sequence:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `It forbade Russia from basing warships in the Black Sea`
- **answer_sentence:** It forbade Russia from basing warships in the Black Sea.
- **supporting_sentences:**
  - [S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman 
  - [S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
  - [S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russ
  - [S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
  - [S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The final event can be questioned in relation to the preceding events, making it suitable for a clear question.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #2

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Hard
- **event path:** crowded/Come_together -> took place/Process_start -> investigated/Criminal_investigation -> closed/Self_motion
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `closed in January 2003 without any indictments as Gerdt was the sole suspect`
- **answer_sentence:** The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.
- **supporting_sentences:**
  - [S0] The Myyrmanni bombing took place on October 11, 2002, in Myyrmäki, Vantaa, Finland, in Greater Helsinki, at the Myyrmanni shopping mall.
  - [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
  - [S4] 66 victims required hospitalization with the remainder treated and released at the scene.
  - [S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.
  - [S6] The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as Gerdt was the sole suspect.
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The final event 'closed' can be questioned in relation to the investigation and the circumstances surrounding it, requiring understanding of the earlier events.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #3

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Hard
- **event path:** warnings/Warning -> called/Communication -> carried out/Attack -> killed/Killing
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `91 people of various nationalities were killed`
- **answer_sentence:** 91 people of various nationalities were killed, and 46 were injured.
- **supporting_sentences:**
  - [S0] The King David Hotel bombing was a terrorist attack carried out on Monday, July 22, 1946, by the militant right-wing Zionist underground organization 
  - [S1] 91 people of various nationalities were killed, and 46 were injured.
  - [S2] The hotel was the site of the central offices of the British Mandatory authorities of Palestine, principally the Secretariat of the Government of Pale
  - [S3] When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this
  - [S4] It was conceived as part of a response to Operation Agatha (a series of widespread raids, including one on the Jewish Agency, conducted by the British
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The path supports a clear question about the final event, requiring knowledge of the preceding events.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #4

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event path:** uprising/Change_of_leadership -> negotiate/Communication -> refused/Agree_or_refuse_to_act -> killed/Killing
- **relation sequence:** TEMPORAL/CONTAINS, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `After a short hand-to-hand fight the Russian commander was killed`
- **answer_sentence:** After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **supporting_sentences:**
  - [S2] 100 men under Aleksander Rogaliński and a company of the Russian Murom Regiment under Col. Kozlaninov, the skirmish resulted in Polish victory.
  - [S3] As the engagement started on the very first day of the uprising, the Russian force still obeyed the orders of marching through the occupied country wi
  - [S4] When the Russians approached a local manor in which the Poles had their quarters, the Russian commander ordered a loose formation and tried to negotia
  - [S5] However, Rogaliński refused to negotiate and ordered a charge of the Russians.
  - [S6] After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The path provides a clear sequence of events leading to the final event, allowing for a natural question about the circumstances of the Russian commander's death.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #5

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event path:** stopped/Preventing_or_letting -> destroyed/Destroying -> arriving/Arriving -> signed on/Sign_agreement
- **relation sequence:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
- **gold_answer_phrase:** `signed on 30 March`
- **answer_sentence:** The Treaty of Paris, signed on 30 March 1856, ended the war.
- **supporting_sentences:**
  - [S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman 
  - [S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
  - [S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russ
  - [S13] Fearing an Ottoman collapse, France and Britain rushed forces to Gallipoli.
  - [S14] They then moved north to Varna in June 1854, arriving just in time for the Russians to abandon Silistra.
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The final event can be questioned in relation to the preceding events, and the answer is clear from the context.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #6

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event path:** uprising/Change_of_leadership -> approached/Arriving -> refused/Agree_or_refuse_to_act -> ordered/Arranging
- **relation sequence:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `ordered a charge of the Russians`
- **answer_sentence:** However, Rogaliński refused to negotiate and ordered a charge of the Russians.
- **supporting_sentences:**
  - [S2] 100 men under Aleksander Rogaliński and a company of the Russian Murom Regiment under Col. Kozlaninov, the skirmish resulted in Polish victory.
  - [S3] As the engagement started on the very first day of the uprising, the Russian force still obeyed the orders of marching through the occupied country wi
  - [S4] When the Russians approached a local manor in which the Poles had their quarters, the Russian commander ordered a loose formation and tried to negotia
  - [S5] However, Rogaliński refused to negotiate and ordered a charge of the Russians.
  - [S6] After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The final event 'ordered a charge of the Russians' can be questioned in the context of the preceding events, making it suitable for a clear question.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #7

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **title:** Operation Deny Flight
- **difficulty:** Hard
- **event path:** began/Process_start -> providing/Supply -> helped/Assistance -> adapted/Coming_to_be
- **relation sequence:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engagement on the plains of Central Europe`
- **answer_sentence:** These engagements helped show that NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engag
- **supporting_sentences:**
  - [S0] Operation Deny Flight was a North Atlantic Treaty Organization (NATO) operation that began on 12 April 1993 as the enforcement of a United Nations (UN
  - [S1] The United Nations and NATO later expanded the mission of the operation to include providing close air support for UN troops in Bosnia and carrying ou
  - [S2] Twelve NATO members contributed forces to the operation and, by its end on 20 December 1995, NATO pilots had flown 100,420 sorties.
  - [S4] The operation included the first combat engagement in NATO's history, a 28 February 1994 air battle over Banja Luka, and in April 1994, NATO aircraft 
  - [S5] These engagements helped show that NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engag
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The final event can be questioned in relation to earlier events, and the answer is clear from the context.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #8

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **title:** Battle of Malacca (1641)
- **difficulty:** Hard
- **event path:** began/Process_start -> launching/Military_operation -> took/Conquering -> agreed/Agree_or_refuse_to_act
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
- **gold_answer_phrase:** `agreed not to seek territories or wage war with the Malay kingdoms`
- **answer_sentence:** In line with the agreement with Johor in 1606, the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdom
- **supporting_sentences:**
  - [S0] The Battle of Malacca (2 August 1640 – 14 January 1641) was a successful attempt by the Dutch to capture Malacca from the Portuguese.
  - [S1] In the early 17th century, the Dutch East India Company ("Verenigde Oostindische Compagnie", "VOC") began the campaign to destroy Portuguese power in 
  - [S2] At that time, the Portuguese had transformed Malacca into an impregnable fortress (the "Fortaleza de Malaca"), controlling access to the sea lanes of 
  - [S3] The Dutch started by launching small incursions and skirmishes against the Portuguese.
  - [S4] The first serious attempt was the siege of Malacca in 1606 by the third VOC fleet from Holland with eleven ships, led by Admiral Cornelis Matelief de 
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The final event can be questioned in relation to the earlier events, and the answer is clear from the context.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

### Hard Strict #9

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Hard
- **event path:** Moving/Motion -> surrounded/Surrounding -> pushed/Motion -> battle/Hostile_encounter
- **relation sequence:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the French marshal offered battle`
- **answer_sentence:** Resuming their eastward drive, the remaining two Allied corps pushed Soult's army back to Orthez where the French marshal offered battle.
- **supporting_sentences:**
  - [S4] In mid-February, Wellington's army broke out of its small area of conquered territory near Bayonne.
  - [S5] Moving east, the Allies drove the French back from several river lines.
  - [S6] After a pause in the campaign, the western-most Allied corps surrounded and isolated Bayonne.
  - [S7] Resuming their eastward drive, the remaining two Allied corps pushed Soult's army back to Orthez where the French marshal offered battle.
  - [S8] In subsequent operations, Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse.
- **LLM judge:** pq=yes risk=high steps=3+ rec=hard path_dep=yes
- **judge reason:** The path supports a clear question about the final event, requiring understanding of the preceding events.
- **strict_reason:** keep_path_dep_yes
- **relaxed_reason:** keep_path_dep_yes

## Hard Relaxed-Only (partial) (10 samples)

These are Hard paths with `can_write_path_dependent_question=partial`.

### Hard Partial #1

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event path:** uprising/Change_of_leadership -> refused/Agree_or_refuse_to_act -> fight/Hostile_encounter -> losses/Earnings_and_losses
- **relation sequence:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, CAUSE/CAUSE
- **gold_answer_phrase:** `Polish losses were`
- **answer_sentence:** Polish losses were negligible, but the Polish commander was wounded and lost his eye.
- **supporting_sentences:**
  - [S2] 100 men under Aleksander Rogaliński and a company of the Russian Murom Regiment under Col. Kozlaninov, the skirmish resulted in Polish victory.
  - [S3] As the engagement started on the very first day of the uprising, the Russian force still obeyed the orders of marching through the occupied country wi
  - [S4] When the Russians approached a local manor in which the Poles had their quarters, the Russian commander ordered a loose formation and tried to negotia
  - [S5] However, Rogaliński refused to negotiate and ordered a charge of the Russians.
  - [S6] After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned, but the answer phrase is truncated and lacks clarity.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #2

- **doc_id:** `28a13a10cb57f8245b1f98270bad9860`
- **title:** Territorial era of Minnesota
- **difficulty:** Hard
- **event path:** become/Becoming -> formed/Coming_to_be -> changed/Exchange -> influence/Influence
- **relation sequence:** TEMPORAL/SIMULTANEOUS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `mixed-race populations continued to influence the territory's culture and`
- **answer_sentence:** The native and mixed-race populations continued to influence the territory's culture and politics, even at the end of the territorial era, though by t
- **supporting_sentences:**
  - [S0] The territorial era of Minnesota lasted from the Louisiana Purchase in 1803 to Minnesota's achieving statehood in 1858.
  - [S1] The Minnesota Territory itself was formed only in 1849 but the area had a rich history well before this.
  - [S2] Though there was a long history of European presence in the area before 19th century, it was during the 19th century that the United States began to e
  - [S3] Many of the facets of Minnesota culture that are perceived as the area's "early" history in fact originated after this period.
  - [S19] The economic influence of the Native Americans diminished and American territorial ideology increasingly sought to limit their influence.
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned but requires context from earlier events for clarity.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #3

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **title:** King David Hotel bombing
- **difficulty:** Hard
- **event path:** warnings/Warning -> carried out/Attack -> carried out/Attack -> injured/Bodily_harm
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/SIMULTANEOUS, TEMPORAL/BEFORE
- **gold_answer_phrase:** `46 were injured`
- **answer_sentence:** 91 people of various nationalities were killed, and 46 were injured.
- **supporting_sentences:**
  - [S0] The King David Hotel bombing was a terrorist attack carried out on Monday, July 22, 1946, by the militant right-wing Zionist underground organization 
  - [S1] 91 people of various nationalities were killed, and 46 were injured.
  - [S2] The hotel was the site of the central offices of the British Mandatory authorities of Palestine, principally the Secretariat of the Government of Pale
  - [S3] When planned, the attack had the approval of the Haganah, the principal Jewish paramilitary group in Palestine, though, unbeknownst to the Irgun, this
  - [S4] It was conceived as part of a response to Operation Agatha (a series of widespread raids, including one on the Jewish Agency, conducted by the British
- **LLM judge:** pq=partial risk=high steps=2 rec=medium path_dep=partial
- **judge reason:** The final event can be answered from the sentence, but context about the attack is needed for a full understanding.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #4

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **title:** Battle of Orthez
- **difficulty:** Hard
- **event path:** Moving/Motion -> isolated/Having_or_lacking_access -> overcome/Conquering -> conducted/Action
- **relation sequence:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `At first the withdrawal was conducted in good`
- **answer_sentence:** At first the withdrawal was conducted in good order, but it eventually ended in a scramble for safety and many French soldiers became prisoners.
- **supporting_sentences:**
  - [S0] The Battle of Orthez (27 February 1814) saw the Anglo-Portuguese Army under Field Marshal Arthur Wellesley, Marquess of Wellington attack an Imperial 
  - [S1] The outnumbered French repelled several Allied assaults on their right flank, but their center and left flank were overcome and Soult was compelled to
  - [S2] At first the withdrawal was conducted in good order, but it eventually ended in a scramble for safety and many French soldiers became prisoners.
  - [S3] The engagement occurred near the end of the Peninsular War.
  - [S4] In mid-February, Wellington's army broke out of its small area of conquered territory near Bayonne.
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned, but the proposed answer phrase is truncated and lacks clarity.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #5

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event path:** stopped/Preventing_or_letting -> depleting/Expend_resource -> ended/Process_end -> identify/Check
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `The humiliation forced Russia's educated elites to identify the Empire's problems and to recognize the need for fundamental transformations aimed at modernizing`
- **answer_sentence:** The humiliation forced Russia's educated elites to identify the Empire's problems and to recognize the need for fundamental transformations aimed at m
- **supporting_sentences:**
  - [S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman 
  - [S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
  - [S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russ
  - [S18] After extended preparations, the forces landed on the peninsula in September 1854 and marched their way to a point south of Sevastopol after the succe
  - [S19] The Russians counterattacked on 25 October in what became the Battle of Balaclava and were repulsed, but at the cost of seriously depleting the Britis
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned but relies heavily on context from earlier events.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #6

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Hard
- **event path:** titled/Name_conferral -> projected/Arranging -> reaching/Arriving -> incorporated/Cause_to_amalgamate
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `Girl Tour incorporated multimedia components to make the show more appealing`
- **answer_sentence:** Musically and technically superior to her previous initiative, the Who's That Girl Tour incorporated multimedia components to make the show more appea
- **supporting_sentences:**
  - [S1] The tour supported her 1986 third studio album "True Blue", as well as the 1987 soundtrack "Who's That Girl".
  - [S2] It was Madonna's first world tour, reaching Asia, North America and Europe.
  - [S3] Musically and technically superior to her previous initiative, the Who's That Girl Tour incorporated multimedia components to make the show more appea
  - [S4] Madonna trained physically doing aerobics, jogging and weight-lifting, to cope with the choreography and the dance routines.
  - [S7] Patrick Leonard, who was the music director, encouraged Madonna to go with the idea of remixing and presenting her older songs for the show.
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned, but the context of earlier events is not strictly necessary to understand the answer.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #7

- **doc_id:** `6fc6538bd2c19d34942f7f36274d83ae`
- **title:** Cyclone Winifred
- **difficulty:** Hard
- **event path:** came/Arriving -> producing/Creating -> winds/Motion -> aid/Assistance
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `requests for financial aid, and filings for unemployment benefits`
- **answer_sentence:** The Department of Social Security (DSS) sent employees to receive claims for damage, requests for financial aid, and filings for unemployment benefits
- **supporting_sentences:**
  - [S3] Meandering southward, the cyclone began to curve southeastward that evening before suddenly turning toward the coast, southwestward, on 31 January, st
  - [S4] By the time it came ashore near Silkwood, Queensland at 0445 UTC on 1 February, it was producing Category 3-force winds on the Australian tropical cyc
  - [S5] Weakening as it drifted inland, Winifred persisted as a tropical depression for another five days after landfall before finally dissipating on 5 Febru
  - [S14] Hundreds of State Emergency Service (SES) volunteers were deployed to restore electrical and water services, evacuate local citizens, provide food, an
  - [S15] The Department of Social Security (DSS) sent employees to receive claims for damage, requests for financial aid, and filings for unemployment benefits
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned but relies on context from earlier events.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #8

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **title:** Myyrmanni bombing
- **difficulty:** Hard
- **event path:** crowded/Come_together -> died/Death -> injured/Bodily_harm -> treated/Cure
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/OVERLAP, CAUSE/PRECONDITION
- **gold_answer_phrase:** `66 victims required hospitalization with the remainder treated`
- **answer_sentence:** 66 victims required hospitalization with the remainder treated and released at the scene.
- **supporting_sentences:**
  - [S1] A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five immediately, including Gerdt.
  - [S2] In total seven died, including two teenagers and a 7-year-old.
  - [S3] 166 people were injured, including 10 children.
  - [S4] 66 victims required hospitalization with the remainder treated and released at the scene.
  - [S5] The shopping center was especially crowded, with 1,000–2,000 people, including many children who had come to see a clown performance.
- **LLM judge:** pq=partial risk=high steps=2 rec=medium path_dep=partial
- **judge reason:** The final event can be questioned, but the context of injuries and deaths adds complexity that may not be fully captured in a single question.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #9

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **title:** Crimean War
- **difficulty:** Hard
- **event path:** stopped/Preventing_or_letting -> depleting/Expend_resource -> forbade/Preventing_or_letting -> granted/Preventing_or_letting
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `Christians there were granted a degree of official`
- **answer_sentence:** Christians there were granted a degree of official equality, and the Orthodox Church regained control of the Christian churches in dispute.
- **supporting_sentences:**
  - [S10] The war started in the Balkans in July 1853, when Russian troops occupied the Danubian Principalities (now part of Romania), which were under Ottoman 
  - [S11] Led by Omar Pasha, the Ottomans fought a strong defensive campaign and stopped the advance at Silistra.
  - [S12] A separate action on the fort town of Kars in eastern Anatolia led to a siege, and a Turkish attempt to reinforce the garrison was destroyed by a Russ
  - [S18] After extended preparations, the forces landed on the peninsula in September 1854 and marched their way to a point south of Sevastopol after the succe
  - [S19] The Russians counterattacked on 25 October in what became the Battle of Balaclava and were repulsed, but at the cost of seriously depleting the Britis
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned but relies on context from earlier events.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

### Hard Partial #10

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Hard
- **event path:** presenting/Presence -> incorporated/Cause_to_amalgamate -> marked/Recording -> commented/Statement
- **relation sequence:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `Randy Taraborrelli commented`
- **answer_sentence:** Biographer J. Randy Taraborrelli commented that "Many female artists behave like a diva for a period when they reach superstar status, and the 'Who's 
- **supporting_sentences:**
  - [S2] It was Madonna's first world tour, reaching Asia, North America and Europe.
  - [S3] Musically and technically superior to her previous initiative, the Who's That Girl Tour incorporated multimedia components to make the show more appea
  - [S4] Madonna trained physically doing aerobics, jogging and weight-lifting, to cope with the choreography and the dance routines.
  - [S6] The stage had four video screens, multimedia projectors and a flight of stairs in the middle.
  - [S7] Patrick Leonard, who was the music director, encouraged Madonna to go with the idea of remixing and presenting her older songs for the show.
- **LLM judge:** pq=partial risk=high steps=3+ rec=hard path_dep=partial
- **judge reason:** The final event can be questioned, but the context of earlier events adds depth that is not strictly necessary for understanding the comment.
- **strict_reason:** hard_path_dep=partial
- **relaxed_reason:** keep_path_dep_partial

## Hard Rejected (10 samples)

### Hard Rejected #1

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **title:** Battle of Ciołków
- **difficulty:** Hard
- **event path:** uprising/Change_of_leadership -> refused/Agree_or_refuse_to_act -> ordered/Arranging -> lost/Earnings_and_losses
- **relation sequence:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `lost his eye`
- **answer_sentence:** Polish losses were negligible, but the Polish commander was wounded and lost his eye.
- **supporting_sentences:**
  - [S2] 100 men under Aleksander Rogaliński and a company of the Russian Murom Regiment under Col. Kozlaninov, the skirmish resulted in Polish victory.
  - [S3] As the engagement started on the very first day of the uprising, the Russian force still obeyed the orders of marching through the occupied country wi
  - [S4] When the Russians approached a local manor in which the Poles had their quarters, the Russian commander ordered a loose formation and tried to negotia
  - [S5] However, Rogaliński refused to negotiate and ordered a charge of the Russians.
  - [S6] After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, war scythes and improvised weap
- **LLM judge:** pq=no risk=high steps=1 rec=easy path_dep=no
- **judge reason:** The final event can be answered directly from the answer sentence without needing context from earlier events.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no

### Hard Rejected #2

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **title:** Who's That Girl World Tour
- **difficulty:** Hard
- **event path:** changes/Change -> marked/Recording -> commented/Statement -> wearing/Wearing
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `wearing a conical`
- **answer_sentence:** A statue of Madonna, wearing a conical bra, was erected in her name at the center of the town of Pacentro in Italy, where her ancestors used to live.
- **supporting_sentences:**
  - [S8] The title Who's That Girl came to Madonna's mind when during rehearsals one-day when she looked at a gigantic image of herself, projected on a screen 
  - [S9] The show consisted of seven costume changes, with song-and-dance routines, theatrics and addressing social causes.
  - [S10] The tour was critically appreciated, with reviewers commending the extravagant nature of the concert and Madonna as a performer.
  - [S13] Who's That Girl was broadcast in a number of international television channels and was released in VHS titled "".
  - [S14] Biographer J. Randy Taraborrelli commented that "Many female artists behave like a diva for a period when they reach superstar status, and the 'Who's 
- **LLM judge:** pq=no risk=high steps=1 rec=easy path_dep=no
- **judge reason:** The final event can be answered directly from the answer sentence without needing context from earlier events.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no

### Hard Rejected #3

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **title:** 2006 state of emergency in the Philippines
- **difficulty:** Hard
- **event path:** announced/Expressing_publicly -> revocation/Process_end -> suspended/Change_event_time -> allowed/Preventing_or_letting
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `the government was allowed at the moment to detain anyone indefinitely without the privilege of the writ of habeas corpus`
- **answer_sentence:** Under the provisions of the 1987 Constitution, the government was allowed at the moment to detain anyone indefinitely without the privilege of the wri
- **supporting_sentences:**
  - [S0] The Philippines was under a state of emergency, announced by presidential spokesperson Ignacio Bunye on the morning of February 24, 2006, by the virtu
  - [S1] 1017.
  - [S5] 1021.
  - [S6] The state of national emergency also led to a temporary suspension of lower-level education classes and an immediate revocation on all licenses and pe
  - [S7] The government, informally known as "Malacañang", after the presidential palace, also suspended all public activities on the same day and even on succ
- **LLM judge:** pq=no risk=high steps=1 rec=easy path_dep=no
- **judge reason:** The final event's answer can be directly obtained from the answer sentence without needing context from earlier events.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no

### Hard Rejected #4

- **doc_id:** `28a13a10cb57f8245b1f98270bad9860`
- **title:** Territorial era of Minnesota
- **difficulty:** Hard
- **event path:** replacing/Change_of_leadership -> become/Becoming -> changed/Exchange -> influence/Influence
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `mixed-race populations continued to influence the territory's culture and`
- **answer_sentence:** The native and mixed-race populations continued to influence the territory's culture and politics, even at the end of the territorial era, though by t
- **supporting_sentences:**
  - [S1] The Minnesota Territory itself was formed only in 1849 but the area had a rich history well before this.
  - [S2] Though there was a long history of European presence in the area before 19th century, it was during the 19th century that the United States began to e
  - [S3] Many of the facets of Minnesota culture that are perceived as the area's "early" history in fact originated after this period.
  - [S9] This trade gradually declined during the early 19th century as demand for furs in Europe diminished.
  - [S10] The lumber industry grew rapidly, replacing furs as the key economic resource.
- **LLM judge:** pq=no risk=high steps=1 rec=easy path_dep=no
- **judge reason:** The final event's answer can be directly obtained from the answer sentence without needing context from earlier events.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no

### Hard Rejected #5

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **title:** Death of Joy Gardner
- **difficulty:** Hard
- **event path:** deportation/Removing -> acquitted/Judgment_communication -> orders/Request -> inquiry/Criminal_investigation
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `public inquiry into the circumstances of Gardner's death has been held`
- **answer_sentence:** Despite continuing pressure by campaigners, no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held.
- **supporting_sentences:**
  - [S0] Joy Angelia Gardner (née Burke, 29 May 1953 – 1 August 1993) was a 40-year-old Jamaican mature student living as an illegal alien in London, England.
  - [S1] She died following an attempt by police officers to detain her for deportation.
  - [S2] During a police raid on her home in Crouch End, London, she was restrained with handcuffs and leather straps and gagged with a 13-foot length of adhes
  - [S4] She was placed on life support but died following a cardiac arrest four days later.
  - [S5] In 1995, three of the police officers involved stood trial for Gardner's manslaughter, but were acquitted.
- **LLM judge:** pq=no risk=high steps=1 rec=easy path_dep=no
- **judge reason:** The final event's answer can be directly found in the answer sentence, making the path unsuitable for generating a question that requires understanding earlier events.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no

### Hard Rejected #6

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **title:** Cyclone Forrest
- **difficulty:** Hard
- **event path:** repeat/Change -> turned/Becoming -> damage/Damaging -> damaged/Damaging
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
- **gold_answer_phrase:** `Martin's Island were damaged`
- **answer_sentence:** Only two deaths were recorded and overall damage was light, though half of all homes on St. Martin's Island were damaged.
- **supporting_sentences:**
  - [S9] Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million).
  - [S10] On November 20, as Forrest reached its peak intensity, fears arose across Bangladesh that a repeat of the catastrophic April 1991 cyclone would take p
  - [S11] As a result, mass evacuation plans were enacted across coastal areas of the country, with plans to relocate up to 2 million people.
  - [S12] But the storm abruptly turned eastward, and the successful evacuation of 600,000 residents spared countless lives.
  - [S13] Only two deaths were recorded and overall damage was light, though half of all homes on St. Martin's Island were damaged.
- **LLM judge:** pq=no risk=high steps=1 rec=easy path_dep=no
- **judge reason:** The final event can be answered directly from the answer sentence without needing context from earlier events.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no

### Hard Rejected #7

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **title:** Death of Joy Gardner
- **difficulty:** Hard
- **event path:** deportation/Removing -> orders/Request -> inquest/Criminal_investigation -> held/Hold
- **relation sequence:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
- **gold_answer_phrase:** `public inquiry into the circumstances of Gardner's death has been held`
- **answer_sentence:** Despite continuing pressure by campaigners, no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held.
- **supporting_sentences:**
  - [S0] Joy Angelia Gardner (née Burke, 29 May 1953 – 1 August 1993) was a 40-year-old Jamaican mature student living as an illegal alien in London, England.
  - [S1] She died following an attempt by police officers to detain her for deportation.
  - [S2] During a police raid on her home in Crouch End, London, she was restrained with handcuffs and leather straps and gagged with a 13-foot length of adhes
  - [S5] In 1995, three of the police officers involved stood trial for Gardner's manslaughter, but were acquitted.
  - [S6] The case became a cause célèbre for civil rights and justice campaigners, and for the first time brought wide public attention to what the "Modern Law
- **LLM judge:** pq=no risk=high steps=3+ rec=hard path_dep=no
- **judge reason:** The final event cannot be clearly questioned based on the preceding events, and the answer is directly contradicted by the supporting context.
- **strict_reason:** path_questionable=no
- **relaxed_reason:** path_questionable=no
