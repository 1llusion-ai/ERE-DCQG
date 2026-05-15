# Full-Chain Debug Trace

**Generated:** 2026-05-02T23:08:48.693893
**Total items:** 76

---

## Item 0 [Easy] -- FAIL

**doc_id:** 06f91ced00b41867979f3d5dc6996da2
**Raw source:** events=2, relations=1
**Path (1 hops):** "attack" -> "blamed"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "blamed Yamamoto for the attack on Pearl Harbor" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What was the intended outcome of the U.S. leaders' actions during the mission to kill Yamamoto?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 1 [Easy] -- PASS

**doc_id:** 84ce009a07b987d60a79d92bc4d45744
**Raw source:** events=2, relations=1
**Path (1 hops):** "announced" -> "detain"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the government was allowed at the moment to detain anyone indefinitely without the privilege of the writ of habeas corpus" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What action was the government permitted to take immediately after the state of emergency was announced?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The privilege of the writ of habeas corpus wasong- suspendedmediately correct=0.0 confidence=0.0

---

## Item 2 [Easy] -- FAIL

**doc_id:** a24058769038462f489b0091ebb24597
**Raw source:** events=2, relations=1
**Path (1 hops):** "ordered" -> "wounded"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the Polish commander was wounded" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What after the Russian commander refused to negotiate and ordered a charge of the Russians, , what the outcome of the engagement was ? " (status=ok, attempts=3)
**Grammar:** FAIL
**Filter reason:** grammar=broken_grammar: What after; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 3 [Medium] -- FAIL

**doc_id:** 81c576926e0c52f158b210c244028f0b
**Raw source:** events=3, relations=2
**Path (2 hops):** "prepared" -> "granted" -> "recognize"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "to recognize the need for fundamental transformations aimed at modernizing" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What significant realization did Russia's educated elites come to after the Crimean War, prompting a push for fundamental transformations? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 4 [Hard] -- FAIL

**doc_id:** 81c576926e0c52f158b210c244028f0b
**Raw source:** events=4, relations=3
**Path (3 hops):** "stopped" -> "destroyed" -> "rushed" -> "forbade"
**Relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "It forbade Russia from basing warships in the Black Sea" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the Ottomans stopped the Russian advance at Silistra and their fleet destroyed a Russian reinforcement attempt at Sinop, what did France and Britain do to prevent further Russian aggression?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=no cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 5 [Easy] -- FAIL

**doc_id:** db50381e7d1dd4a41fb4ac60eaebe3a4
**Raw source:** events=2, relations=1
**Path (1 hops):** "Moving" -> "attack"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "The Battle of Orthez saw the Anglo-Portuguese Army under Field Marshal Arthur Wellesley, Marquess of Wellington attack an Imperial French army led by Marshal Nicolas Soult in southern France" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What action did the Anglo-Portuguese Army take against the Imperial French army during the Battle of Orthez in February 1814?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 6 [Easy] -- FAIL

**doc_id:** 28a13a10cb57f8245b1f98270bad9860
**Raw source:** events=2, relations=1
**Path (1 hops):** "diminished" -> "establish"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the United States began to establish a firm presence in what would become Minnesota" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What economic resource replaced furs as the key economic activity in the area during the early 19th century?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: brief; path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 7 [Easy] -- FAIL

**doc_id:** 6dabade56742b6040cda6a5838176f6c
**Raw source:** events=2, relations=1
**Path (1 hops):** "supported" -> "commending"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "with reviewers commending the extravagant nature of the concert and Madonna as a performer" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did the reviewers think about the extravagant nature of the concert and Madonna's performance during the Who's That Girl World Tour?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 8 [Easy] -- PASS

**doc_id:** 37d153abeafe0477ce6a6398e26eb442
**Raw source:** events=2, relations=1
**Path (1 hops):** "defense" -> "reveal"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "commander Xie Jinyuan not wanting to reveal their true strength to the" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What did commander Xie Jinyuan do to the Japanese forces during the defense of Sihang Warehouse?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Provided exaggerated numbers. correct=0.0 confidence=0.0

---

## Item 9 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Path (1 hops):** "Disguised" -> "operation"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "this had been cancelled by the time the operation was carried out" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What was the name of the event where disguised Irgun members planted a bomb in the basement of the hotel's main building, causing the collapse of the western half of the southern wing? ?" (status=ok, attempts=3)
**Grammar:** FAIL
**Filter reason:** grammar=repeat_question_mark; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 10 [Hard] -- PASS

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=4, relations=3
**Path (3 hops):** "warnings" -> "called" -> "carried out" -> "killed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "91 people of various nationalities were killed" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "When did the warnings lead the I on the Irgun on carried the bombing and did the hotel staff on did evacuate?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=No, they did not. correct=0.0 confidence=0.0

---

## Item 11 [Hard] -- PASS

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=4, relations=3
**Path (3 hops):** "began" -> "launching" -> "took" -> "agreed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
**Answer phrase:** "agreed not to seek territories or wage war with the Malay kingdoms" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the Dutch began their campaign to destroy Portuguese power and launched small incursions against the Portuguese, what did they agree to do in line with their agreement with Johor in 1606?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Not seek territories. correct=0.0 confidence=0.0

---

## Item 12 [Medium] -- PASS

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=3, relations=2
**Path (2 hops):** "Originating" -> "Tracking" -> "storm"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "also referred to as Tropical Storm" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What was the name of the tropical cyclone that prompted the evacuation of 600,000 people in Bangladesh in late November 1992, and how did it develop over time?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Cyclone Forrest, developed from disturbed weather. correct=0.0 confidence=0.0

---

## Item 13 [Medium] -- PASS

**doc_id:** 37d153abeafe0477ce6a6398e26eb442
**Raw source:** events=3, relations=2
**Path (2 hops):** "defense" -> "invasion" -> "aftermath"
**Relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
**Answer phrase:** "people in the demoralizing aftermath of the Japanese invasion of Shanghai" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What provided a morale-lifting consolation to the Chinese people during the difficult period following the Japanese invasion of Shanghai?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The defense of Sihang Warehouse. correct=0.0 confidence=0.0

---

## Item 14 [Medium] -- FAIL

**doc_id:** 6dabade56742b6040cda6a5838176f6c
**Raw source:** events=3, relations=2
**Path (2 hops):** "trained" -> "addressing" -> "wearing"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "wearing a conical" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What kind of attire did the statue of Madonna in Pacentro feature, and how did her physical training contribute to this choice?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=yes ans=no cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 15 [Medium] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=3, relations=2
**Path (2 hops):** "expanded" -> "bombed" -> "helped"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "These engagements helped show" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What did these operations demonstrate about NATO's capabilities after expanding the mission and carrying out bombing raids?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=NATO aircraft bombed ground correct=0.0 confidence=0.0

---

## Item 16 [Easy] -- PASS

**doc_id:** 84ce009a07b987d60a79d92bc4d45744
**Raw source:** events=2, relations=1
**Path (1 hops):** "declared" -> "foiled"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "it foiled an alleged coup d'état attempt against the rule of President Gloria Macapagal-Arroyo earlier" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did the government claim to have done on the same day it declared a state of emergency?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The government claimed to have foiled a coup d'état attempt that day and to have on a state manh on (state of emergency due that " clear " " clear " to one a " correct=0.0 confidence=0.0

---

## Item 17 [Medium] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=3, relations=2
**Path (2 hops):** "restrain" -> "took place" -> "descended on"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "Mohawks descended on Cherry" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What actions did the Loyalists, British soldiers, Seneca, and Mohawks take that led to the unpreparedness of Cherry Valley's defenders during the attack?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 18 [Medium] -- PASS

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=3, relations=2
**Path (2 hops):** "capture" -> "combined" -> "removing"
**Relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
**Answer phrase:** "removing their influence in the Malay archipelago" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What was the outcome of the combined Dutch-Johor effort that followed the successful capture of Malacca from the Portuguese?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Dutchutch destroyed Portuguese power influence. correct=0.0 confidence=0.0

---

## Item 19 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Path (1 hops):** "permitted" -> "restrain"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "he was powerless to restrain the Seneca" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What was Butler claimed to be unable to do during the campaigns of 1778 regarding the Seneca?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 20 [Medium] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=3, relations=2
**Path (2 hops):** "began" -> "combat" -> "end"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "by its end on 20 December" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What was the final outcome of Operation Deny Flight, and how many sorties were flown by NATO pilots during the operation?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=14,40 correct=0.0 confidence=0.0

---

## Item 21 [Easy] -- FAIL

**doc_id:** f28bce270df5a122c09365002d247e76
**Raw source:** events=2, relations=1
**Path (1 hops):** "began" -> "assumed"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Nicaragua assumed a quasi-protectorate status under the 1916 Bryan–Chamorro Treaty" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 22 [Medium] -- FAIL

**doc_id:** 275eb0bc9caacb30e4f58d85469458d1
**Raw source:** events=3, relations=2
**Path (2 hops):** "delivered" -> "take" -> "recover"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/CONTAINS
**Answer phrase:** "had been run down by a car earlier in the night (which was an angle for Austin to take time to recover from his injuries)" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "Why was Stone Cold Steve Austin replaced by Big Show, and what was the wrestling company's angle for this change?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 23 [Easy] -- PASS

**doc_id:** 04a82d4eac379a98efcd87ebdba0b0ce
**Raw source:** events=2, relations=1
**Path (1 hops):** "died" -> "funeral"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "a state funeral at the Palace of the Argentine National Congress" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What significant event was organized by Vice President Julio Cobos after Raúl Alfonsín's death?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=onosons the Palace of the Argentine Congress. correct=0.0 confidence=0.0

---

## Item 24 [Easy] -- PASS

**doc_id:** 06f91ced00b41867979f3d5dc6996da2
**Raw source:** events=2, relations=1
**Path (1 hops):** "Vengeance" -> "shot down"
**Relations:** SUBEVENT
**Answer phrase:** "his transport bomber aircraft was shot down by United States Army Air Forces fighter aircraft operating from Kukum Field on Guadalcanal" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "How did Admiral Isoroku Yamamoto die during Operation Vengeance in 1943?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Yes on correct=0.0 confidence=0.0

---

## Item 25 [Medium] -- FAIL

**doc_id:** db50381e7d1dd4a41fb4ac60eaebe3a4
**Raw source:** events=3, relations=2
**Path (2 hops):** "offered" -> "conducted" -> "fall"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "fall back east toward Toulouse" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What decision did Soult make after abandoning Bordeaux in the context of the Peninsular War?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 26 [Medium] -- FAIL

**doc_id:** f83fc49c020ec542b16d463b3f7c2c14
**Raw source:** events=3, relations=2
**Path (2 hops):** "started" -> "enabled" -> "evacuated"
**Relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION
**Answer phrase:** "fled into the Sinjar Mountains to be evacuated" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What assistance allowed the majority of the 50,000 Yazidis who fled into the Sinjar Mountains to be helped out? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 27 [Medium] -- FAIL

**doc_id:** 81c576926e0c52f158b210c244028f0b
**Raw source:** events=3, relations=2
**Path (2 hops):** "changes" -> "quipped" -> "proved"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "The Crimean War proved to be the moment of truth for Nikolaevan Russia" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What significant event did the Crimean War represent for Nikolaevan Russia, considering Karl Marx's comment and the subsequent demands for action from the public?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 28 [Medium] -- PASS

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=3, relations=2
**Path (2 hops):** "enacted" -> "turned" -> "evacuation"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "prompted the evacuation of 600,000 people in Bangladesh in late November 1992" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What event was successfully carried out, leading to the relocation of 600,000 people and preventing a potential disaster during Cyclone Forrest's approach to Bangladesh in 1992?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Mass evacuation plans correct=1.0 confidence=0.0

---

## Item 29 [Easy] -- PASS

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=2, relations=1
**Path (1 hops):** "reached" -> "produced"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the system produced significant storm" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did the system do in Thailand after reaching its peak intensity?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Produced significant storm surge. correct=1.0 confidence=0.0

---

## Item 30 [Easy] -- PASS

**doc_id:** dd2a791aa826766cf0d05dc8102f5c8e
**Raw source:** events=2, relations=1
**Path (1 hops):** "raid" -> "wrapped"
**Relations:** TEMPORAL/CONTAINS
**Answer phrase:** "gagged with a 13-foot length of adhesive "Elastoplast" tape wrapped around her head" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What method did police use to restrain her during the raid in Crouch End, London?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=handcuffs and leather straps on correct=0.0 confidence=0.0

---

## Item 31 [Medium] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=3, relations=2
**Path (2 hops):** "spanned" -> "providing" -> "taken"
**Relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE
**Answer phrase:** "UN peacekeepers were taken as hostages in response to NATO bombing" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What significant event occurred during the span of Deny Flight that led to increased tension between UN and NATO forces?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=UN peacekeepers taken as hostages. correct=1.0 confidence=0.0

---

## Item 32 [Easy] -- FAIL

**doc_id:** dd2a791aa826766cf0d05dc8102f5c8e
**Raw source:** events=2, relations=1
**Path (1 hops):** "placed" -> "involved"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "three of the police officers involved stood trial for Gardner's" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "In 1995, which action did the three police officers take regarding the case? ?" (status=ok, attempts=2)
**Grammar:** FAIL
**Filter reason:** grammar=base: bad start: in; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 33 [Medium] -- FAIL

**doc_id:** 6dabade56742b6040cda6a5838176f6c
**Raw source:** events=3, relations=2
**Path (2 hops):** "looked" -> "playing" -> "presenting"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/CONTAINS
**Answer phrase:** "presenting her older songs for the show" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What did Madonna decide to do with her older songs during the rehearsals according to Patrick Leonard's encouragement? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 34 [Easy] -- PASS

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=2, relations=1
**Path (1 hops):** "crowded" -> "exploded"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "A bomb carried by Petri Erkki-Tapio Gerdt exploded at 19:36 killing five" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What immediately followed the detonation of the bomb at the crowded Myyrmanni shopping mall?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Five died immediately. correct=0.0 confidence=0.0

---

## Item 35 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Path (1 hops):** "planned" -> "planted"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "members of the Irgun planted a bomb in the basement of the main building of the" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did the members of the Irgun do after they disguised themselves as Arab workers and hotel waiters?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 36 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Path (1 hops):** "Disguised" -> "carried out"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "a bomb search had already been carried out, it appears" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 37 [Hard] -- FAIL

**doc_id:** 81c576926e0c52f158b210c244028f0b
**Raw source:** events=4, relations=3
**Path (3 hops):** "stopped" -> "destroyed" -> "arriving" -> "signed on"
**Relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "signed on 30 March" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the Russians were stopped at Silistra and their fleet destroyed at Sinop, when did France and Britain act to welcome the development of the war ending?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=yes ans=no cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 38 [Easy] -- PASS

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=2, relations=1
**Path (1 hops):** "classified" -> "damaged"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Martin's Island were damaged" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What happened to St. Martin's Island after Cyclone Forrest turned eastward?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Cyclone Forrest turned St. Martin's Island on turned an eastward turn. correct=0.0 confidence=0.0

---

## Item 39 [Medium] -- PASS

**doc_id:** db50381e7d1dd4a41fb4ac60eaebe3a4
**Raw source:** events=3, relations=2
**Path (2 hops):** "isolated" -> "overcome" -> "decided"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What decision did Marshal Soult make after his army was isolated and forced to retreat during the subsequent operations following the Battle of Orthez?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Abandon Bordeaux, retreat east. correct=1.0 confidence=0.0

---

## Item 40 [Hard] -- PASS

**doc_id:** a24058769038462f489b0091ebb24597
**Raw source:** events=4, relations=3
**Path (3 hops):** "uprising" -> "negotiate" -> "refused" -> "killed"
**Relations:** TEMPORAL/CONTAINS, CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "After a short hand-to-hand fight the Russian commander was killed" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "When the uprising began and the Polish refused to negotiate, the Russian commander did what?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=was killed correct=0.0 confidence=0.0

---

## Item 41 [Medium] -- PASS

**doc_id:** a24058769038462f489b0091ebb24597
**Raw source:** events=3, relations=2
**Path (2 hops):** "refused" -> "fight" -> "lost"
**Relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
**Answer phrase:** "lost his eye" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What happened to the Polish commander after he refused to negotiate with the Russians?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The Polish commander was and pérdida wasonestly suffered as wasansible
 pérdida one de dos dos dos dos dos dos dos pérdida one eye
icílos dos dos dos dos dos dos dos dos dos dos correct=0.0 confidence=0.0

---

## Item 42 [Hard] -- PASS

**doc_id:** a24058769038462f489b0091ebb24597
**Raw source:** events=4, relations=3
**Path (3 hops):** "uprising" -> "approached" -> "refused" -> "ordered"
**Relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "ordered a charge of the Russians" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "During the uprising, when the Russians approached the Polish quarters and the commander tried to negotiate, what did Rogaliński decide to do next?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Rogaliński refused to negotiate. correct=1.0 confidence=0.0

---

## Item 43 [Medium] -- PASS

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=3, relations=2
**Path (2 hops):** "capture" -> "began" -> "destroyed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What significant action did the Dutch and their allies take that marked the end of Portuguese power in the Malay archipelago after they began their campaign?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Captured Malacca in January 1641. correct=0.0 confidence=0.0

---

## Item 44 [Medium] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=3, relations=2
**Path (2 hops):** "began" -> "expanded" -> "bombed"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "NATO aircraft first bombed ground targets in an operation near Goražde" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission scope was expanded?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=NATO aircraft provided air-ground support and conducted coercive strikes on targets kukos in Bosnia. kukos the mission scope was expanded kukos kukos kuk kukos kuk kukos kuk kukos kuk correct=0.0 confidence=0.0

---

## Item 45 [Easy] -- FAIL

**doc_id:** f83fc49c020ec542b16d463b3f7c2c14
**Raw source:** events=2, relations=1
**Path (1 hops):** "started" -> "killing"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "The Sinjar massacre was the genocidal killing" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did ISIL's attack on Sinjar and neighboring towns on August 3, 2014, lead to?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 46 [Easy] -- FAIL

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=2, relations=1
**Path (1 hops):** "striking" -> "losses"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million)" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What were the estimated agricultural damages in the hardest hit areas of Thailand due to Forrest?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 47 [Easy] -- PASS

**doc_id:** 37d153abeafe0477ce6a6398e26eb442
**Raw source:** events=2, relations=1
**Path (1 hops):** "defense" -> "Battle"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "marked the beginning of the end of the three-month Battle of Shanghai in the opening phase of the Second Sino-Japanese War" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What was the broader conflict during which the defense of Sihang Warehouse took place?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Second Sino-Japanese War correct=0.0 confidence=0.0

---

## Item 48 [Easy] -- FAIL

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=2, relations=1
**Path (1 hops):** "investigated" -> "determined"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "His motive was not determined" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What was not established regarding Gerdt's motive for the incident?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 49 [Medium] -- FAIL

**doc_id:** 81c576926e0c52f158b210c244028f0b
**Raw source:** events=3, relations=2
**Path (2 hops):** "prepared" -> "forbade" -> "restoring"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "restoring Russia's position in the ranks of European powers" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 50 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Path (1 hops):** "ordered" -> "minimize"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "he actively sought to minimize the atrocities" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did Brant do regarding the atrocities at Cherry Valley?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 51 [Medium] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=3, relations=2
**Path (2 hops):** "occurred" -> "carried out" -> "arisen"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "Controversy has arisen over the timing" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What controversy arose regarding the timing and adequacy of the warnings, and the reasons why the hotel was evacuated evacuated before before the explosion?" (status=ok, attempts=3)
**Grammar:** FAIL
**Filter reason:** grammar=base: word repetition: evacuated; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 52 [Easy] -- FAIL

**doc_id:** dd2a791aa826766cf0d05dc8102f5c8e
**Raw source:** events=2, relations=1
**Path (1 hops):** "attention" -> "inquest"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 53 [Medium] -- PASS

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=3, relations=2
**Path (2 hops):** "come to" -> "exploded" -> "injured"
**Relations:** TEMPORAL/BEFORE, CAUSE/CAUSE
**Answer phrase:** "166 people were injured, including 10 children" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What happened to the people who were at the shopping center when the bomb exploded, and how many of them were children?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=7 died, two were children. correct=0.0 confidence=0.0

---

## Item 54 [Medium] -- FAIL

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=3, relations=2
**Path (2 hops):** "launching" -> "assaulted" -> "control"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the Dutch took control of Malacca and agreed not to seek territories or wage war with the Malay kingdoms" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What was the ultimate result of the combined Dutch and Johor forces' efforts against the Portuguese in Malacca in 1641?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 55 [Medium] -- PASS

**doc_id:** 6fc6538bd2c19d34942f7f36274d83ae
**Raw source:** events=3, relations=2
**Path (2 hops):** "persisted" -> "aid" -> "relief"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "$150 million program to provide relief to damaged areas" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What kind of assistance did the Commonwealth of Australia provide to help with the aftermath of Winifred's damage?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Three-year, $150 million relief program. correct=0.0 confidence=0.0

---

## Item 56 [Medium] -- FAIL

**doc_id:** 84ce009a07b987d60a79d92bc4d45744
**Raw source:** events=3, relations=2
**Path (2 hops):** "claimed" -> "occurred" -> "revocation"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "an immediate revocation on all licenses and permits to hold demonstrations and protests" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 57 [Medium] -- PASS

**doc_id:** 06f91ced00b41867979f3d5dc6996da2
**Raw source:** events=3, relations=2
**Path (2 hops):** "killed" -> "revenge" -> "claimed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "pilots claimed to have shot down three twin-engined bombers and two fighters during the" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What did the U.S. pilots claim to have achieved during their mission to kill Yamamoto, and how does this claim compare to Japanese sources?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=U.S. pilots claim to shoot down two bombers and two fighters., Japanese sources only confirm two bombers two shot down. correct=0.0 confidence=0.0

---

## Item 58 [Easy] -- FAIL

**doc_id:** a24058769038462f489b0091ebb24597
**Raw source:** events=2, relations=1
**Path (1 hops):** "killed" -> "dispersed"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "After a short hand-to-hand fight the Russian commander was killed and his unit dispersed" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What happened to the Russian unit after the hand-to-hand fight?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 59 [Medium] -- FAIL

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=3, relations=2
**Path (2 hops):** "capture" -> "casualties" -> "rallied"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 60 [Medium] -- PASS

**doc_id:** 28a13a10cb57f8245b1f98270bad9860
**Raw source:** events=3, relations=2
**Path (2 hops):** "transition" -> "replacing" -> "formed"
**Relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE
**Answer phrase:** "The Minnesota Territory itself was formed only in 1849" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What significant change occurred in the economic resources of the area before the formation of the Minnesota Territory in 1849?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Furs declined, lumber rose. correct=0.0 confidence=0.0

---

## Item 61 [Medium] -- PASS

**doc_id:** f83fc49c020ec542b16d463b3f7c2c14
**Raw source:** events=3, relations=2
**Path (2 hops):** "reacted" -> "airstrikes" -> "started"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "YPG forces started the December 2014 Sinjar offensive with the support of US airstrikes" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What offensive did the Kurdish forces begin after the United States launched airstrikes against ISIL in northern Iraq?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=December 2014 Sinjar offensive correct=0.0 confidence=0.0

---

## Item 62 [Medium] -- FAIL

**doc_id:** f83fc49c020ec542b16d463b3f7c2c14
**Raw source:** events=3, relations=2
**Path (2 hops):** "started" -> "reacted" -> "broke"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "This offensive broke ISIL's troop transport routes and supply lines between Mosul and" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 63 [Hard] -- FAIL

**doc_id:** db50381e7d1dd4a41fb4ac60eaebe3a4
**Raw source:** events=4, relations=3
**Path (3 hops):** "Moving" -> "surrounded" -> "pushed" -> "battle"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the French marshal offered battle" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the western-most Allied corps surrounded and isolated Bayonne, what event occurred as the remaining two Allied corps continued their eastward drive?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 1 prior events, need >= 2 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 64 [Medium] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=3, relations=2
**Path (2 hops):** "descended on" -> "minimize" -> "drove"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What action ultimately led to the Iroquois being forced out of western New York in 1779?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 65 [Medium] -- FAIL

**doc_id:** 37d153abeafe0477ce6a6398e26eb442
**Raw source:** events=3, relations=2
**Path (2 hops):** "shot" -> "dare" -> "keep"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "whom the Japanese wanted to keep out of the war" (status=complete)
**Prefilter:** PASS -- pass [risk: single_sentence_risk_high]
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "Why the Japanese avoid using naval artillery strikes and mustard gas in the area where the battle took took take place? " (status=ok, attempts=2)
**Grammar:** FAIL
**Filter reason:** grammar=base: word repetition: took; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 66 [Hard] -- PASS

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=4, relations=3
**Path (3 hops):** "crowded" -> "took place" -> "investigated" -> "closed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
**Answer phrase:** "closed in January 2003 without any indictments as Gerdt was the sole suspect" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "How was the Myyrmanni shopping mall when the bombing took place, and what was the subsequent investigation like?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Crowded; no indictments. correct=0.0 confidence=0.0

---

## Item 67 [Easy] -- PASS

**doc_id:** db50381e7d1dd4a41fb4ac60eaebe3a4
**Raw source:** events=2, relations=1
**Path (1 hops):** "pause" -> "Battle"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "The next action was the Battle of Toulouse" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What was the next action after the Allied corps paused in the campaign?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Resuming eastwardwardward march correct=0.0 confidence=0.0

---

## Item 68 [Hard] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=4, relations=3
**Path (3 hops):** "began" -> "providing" -> "helped" -> "adapted"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
**Answer phrase:** "NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engagement on the plains of Central Europe" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the operation began and started providing close air support, how did NATO demonstrate its ability in the post-Cold War era?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Operation Den Den Den correct=0.0 confidence=0.0

---

## Item 69 [Easy] -- PASS

**doc_id:** 04a82d4eac379a98efcd87ebdba0b0ce
**Raw source:** events=2, relations=1
**Path (1 hops):** "arranged" -> "mourning"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "arranged three days of national mourning" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did Vice President Julio Cobos arrange after Alfonsín's death?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=three days of national mourning and a state state funeral at correct=0.0 confidence=0.0

---

## Item 70 [Easy] -- FAIL

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=2, relations=1
**Path (1 hops):** "took place" -> "released"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "released at the scene" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 71 [Medium] -- FAIL

**doc_id:** 84ce009a07b987d60a79d92bc4d45744
**Raw source:** events=3, relations=2
**Path (2 hops):** "occurred" -> "revocation" -> "lifted"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the President lifted the state of emergency" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What action did the government take on March 3, 2006, regarding the state of emergency?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 72 [Easy] -- FAIL

**doc_id:** 6dabade56742b6040cda6a5838176f6c
**Raw source:** events=2, relations=1
**Path (1 hops):** "trained" -> "collaborated"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "she collaborated with designer Marlene" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did Madonna do after she trained for the choreography?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: The question does not ask about the target final event, which is 'collaborated.' It asks about Madonna's training for choreography, which is not directly related to the target event.
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 73 [Medium] -- PASS

**doc_id:** 04a82d4eac379a98efcd87ebdba0b0ce
**Raw source:** events=3, relations=2
**Path (2 hops):** "died" -> "took" -> "left"
**Relations:** CAUSE/PRECONDITION, CAUSE/PRECONDITION
**Answer phrase:** "left him at the pantheon for the veterans of the Revolution of the Park" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What final resting place was given to Raúl Alfonsín after his death?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=La Recoleta Cemetery correct=0.0 confidence=0.0

---

## Item 74 [Medium] -- PASS

**doc_id:** 84ce009a07b987d60a79d92bc4d45744
**Raw source:** events=3, relations=2
**Path (2 hops):** "allowed" -> "detain" -> "lifted"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "President Arroyo lifted the state of emergency on March" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What action did President Arroyo take regarding detentions after the state of emergency was allowed on March 3, 2006?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Ar the on correct=0.0 confidence=0.0

---

## Item 75 [Easy] -- PASS

**doc_id:** f28bce270df5a122c09365002d247e76
**Raw source:** events=2, relations=1
**Path (1 hops):** "assumed" -> "opposed"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "President Herbert Hoover (1929–1933) opposed the relationship" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "How Herbert Hoover oppose the United of Nicaragua and the United States as in the context of American military interventions in Nicaragua to prevent a Nicaraguan Canal? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Opposed 1916 treaty. correct=0.0 confidence=0.0

---
