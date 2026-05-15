# Full-Chain Debug Trace

**Generated:** 2026-05-02T22:38:32.687515
**Total items:** 29

---

## Item 0 [Medium] -- FAIL

**doc_id:** 84ce009a07b987d60a79d92bc4d45744
**Raw source:** events=3, relations=2
**Path (2 hops):** "occurred" -> "revocation" -> "lifted"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the President lifted the state of emergency" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What action did the government take on March 3, 2006, one week after proclaiming the state of emergency, and what immediate effects did this have on public activities and demonstrations?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 1 [Hard] -- PASS

**doc_id:** a24058769038462f489b0091ebb24597
**Raw source:** events=4, relations=3
**Path (3 hops):** "uprising" -> "approached" -> "refused" -> "ordered"
**Relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "ordered a charge of the Russians" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "During the uprising, when the Russians approached the Polish quarters and the commander tried to negotiate, what did Rogaliński decide to do?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Rogaliński refused to negotiate. correct=1.0 confidence=0.0

---

## Item 2 [Medium] -- FAIL

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=3, relations=2
**Path (2 hops):** "capture" -> "casualties" -> "rallied"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What event brought the Sultanate of Johor and the Dutch together in a strategic alliance? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 3 [Easy] -- FAIL

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=2, relations=1
**Path (1 hops):** "took place" -> "released"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "released at the scene" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What happened to the victims after they were treated?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 4 [Medium] -- PASS

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
**Solver:** status=ok result=Lost an eye. correct=1.0 confidence=0.0

---

## Item 5 [Medium] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=3, relations=2
**Path (2 hops):** "began" -> "expanded" -> "bombed"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "NATO aircraft first bombed ground targets in an operation near Goražde" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission's scope was expanded?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=NATO aircraft conducted bombing operations on ground targets targets in missions near Goražde kukly April 1/ay correct=1.0 confidence=0.0

---

## Item 6 [Easy] -- PASS

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

## Item 7 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Path (1 hops):** "Disguised" -> "operation"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "this had been cancelled by the time the operation was carried out" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What happened after the Irgun members had disguised themselves?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 8 [Medium] -- FAIL

**doc_id:** 6dabade56742b6040cda6a5838176f6c
**Raw source:** events=3, relations=2
**Path (2 hops):** "trained" -> "addressing" -> "wearing"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "wearing a conical" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What kind of attire did the statue of Madonna, erected in Pacentro, feature, and how did her physical training influence this choice?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=no cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 9 [Easy] -- FAIL

**doc_id:** dd2a791aa826766cf0d05dc8102f5c8e
**Raw source:** events=2, relations=1
**Path (1 hops):** "attention" -> "inquest"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What has not been conducted despite the case becoming a cause célèbre?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 10 [Hard] -- FAIL

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

## Item 11 [Medium] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=3, relations=2
**Path (2 hops):** "expanded" -> "bombed" -> "helped"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "These engagements helped show" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What did the engagements during Operation Deny Flight, such as the air battle over Banja Luka and the bombing of ground targets near Goražde, demonstrate about NATO's capabilities in the post-Cold War era?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=NATO could operate globally. correct=0.0 confidence=0.0

---

## Item 12 [Easy] -- PASS

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=2, relations=1
**Path (1 hops):** "striking" -> "losses"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million)" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What were the estimated agricultural losses in Thailand due to the storm?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=฿800 million correct=0.0 confidence=0.0

---

## Item 13 [Hard] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=4, relations=3
**Path (3 hops):** "warnings" -> "called" -> "carried out" -> "killed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "91 people of various nationalities were killed" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "What was the outcome after warnings were sent and calls were made but no evacuation was ordered?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: brief
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 14 [Medium] -- PASS

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

## Item 15 [Medium] -- PASS

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
**Solver:** status=ok result=Sinjar correct=0.0 confidence=0.0

---

## Item 16 [Easy] -- PASS

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
**Solver:** status=ok result=Significant storm surge damaged or destroyed on on on homes and caused two. correct=0.0 confidence=0.0

---

## Item 17 [Medium] -- PASS

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

## Item 18 [Hard] -- PASS

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=4, relations=3
**Path (3 hops):** "crowded" -> "took place" -> "investigated" -> "closed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
**Answer phrase:** "closed in January 2003 without any indictments as Gerdt was the sole suspect" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the shopping center was especially crowded with many children present, and the bombing took place, what was the outcome of the investigation?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=No indictments made. correct=0.0 confidence=0.0

---

## Item 19 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Path (1 hops):** "ordered" -> "minimize"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "he actively sought to minimize the atrocities" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 20 [Hard] -- PASS

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

## Item 21 [Hard] -- PASS

**doc_id:** db50381e7d1dd4a41fb4ac60eaebe3a4
**Raw source:** events=4, relations=3
**Path (3 hops):** "Moving" -> "surrounded" -> "pushed" -> "battle"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the French marshal offered battle" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the Allied corps surrounded Bayonne and then pushed Soult's army back, what did the French marshal do?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The French abandoned thez of of of Bordeaux and fell back fall fall back back toward Toulouse..............z.. correct=0.0 confidence=0.0

---

## Item 22 [Hard] -- PASS

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=4, relations=3
**Path (3 hops):** "began" -> "launching" -> "took" -> "agreed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
**Answer phrase:** "agreed not to seek territories or wage war with the Malay kingdoms" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the Dutch began their campaign to destroy Portuguese power and launched small incursions against the Portuguese, what did they agree to do with the Malay kingdoms?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The Portuguese agreed this by by agreeingis tois by correct=0.0 confidence=0.0

---

## Item 23 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Path (1 hops):** "planned" -> "planted"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "members of the Irgun planted a bomb in the basement of the main building of the" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did members of the Irgun do in the basement of the hotel's main building, as described in the context provided? ?" (status=ok, attempts=3)
**Grammar:** FAIL
**Filter reason:** grammar=repeat_question_mark; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 24 [Hard] -- PASS

**doc_id:** 9fcf7e509cc4e59026333ba469e22ec3
**Raw source:** events=4, relations=3
**Path (3 hops):** "began" -> "providing" -> "helped" -> "adapted"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/CONTAINS
**Answer phrase:** "NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engagement on the plains of Central Europe" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "What did Operation Deny Flight help show about NATO's ability to operate in the post-Cold War era after providing close air support and engaging in combat over Banja Luka and Goražde?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=NATO could engage in diverse operations. correct=1.0 confidence=0.0

---

## Item 25 [Easy] -- FAIL

**doc_id:** 28a13a10cb57f8245b1f98270bad9860
**Raw source:** events=2, relations=1
**Path (1 hops):** "diminished" -> "establish"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the United States began to establish a firm presence in what would become Minnesota" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What process replaced the diminishing trade in furs as the primary economic activity in Minnesota during the early 19th century?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 26 [Hard] -- FAIL

**doc_id:** 81c576926e0c52f158b210c244028f0b
**Raw source:** events=4, relations=3
**Path (3 hops):** "stopped" -> "destroyed" -> "rushed" -> "forbade"
**Relations:** TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "It forbade Russia from basing warships in the Black Sea" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the Ottomans stopped the Russian advance at Silistra and their fleet destroyed a Russian attempt to reinforce Kars, what did France and Britain do to prevent further Russian aggression?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=no cons=no
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
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 28 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Path (1 hops):** "permitted" -> "restrain"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "he was powerless to restrain the Seneca" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did Butler claim he was unable to do during the campaigns of 1778 regarding the Seneca?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---
