# Full-Chain Debug Trace

**Generated:** 2026-05-02T22:30:52.578294
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
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
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
**Solver:** status=ok result=Ordered a charge. correct=1.0 confidence=0.0

---

## Item 2 [Medium] -- PASS

**doc_id:** 3dcfd60153822a6a8f6a516f161fc506
**Raw source:** events=3, relations=2
**Path (2 hops):** "capture" -> "casualties" -> "rallied"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "the battle rallied the forces of the Sultanate of Johor in an alliance with the Dutch" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What event followed the heavier casualties suffered by the Portuguese fleet and led to the forces of the Sultanate of Johor joining the Dutch in their campaign against the Portuguese?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Battle rallied forces. correct=0.0 confidence=0.0

---

## Item 3 [Easy] -- FAIL

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=2, relations=1
**Path (1 hops):** "took place" -> "released"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "released at the scene" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "How were the 100 victims who did not require hospitalization after the Myyrmanni bombing treated?" (status=ok, attempts=3)
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
**Question:** "What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission scope was expanded?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Bombing ground targets near Goražde. correct=1.0 confidence=0.0

---

## Item 6 [Easy] -- PASS

**doc_id:** 04a82d4eac379a98efcd87ebdba0b0ce
**Raw source:** events=2, relations=1
**Path (1 hops):** "died" -> "funeral"
**Relations:** CAUSE/PRECONDITION
**Answer phrase:** "a state funeral at the Palace of the Argentine National Congress" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What event did Vice president Julio Cobos arrange after Raúl Alfonsín's death in 2009?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Three days of national mourning. correct=0.0 confidence=0.0

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
**Filter reason:** answer_consistency=no: The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves.
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
**Question:** "How did Marlene Stewart contribute to the costumes for the Who's That Girl Tour, and what was the impact of these costumes on the show?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 0 prior events, need >= 1 [FAIL]
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
**Question:** "What did not happen despite the case becoming a cause célèbre?" (status=ok, attempts=3)
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
**Question:** "What did the NATO operation's engagement in bombing ground targets near Goražde in April 1994 help to demonstrate?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=NATO's adaptability post-Cold War. correct=0.0 confidence=0.0

---

## Item 12 [Easy] -- FAIL

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=2, relations=1
**Path (1 hops):** "striking" -> "losses"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Agricultural losses in the hardest hit areas were estimated at ฿800 million (US$32 million)" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** generation_error
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 13 [Hard] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=4, relations=3
**Path (3 hops):** "warnings" -> "called" -> "carried out" -> "killed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "91 people of various nationalities were killed" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "When did the warnings and the hotel manager notification occur before occur before the bombing occurred take take place?" (status=ok, attempts=2)
**Grammar:** FAIL
**Filter reason:** grammar=base: word repetition: take; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
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
**Question:** "What provided a morale-lifting consolation to the Chinese people following the Japanese invasion of Shanghai?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=The successful defense of Sihang Warehouse. correct=0.0 confidence=0.0

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
**Solver:** status=ok result=December 2014 Sinjar offensive correct=0.0 confidence=0.0

---

## Item 16 [Easy] -- PASS

**doc_id:** 8af040b8bfa2eec89d1197392f2fdac5
**Raw source:** events=2, relations=1
**Path (1 hops):** "reached" -> "produced"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the system produced significant storm" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did the cyclone produce in Thailand?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Significant storm surge, damage/loss. correct=1.0 confidence=0.0

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
**Solver:** status=ok result=NATO bombing took two UN UN peacekeepers in hostages. correct=0.0 confidence=0.0

---

## Item 18 [Hard] -- PASS

**doc_id:** e253b7fd1109bd5f87966022eea7762f
**Raw source:** events=4, relations=3
**Path (3 hops):** "crowded" -> "took place" -> "investigated" -> "closed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
**Answer phrase:** "closed in January 2003 without any indictments as Gerdt was the sole suspect" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "After the shopping center was especially crowded with many children present and the bombing took place, how was the investigation concluded?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Without indictments, closed. correct=1.0 confidence=0.0

---

## Item 19 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Path (1 hops):** "ordered" -> "minimize"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "he actively sought to minimize the atrocities" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did Brant do in response to the atrocities at Cherry Valley?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
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
**Question:** "What event followed the western-most Allied corps surrounding and isolating Bayonne and the remaining two Allied corps pushing Soult's army back to Orthez?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Soult offered battle. correct=1.0 confidence=0.0

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
**Question:** "What did the members of the Irgun do in the basement of the hotel's main building to carry out the attack on British authorities? ?" (status=ok, attempts=3)
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
**Question:** "After Operation Deny Flight began and provided close air support for UN troops, how did NATO's actions during the mission help demonstrate its adaptation to the post-Cold War era?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Engaged in combat, showed versatility. correct=1.0 confidence=0.0

---

## Item 25 [Easy] -- FAIL

**doc_id:** 28a13a10cb57f8245b1f98270bad9860
**Raw source:** events=2, relations=1
**Path (1 hops):** "diminished" -> "establish"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "the United States began to establish a firm presence in what would become Minnesota" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What economic resource became key in Minnesota during the 19th century after fur trade declined?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
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
**Question:** "After the Ottomans stopped the Russian advance at Silistra and their fleet was destroyed at Sinop, what did France and Britain do to prevent further Russian aggression?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no
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
**Question:** "What did the Crimean War prove to be for Nikolaevan Russia's educated elites? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 0 prior events, need >= 1 [FAIL]
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
