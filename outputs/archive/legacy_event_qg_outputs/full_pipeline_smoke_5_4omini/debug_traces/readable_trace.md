# Full-Chain Debug Trace

**Generated:** 2026-05-01T04:09:32.629017
**Total items:** 5

---

## Item 0 [Medium] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=3, relations=2
**Graph:** nodes=37, edges=205
**Path (2 hops):** "planned" -> "warnings" -> "deaths"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "deaths occurred in the road caused outside the hotel and in adjacent buildings" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What happened to some of the people outside the hotel and in nearby buildings after the Irgun sent warnings, including to the hotel's switchboard, which were likely ignored due to previous false alarms?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 1 [Easy] -- PASS

**doc_id:** 975e0b195ecae47acba052c73050c1fa
**Raw source:** events=2, relations=1
**Graph:** nodes=17, edges=124
**Path (1 hops):** "test" -> "employed"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "RACER IV was employed as primary for the ZOMBIE, RAMROD and MORGENSTERN devices." (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What happened to RACER IV after it was proof-tested in the Simon test?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=redesigned and tested correct=0.0 confidence=0.0

---

## Item 2 [Medium] -- PASS

**doc_id:** 37d153abeafe0477ce6a6398e26eb442
**Raw source:** events=3, relations=2
**Graph:** nodes=35, edges=150
**Path (2 hops):** "shot" -> "land" -> "incident"
**Relations:** CAUSE/PRECONDITION, CAUSE/CAUSE
**Answer phrase:** "provoke an incident with the Europeans and" (status=complete)
**Prefilter:** PASS -- pass [risk: single_sentence_risk_high]
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What was the Japanese concern that made them avoid using naval artillery strikes and mustard gas in the area?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=fear did did not correct=0.0 confidence=0.0

---

## Item 3 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Graph:** nodes=37, edges=205
**Path (1 hops):** "received" -> "decided"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Deciding to ignore the bomb threats" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What did the hotel staff decide to do after receiving the Irgun's warning?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** answer_phrase=trigger 'decided' not in phrase 'Deciding to ignore the bomb threats'
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 4 [Hard] -- FAIL

**doc_id:** 975e0b195ecae47acba052c73050c1fa
**Raw source:** events=4, relations=3
**Graph:** nodes=17, edges=124
**Path (3 hops):** "test" -> "shot" -> "tested" -> "test"
**Relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "proof-tested in the Simon test) was employed as primary for the ZOMBIE, RAMROD" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=partial recommended=hard
**Question:** "" (status=not_run, attempts=0)
**Grammar:** FAIL
**Filter reason:** skipped_by_path_judge: hard_single_sentence_risk=high
**Solver:** status=not_run result= correct=None confidence=0.0

---
