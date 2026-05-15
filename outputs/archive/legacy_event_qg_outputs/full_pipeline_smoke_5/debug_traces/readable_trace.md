# Full-Chain Debug Trace

**Generated:** 2026-05-01T04:05:56.071843
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
**Path judge:** status=api_error questionable=unknown recommended=unknown
**Question:** "What happened to some of the people outside the hotel and in nearby buildings after the Irgun sent warnings, including one to the hotel's switchboard, which was likely ignored due to previous hoax warnings?" (attempts=1)
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
**Answer phrase:** "Raven IV was employed for the ZOMBie RAM RAMrod and Morgenstern devices." (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=api_error questionable=unknown recommended=unknown
**Question:** "What happened to RACER IV after it was redesigned and proof-tested in the Simon test?" (attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=tested correct=0.0 confidence=0.0

---

## Item 2 [Medium] -- FAIL

**doc_id:** 37d153abeafe0477ce6a6398e26eb442
**Raw source:** events=3, relations=2
**Graph:** nodes=35, edges=150
**Path (2 hops):** "shot" -> "land" -> "incident"
**Relations:** CAUSE/PRECONDITION, CAUSE/CAUSE
**Answer phrase:** "provoke an incident with the Europeans" (status=complete)
**Prefilter:** PASS -- pass [risk: single_sentence_risk_high]
**Path judge:** status=api_error questionable=unknown recommended=unknown
**Question:** "Why the Japanese avoid to use naval artillery strikes in the area where the battle took took in Shanghai, because  because the Suzhou Creek?," (attempts=3)
**Grammar:** FAIL
**Filter reason:** grammar=base: no question mark; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 3 [Easy] -- FAIL

**doc_id:** f46091471f38006751fcdcda15d5775b
**Raw source:** events=2, relations=1
**Graph:** nodes=37, edges=205
**Path (1 hops):** "received" -> "decided"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Deciding to ignore the bomb threats" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=api_error questionable=unknown recommended=unknown
**Question:** "What did the hotel staff decide based on the telephone warnings?" (attempts=1)
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
**Answer phrase:** "test" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=api_error questionable=unknown recommended=unknown
**Question:** "After the RACER device was shot and tested in thermonuclear system mockup assemblies, which devices used its redesigned primary in Operation Castle? " (attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 1/4 events, need >= 2 [FAIL]; hard_degraded=can_answer_from_single_sentence=yes (sent=S5)
**Solver:** status=not_run result= correct=None confidence=0.0

---
