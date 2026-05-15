# Full-Chain Debug Trace

**Generated:** 2026-05-02T17:43:34.108307
**Total items:** 5

---

## Item 0 [Medium] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=3, relations=2
**Graph:** nodes=15, edges=50
**Path (2 hops):** "committed" -> "permitted" -> "restrain"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "he was powerless to restrain the Seneca" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What was Butler claiming he could not do in response to accusations about the atrocities committed by the Seneca during the Battle of Wyoming?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 1 [Hard] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=4, relations=3
**Graph:** nodes=15, edges=50
**Path (3 hops):** "took place" -> "descended on" -> "minimize" -> "drove"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "" (status=not_run, attempts=0)
**Grammar:** FAIL
**Filter reason:** skipped_by_path_judge: hard_single_sentence_risk=high
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 2 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Graph:** nodes=15, edges=50
**Path (1 hops):** "took place" -> "drove"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What action led to the Iroquois being driven out of western New York in 1779?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0/2 events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 3 [Medium] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=3, relations=2
**Graph:** nodes=15, edges=50
**Path (2 hops):** "committed" -> "descended on" -> "drove"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What event followed the massacre at Cherry Valley that led to the Iroquois being driven out of western New York?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0/3 events, need >= 2 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 4 [Hard] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=4, relations=3
**Graph:** nodes=15, edges=50
**Path (3 hops):** "permitted" -> "restrain" -> "minimize" -> "drove"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/SIMULTANEOUS, TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=ok questionable=yes recommended=hard
**Question:** "" (status=not_run, attempts=0)
**Grammar:** FAIL
**Filter reason:** skipped_by_path_judge: hard_single_sentence_risk=high
**Solver:** status=not_run result= correct=None confidence=0.0

---
