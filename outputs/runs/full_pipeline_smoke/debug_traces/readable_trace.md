# Full-Chain Debug Trace

**Generated:** 2026-05-06T01:08:26.933525
**Total items:** 3

---

## Item 0 [Easy] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=2, relations=1
**Graph:** nodes=15, edges=50
**Path (1 hops):** "took place" -> "drove"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=not_run questionable= recommended=
**Question:** "" (status=error, attempts=3)
**Grammar:** FAIL
**Filter reason:** grammar=empty; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 1 [Medium] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=3, relations=2
**Graph:** nodes=15, edges=50
**Path (2 hops):** "committed" -> "descended on" -> "drove"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=not_run questionable= recommended=
**Question:** "What significant action followed the massacre that contributed to the displacement of the Iroquois from their lands? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---

## Item 2 [Hard] -- FAIL

**doc_id:** c0c67db40cd5e2e03645ff1116fafcfc
**Raw source:** events=4, relations=3
**Graph:** nodes=15, edges=50
**Path (3 hops):** "permitted" -> "restrain" -> "minimize" -> "drove"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/SIMULTANEOUS, TEMPORAL/BEFORE
**Answer phrase:** "drove the Iroquois out of western New York" (status=complete)
**Prefilter:** PASS -- pass [risk: temporal_only_hard]
**Path judge:** status=not_run questionable= recommended=
**Question:** "How did Butler's poor treatment of Joseph Brant and the subsequent public outcry against the atrocities contribute to the Iroquois' displacement?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 0 prior events, need >= 2 [FAIL]
**Solver:** status=not_run result= correct=None confidence=0.0

---
