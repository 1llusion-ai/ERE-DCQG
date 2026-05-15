# Full-Chain Debug Trace

**Generated:** 2026-05-02T17:46:24.662417
**Total items:** 6

---

## Item 0 [Easy] -- PASS

**doc_id:** 5e7320885b59230124849b185c013774
**Raw source:** events=2, relations=1
**Path (1 hops):** "War" -> "defeat"
**Relations:** TEMPORAL/CONTAINS
**Answer phrase:** "The Battle of the Atlantic was the longest continuous military campaign in World War II, running from 1939 to the defeat of Nazi Germany in 1945" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "When did the Battle of the Atlantic end?" (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=May 1945 correct=1.0 confidence=0.0

---

## Item 1 [Easy] -- PASS

**doc_id:** 972a33cf02e5a639ae961fbc6dc0439d
**Raw source:** events=2, relations=1
**Path (1 hops):** "struggle" -> "march"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "Approximately 250,000-500,000 people took part in each march" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "How many people participated in each Solidarity Day march? " (status=ok, attempts=3)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=250,000-500,000 correct=1.0 confidence=0.0

---

## Item 2 [Easy] -- PASS

**doc_id:** 6fc6538bd2c19d34942f7f36274d83ae
**Raw source:** events=2, relations=1
**Path (1 hops):** "sent" -> "requests"
**Relations:** TEMPORAL/BEFORE
**Answer phrase:** "The Department of Social Security (DSS) sent employees to receive claims for damage, requests for financial aid" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=easy
**Question:** "What did the Department of Social Security (DSS) do after sending employees to damaged areas?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** all checks passed
**Solver:** status=ok result=Received claims, aid requests, unemployment filings. correct=1.0 confidence=0.0

---

## Item 3 [Medium] -- FAIL

**doc_id:** 278aaf2b10664637fdc74612c3ea3012
**Raw source:** events=3, relations=2
**Path (2 hops):** "attack" -> "detected" -> "damaged"
**Relations:** CAUSE/PRECONDITION, TEMPORAL/BEFORE
**Answer phrase:** "damaged beyond repair by Kaafjord's defenders" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What happened to the British aircraft after they were detected by German radar stations during the Operation Mascot raid?" (status=ok, attempts=1)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
**Solver:** status=skipped result= correct=None confidence=0.0

---

## Item 4 [Medium] -- FAIL

**doc_id:** 47ef7d339cee55cd04ec8e4f3244fc1c
**Raw source:** events=3, relations=2
**Path (2 hops):** "War" -> "constructed" -> "assault"
**Relations:** TEMPORAL/CONTAINS, TEMPORAL/BEFORE
**Answer phrase:** "Arriving there in early June, troops were landed on Long Island near Sullivan's Island where Colonel William Moultrie commanded a partially constructed fort, in preparation for a naval bombardment and land assault" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=partial recommended=medium
**Question:** "What preparations were made by General Henry Clinton and Admiral Sir Peter Parker before their planned attack on Long Island near Sullivan's Island during the American Revolutionary War?" (status=ok, attempts=2)
**Grammar:** PASS
**Filter reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
**Solver:** status=skipped result= correct=None confidence=0.0

---

## Item 5 [Medium] -- FAIL

**doc_id:** cafed1ab3af5b65670d231294a3a18a7
**Raw source:** events=3, relations=2
**Path (2 hops):** "airlift" -> "armored" -> "pushed"
**Relations:** TEMPORAL/BEFORE, TEMPORAL/BEFORE
**Answer phrase:** "In the north, 2nd Battalion, 8th Marines (2/8) pushed into Garmsir district" (status=complete)
**Prefilter:** PASS -- pass
**Path judge:** status=ok questionable=yes recommended=medium
**Question:** "What event was the largest Marine offensive since since since since the Battle of Fallujah in 24?" (status=ok, attempts=2)
**Grammar:** FAIL
**Filter reason:** grammar=base: word repetition: since; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
**Solver:** status=skipped result= correct=None confidence=0.0

---
