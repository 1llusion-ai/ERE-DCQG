# Path Pilot 90v2 — Hard Path Manual Review

**Date:** 2026-05-02
**Run:** `outputs/runs/path_pilot_90v2/`
**Policy change:** Hard no longer auto-rejects on `single_sentence_risk=high`. Uses `can_write_path_dependent_question` instead.

## Hard Results

| Metric | v1 (old policy) | v2 (new policy) |
|--------|----------------:|----------------:|
| Hard kept | 2/30 (7%) | 26/30 (87%) |
| Hard rejected | 28/30 | 4/30 |

### `can_write_path_dependent_question` for Hard

| Label | Count |
|-------|------:|
| yes   | 12 |
| partial | 14 |
| no | 4 |

---

## Rejected Hard (4 items) — All Valid

| # | Trigger | Phrase | Verdict | Why rejected is correct |
|---|---------|--------|---------|------------------------|
| 9 | wearing | "wearing a conical" | **Correct reject** | Truncated phrase, no complete answer. |
| 14 | developing | "The event is developing like the industry..." | **Correct reject** | Vague, no path dependency. |
| 21 | determined | "His motive was not determined" | **Correct reject** | Negative statement, answerable from one sentence. |
| 22 | held | "public inquiry into the circumstances..." | **Correct reject** | Truncated phrase + no actual inquiry held. |

All 4 rejections are correct. The new policy is not too permissive.

---

## Kept Hard — path_dep=yes (12 items)

These are the strongest Hard items: the LLM confirms questions can be written that REQUIRE path knowledge.

| # | Trigger | Phrase | Quality | Notes |
|---|---------|--------|---------|-------|
| 3 | started | YPG forces started the December 2014 Sinjar offensive | **Good** | Clear, complete answer. Path provides context about Kurdish forces. |
| 5 | injuries | injuries occurred in the road outside the hotel | **Good** | Path events explain the bombing chain. |
| 7 | signed on | signed on 30 March | **OK** | Short but complete. Path gives signing context. |
| 10 | conflict | Deny Flight led to conflict between the two organizations | **Good** | Multi-event causal chain. |
| 11 | injuries | injuries occurred in the road outside the hotel | **Good** | Duplicate of #5. |
| 15 | broke | This offensive broke ISIL's troop transport routes | **Good** | Military operation chain. |
| 16 | developing | The event is developing like the industry | **Weak** | Same trigger as #14 but different path — LLM says yes here. Borderline. |
| 20 | agreed | agreed not to seek territories or wage war | **Good** | Treaty negotiation chain. |
| 24 | victory | the skirmish resulted in Polish victory | **Good** | Battle chain. |
| 25 | transfer | the transfer of sovereignty from Britain to China | **Good** | Historical chain. |
| 28 | drove | drove the Iroquois out of western New York | **Good** | Classic multi-hop. |
| 29 | closed | closed in January 2003 without any indictments | **Good** | Investigation chain. |

**Verdict:** 11/12 good, 1 weak.

---

## Kept Hard — path_dep=partial (14 items)

These are borderline: the LLM says path events COULD be mentioned but are not strictly necessary.

| # | Trigger | Phrase | Quality | Notes |
|---|---------|--------|---------|-------|
| 1 | recover | had been run down by a car earlier in the night | **Weak** | Phrase truncated. Context helps but answer is in one sentence. |
| 2 | make | Girl Tour incorporated multimedia components to make the show | **Weak** | Path adds context but answer is local. |
| 4 | take | had been run down by a car earlier in the night | **Weak** | Same as #1. |
| 6 | recover | had been run down by a car earlier in the night | **Weak** | Duplicate. |
| 8 | keep | whom the Japanese wanted to keep out of the war | **OK** | Diplomatic chain adds value. |
| 12 | appearance | This was Hong Kong's last appearance at the Games | **OK** | Historical context chain. |
| 13 | became | many French soldiers became prisoners | **OK** | Battle → capture chain. |
| 17 | joined | Jason Duffy joined the band | **Weak** | Biographical, path adds little. |
| 18 | achieved | by the time statehood was achieved | **Weak** | Truncated phrase. |
| 19 | turned | scholars have also turned their attention | **Weak** | Academic, path dependency unclear. |
| 23 | minimize | he actively sought to minimize the atrocities | **OK** | Investigation chain. |
| 26 | keep | whom the Japanese wanted to keep out of the war | **OK** | Same as #8. |
| 27 | closed | closed in January 2003 without any indictments | **OK** | Same doc as #29. |
| 30 | commented | Randy Taraborrelli commented | **Weak** | Celebrity biography, weak path dependency. |

**Verdict:** 6/14 genuinely useful, 8/14 weak but not wrong.

---

## Summary

| Category | Count | % of Hard |
|----------|------:|----------:|
| Strong (path_dep=yes, good phrase) | 11 | 37% |
| Borderline (path_dep=partial) | 14 | 47% |
| Correctly rejected | 4 | 13% |
| Incorrectly kept | 1 | 3% |

### Key Findings

1. **New policy is correct:** All 4 rejections are valid. The old policy (auto-reject on single_sentence_risk=high) was throwing away 26 usable Hard items.

2. **"partial" path dependency is real but noisy:** 14 items have partial path dependency. Some genuinely benefit from path context (investigation chains, battle sequences). Others are biographical/vague where the path adds little.

3. **Hard path quality varies by domain:** MAVEN-ERE military/political documents produce the best Hard paths. Celebrity bios and vague "developing" events produce the weakest.

4. **Recommendation:** Use `can_write_path_dependent_question` as the primary Hard filter. Consider further filtering `partial` items by relation group (CAUSE/SUBEVENT > TEMPORAL-only) to improve Hard quality.

### Compared to v1

| Level | v1 Keep% | v2 Keep% | Change |
|-------|---------:|---------:|--------|
| Easy | 100% | 97% | -3% (1 more rejection) |
| Medium | 97% | 100% | +3% |
| Hard | 7% | 87% | **+80%** |
| Overall | 68% | 94% | +26% |
