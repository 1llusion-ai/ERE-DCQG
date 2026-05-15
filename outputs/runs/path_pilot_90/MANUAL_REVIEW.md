# Path Pilot 90 — Manual Review

**Date:** 2026-05-02
**Run:** `outputs/runs/path_pilot_90/`
**Total judged:** 90 (30/level)
**LLM kept:** 61 (Easy 30, Medium 29, Hard 2)

---

## Per-Level LLM Judge

| Level | Total | Kept | Rejected | Keep% |
|-------|------:|-----:|---------:|------:|
| Easy  |    30 |   30 |        0 | 100%  |
| Medium|    30 |   29 |        1 |  97%  |
| Hard  |    30 |    2 |       28 |   7%  |

**Rejection reasons:** 28/29 = single_sentence_shortcut (Hard paths answerable from one sentence)

---

## Manual Review: 20 Kept Items

### Easy (8 items)

| # | Trigger | Answer Phrase | Verdict | Notes |
|---|---------|---------------|---------|-------|
| 1 | Starting | Starting 29 October afternoon, a massive popular procession accompanied the remains of Néstor Kirchner from Casa Rosada to the metropolitan airport | **OK** | Long but complete temporal answer |
| 2 | deaths | The majority of deaths associated with Forrest resulted from a plane crash on November 14 in Vietnam | **OK** | Clear, complete |
| 3 | injured | 166 people were injured, including 10 children | **OK** | Concise, complete |
| 4 | started | YPG forces started the December 2014 Sinjar offensive with the support of US airstrikes | **OK** | Good answer phrase |
| 5 | assumed | Nicaragua assumed a quasi-protectorate status under the 1916 Bryan-Chamorro Treaty | **OK** | Complete |
| 6 | mourning | arranged three days of national mourning | **OK** | Complete noun phrase |
| 7 | showcase | new artists showcase their material | **Weak** | Fragment, not a full clause; but functionally OK as answer |
| 8 | deaths | The majority of deaths associated with Forrest resulted from a plane crash | **OK** | Complete |

**Easy verdict:** 7/8 good, 1 weak but usable.

### Medium (8 items)

| # | Trigger | Answer Phrase | Verdict | Notes |
|---|---------|---------------|---------|-------|
| 1 | repelled | The outnumbered French repelled several Allied assaults on their right flank | **OK** | Complete clause |
| 2 | moved | Caroline was moved to percussion | **OK** | Short but complete |
| 3 | deployed | Hundreds of State Emergency Service (SES) volunteers were deployed to restore electrical | **Weak** | "restore electrical" is truncated — should be "restore electrical and water services". Clause boundary stopped at "and". Phrase is usable but not ideal. |
| 4 | Battle | marked the beginning of the end of the three-month Battle of Shanghai in the opening phase of the Second Sino-Japanese War | **OK** | Complete |
| 5 | transfer | the transfer of sovereignty from Britain to China | **OK** | Clean |
| 6 | prompted | prompted the evacuation of 600,000 people in Bangladesh in late November 1992 | **OK** | Complete |
| 7 | formed | The Minnesota Territory itself was formed only in 1849 | **OK** | Complete |
| 8 | Battle | defense of Sihang Warehouse marked the beginning of the end of the three-month Battle of Shanghai | **OK** | Complete |

**Medium verdict:** 7/8 good, 1 weak (truncated at clause boundary "and").

### Hard (2 items)

| # | Trigger | Answer Phrase | Verdict | Notes |
|---|---------|---------------|---------|-------|
| 1 | operate | could operate in environments other than a major force on force engagement on the plains of Central Europe | **Weak** | Starts with "could" — missing subject. Grammatically incomplete. |
| 2 | landfalls | making landfalls on Long Island | **Weak** | "making" is a participle without subject. Not a standalone answer phrase. |

**Hard verdict:** 0/2 good, 2 weak. Both are participle/modal fragments.

---

## Summary

| Quality | Easy | Medium | Hard | Total |
|---------|-----:|-------:|-----:|------:|
| Good    |    7 |      7 |    0 |    14 |
| Weak    |    1 |      1 |    2 |     4 |
| Bad     |    0 |      0 |    0 |     0 |

**14/20 (70%) are good quality.** 4/20 are weak but usable.

### Issues Found

1. **Hard keep rate is very low (7%)** — almost all Hard paths are rejected by LLM judge as single-sentence answerable. This is a known problem: 3-hop paths in MAVEN-ERE often don't add real reasoning difficulty.

2. **Clause boundary truncation** — "deployed to restore electrical" stopped at "and" (clause boundary word), truncating "electrical and water services". The clause-boundary expansion is too aggressive for coordinating conjunctions within a list.

3. **Participle/fragment answers** — "making landfalls" and "could operate in environments" are grammatically incomplete as standalone answers. These pass the current checks but are not ideal gold answers.

### Recommendations

- The dangling-word fix works correctly — no dangling prepositions in kept items.
- Consider adding a check for phrase starting with a bare participle ("making", "starting") or modal ("could", "would") without a subject.
- Consider relaxing clause-boundary expansion for "and" when it's coordinating a list (not joining independent clauses).
- Hard path quality is the biggest remaining issue — need to investigate whether Hard paths can be salvaged or if the LLM judge rejection is correct.
