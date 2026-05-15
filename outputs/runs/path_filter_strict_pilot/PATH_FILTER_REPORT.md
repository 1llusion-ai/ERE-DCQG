# Path Filter Report

## Pipeline Summary

| Stage | Easy | Medium | Hard | Total |
|-------|-----:|-------:|-----:|------:|
| Raw paths | - | - | - | 288 |
| Prefiltered | - | - | - | 224 |
| LLM judged | 50 | 50 | 50 | 150 |
| Strict candidates | 48 | 50 | 21 | 119 |
| Strict dedup removed | 16 | 15 | 12 | 43 |
| **Strict final** | 32 | 35 | 9 | 76 |
| Relaxed final | 31 | 34 | 22 | 87 |
| Rejected | 2 | 0 | 7 | 9 |

## Policy

- **Easy/Medium:** keep if `path_questionable in {yes, partial}` and `answer_phrase_pass=True`
- **Hard strict:** `path_questionable in {yes, partial}` AND `can_write_path_dependent_question=yes` AND `answer_phrase_pass=True`
- **Hard relaxed:** same but accepts `can_write_path_dependent_question in {yes, partial}`
- Dedup: `doc_id + answer_event_id`, fallback `doc_id + normalized(answer_phrase)`

## Hard Path Analysis

### can_write_path_dependent_question Distribution

| Label | Count | % of Hard judged |
|-------|------:|-----------------:|
| yes | 21 | 42.0% |
| partial | 22 | 44.0% |
| no | 7 | 14.0% |

### Hard Strict Pipeline

- LLM judged Hard path_dep=yes: **21/50** (42% of judged)
- Hard strict candidates before dedup: **21**
- Hard strict removed by dedup: **12**
- Hard strict final (deduped): **9**

LLM judged a significant portion of Hard paths as path-dependent, but many
share the same final answer event. Deduplication by `doc_id + answer_event_id`
reduces strict Hard paths to unique final-answer items for QG.

### Hard Strict vs Partial — Relation Group

| Relation | Strict (path_dep=yes) | Partial (path_dep=partial) |
|----------|----------------------:|---------------------------:|
| CAUSE | 7 | 9 |
| TEMPORAL | 2 | 13 |

## Top Reject Reasons

| Reason | Count |
|--------|------:|
| path_questionable=no | 9 |

## Top Weak Trigger Types

| Type | Count |
|------|------:|
| needs_phrase | 13 |

## Examples: Hard Strict Kept (5)

- **Crimean War** [Hard]
  - doc_id: `81c576926e0c52f158b210c244028f0b`
  - path: stopped -> destroyed -> rushed -> forbade
  - relations: TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
  - answer_phrase: `It forbade Russia from basing warships in the Black Sea`
  - answer_sentence: It forbade Russia from basing warships in the Black Sea.
  - judge: pq=yes risk=high path_dep=yes
  - strict_reason: keep_path_dep_yes
  - relaxed_reason: keep_path_dep_yes
  - risk_note: single_sentence_risk=high
- **Myyrmanni bombing** [Hard]
  - doc_id: `e253b7fd1109bd5f87966022eea7762f`
  - path: crowded -> took place -> investigated -> closed
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, CAUSE/PRECONDITION
  - answer_phrase: `closed in January 2003 without any indictments as Gerdt was the sole suspect`
  - answer_sentence: The incident was investigated primarily as six accounts of murder and closed in January 2003 without any indictments as 
  - judge: pq=yes risk=high path_dep=yes
  - strict_reason: keep_path_dep_yes
  - relaxed_reason: keep_path_dep_yes
  - risk_note: single_sentence_risk=high
- **King David Hotel bombing** [Hard]
  - doc_id: `f46091471f38006751fcdcda15d5775b`
  - path: warnings -> called -> carried out -> killed
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `91 people of various nationalities were killed`
  - answer_sentence: 91 people of various nationalities were killed, and 46 were injured.
  - judge: pq=yes risk=high path_dep=yes
  - strict_reason: keep_path_dep_yes
  - relaxed_reason: keep_path_dep_yes
  - risk_note: single_sentence_risk=high
- **Battle of Ciołków** [Hard]
  - doc_id: `a24058769038462f489b0091ebb24597`
  - path: uprising -> negotiate -> refused -> killed
  - relations: TEMPORAL/CONTAINS, CAUSE/PRECONDITION, TEMPORAL/BEFORE
  - answer_phrase: `After a short hand-to-hand fight the Russian commander was killed`
  - answer_sentence: After a short hand-to-hand fight (the Polish unit had only two pieces of firearms and was mostly equipped with sabres, w
  - judge: pq=yes risk=high path_dep=yes
  - strict_reason: keep_path_dep_yes
  - relaxed_reason: keep_path_dep_yes
  - risk_note: single_sentence_risk=high
- **Crimean War** [Hard]
  - doc_id: `81c576926e0c52f158b210c244028f0b`
  - path: stopped -> destroyed -> arriving -> signed on
  - relations: TEMPORAL/BEFORE, CAUSE/PRECONDITION, TEMPORAL/BEFORE
  - answer_phrase: `signed on 30 March`
  - answer_sentence: The Treaty of Paris, signed on 30 March 1856, ended the war.
  - judge: pq=yes risk=high path_dep=yes
  - strict_reason: keep_path_dep_yes
  - relaxed_reason: keep_path_dep_yes
  - risk_note: single_sentence_risk=high

## Examples: Hard Relaxed-Only Partial (5)

- **Cyclone Forrest** [Hard]
  - doc_id: `8af040b8bfa2eec89d1197392f2fdac5`
  - path: repeat -> evacuation -> evacuation -> damaged
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `Martin's Island were damaged`
  - answer_sentence: Only two deaths were recorded and overall damage was light, though half of all homes on St. Martin's Island were damaged
  - judge: pq=partial risk=high path_dep=partial
  - strict_reason: hard_path_dep=partial
  - relaxed_reason: keep_path_dep_partial
  - risk_note: single_sentence_risk=high
- **Crimean War** [Hard]
  - doc_id: `81c576926e0c52f158b210c244028f0b`
  - path: stopped -> depleting -> forbade -> granted
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `Christians there were granted a degree of official`
  - answer_sentence: Christians there were granted a degree of official equality, and the Orthodox Church regained control of the Christian c
  - judge: pq=partial risk=high path_dep=partial
  - strict_reason: hard_path_dep=partial
  - relaxed_reason: keep_path_dep_partial
  - risk_note: single_sentence_risk=high
- **Cyclone Winifred** [Hard]
  - doc_id: `6fc6538bd2c19d34942f7f36274d83ae`
  - path: came -> producing -> winds -> aid
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `requests for financial aid, and filings for unemployment benefits`
  - answer_sentence: The Department of Social Security (DSS) sent employees to receive claims for damage, requests for financial aid, and fil
  - judge: pq=partial risk=high path_dep=partial
  - strict_reason: hard_path_dep=partial
  - relaxed_reason: keep_path_dep_partial
  - risk_note: single_sentence_risk=high
- **Battle of Ciołków** [Hard]
  - doc_id: `a24058769038462f489b0091ebb24597`
  - path: uprising -> refused -> fight -> losses
  - relations: TEMPORAL/CONTAINS, TEMPORAL/BEFORE, CAUSE/CAUSE
  - answer_phrase: `Polish losses were`
  - answer_sentence: Polish losses were negligible, but the Polish commander was wounded and lost his eye.
  - judge: pq=partial risk=high path_dep=partial
  - strict_reason: hard_path_dep=partial
  - relaxed_reason: keep_path_dep_partial
  - risk_note: single_sentence_risk=high
- **Survivor Series (1999)** [Hard]
  - doc_id: `275eb0bc9caacb30e4f58d85469458d1`
  - path: delivered -> suffered -> run -> take
  - relations: CAUSE/PRECONDITION, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `had been run down by a car earlier in the night (which was an angle for Austin to take time to recover from his injuries`
  - answer_sentence: He was a replacement for Stone Cold Steve Austin, who had been run down by a car earlier in the night (which was an angl
  - judge: pq=partial risk=high path_dep=partial
  - strict_reason: hard_path_dep=partial
  - relaxed_reason: keep_path_dep_partial
  - risk_note: single_sentence_risk=high

## Examples: Hard Rejected (5)

- **Battle of Ciołków** [Hard]
  - doc_id: `a24058769038462f489b0091ebb24597`
  - path: uprising -> refused -> ordered -> lost
  - relations: TEMPORAL/CONTAINS, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `lost his eye`
  - answer_sentence: Polish losses were negligible, but the Polish commander was wounded and lost his eye.
  - judge: pq=no risk=high path_dep=no
  - strict_reason: path_questionable=no
  - relaxed_reason: path_questionable=no
  - risk_note: single_sentence_risk=high
- **Who's That Girl World Tour** [Hard]
  - doc_id: `6dabade56742b6040cda6a5838176f6c`
  - path: changes -> marked -> commented -> wearing
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `wearing a conical`
  - answer_sentence: A statue of Madonna, wearing a conical bra, was erected in her name at the center of the town of Pacentro in Italy, wher
  - judge: pq=no risk=high path_dep=no
  - strict_reason: path_questionable=no
  - relaxed_reason: path_questionable=no
  - risk_note: single_sentence_risk=high
- **2006 state of emergency in the Philippines** [Hard]
  - doc_id: `84ce009a07b987d60a79d92bc4d45744`
  - path: announced -> revocation -> suspended -> allowed
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `the government was allowed at the moment to detain anyone indefinitely without the privilege of the writ of habeas corpu`
  - answer_sentence: Under the provisions of the 1987 Constitution, the government was allowed at the moment to detain anyone indefinitely wi
  - judge: pq=no risk=high path_dep=no
  - strict_reason: path_questionable=no
  - relaxed_reason: path_questionable=no
  - risk_note: single_sentence_risk=high
- **Territorial era of Minnesota** [Hard]
  - doc_id: `28a13a10cb57f8245b1f98270bad9860`
  - path: replacing -> become -> changed -> influence
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `mixed-race populations continued to influence the territory's culture and`
  - answer_sentence: The native and mixed-race populations continued to influence the territory's culture and politics, even at the end of th
  - judge: pq=no risk=high path_dep=no
  - strict_reason: path_questionable=no
  - relaxed_reason: path_questionable=no
  - risk_note: single_sentence_risk=high
- **Death of Joy Gardner** [Hard]
  - doc_id: `dd2a791aa826766cf0d05dc8102f5c8e`
  - path: deportation -> acquitted -> orders -> inquiry
  - relations: TEMPORAL/BEFORE, TEMPORAL/BEFORE, TEMPORAL/BEFORE
  - answer_phrase: `public inquiry into the circumstances of Gardner's death has been held`
  - answer_sentence: Despite continuing pressure by campaigners, no coroner's inquest or public inquiry into the circumstances of Gardner's d
  - judge: pq=no risk=high path_dep=no
  - strict_reason: path_questionable=no
  - relaxed_reason: path_questionable=no
  - risk_note: single_sentence_risk=high

## Conclusion

- Strict paths ready for QG: **76** (Easy 32, Medium 35, Hard 9)
- Relaxed-only Hard partial candidates: **13** (for future analysis)
- Hard strict count (9) is limited. Consider running on more documents for Hard QG pilot.