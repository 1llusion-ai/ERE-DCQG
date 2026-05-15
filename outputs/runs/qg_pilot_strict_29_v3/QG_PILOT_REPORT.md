# QG Pilot Report (Strict 29 v3)

**Input:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl`
**Method:** PathQG-HardAware
**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)
**Total selected:** 29

**Traces:**
- full trace: `outputs\runs\qg_pilot_strict_29_v3/debug_traces/full_trace.jsonl`
- readable trace: `outputs\runs\qg_pilot_strict_29_v3/debug_traces/readable_trace.md`

## v2 vs v3 Comparison

| Metric | v2 | v3 | Change |
|--------|---:|---:|-------:|
| Generated | 27 | 27 | 0 |
| Filter pass | 10 | 15 | +5 |
| Easy filter pass | 5/10 | 2/10 | |
| Medium filter pass | 5/10 | 7/10 | |
| Hard filter pass | 0/9 | 6/9 | |
| Path coverage fails | 14 | 9 | -5 |
| Hard path coverage fails | 7 | 1 | -6 |
| Hard degraded fails | 0 | 0 | 0 |
| Answer consistency fails | 5 | 7 | +2 |
| Judge JSON errors | 4 | 6 | +2 |
| Filter exceptions | 1 | 0 | -1 |
| Solver correct | 0/0 | 7/15 | |

### Per-Difficulty Comparison

| Metric | v2 | v3 |
|--------|---:|---:|
| Easy filter pass | 5/10 | 2/10 |
| Medium filter pass | 5/10 | 7/10 |
| Hard filter pass | 0/9 | 6/9 |
| Easy solver correct | 0/0 | 1/2 |
| Medium solver correct | 0/0 | 2/7 |
| Hard solver correct | 0/0 | 4/6 |

## Selection

| Difficulty | Selected |
|------------|---------:|
| Easy | 10 |
| Medium | 10 |
| Hard | 9 |
| **Total** | **29** |

## Generation Success

| Difficulty | Generated | Error | Gen% |
|------------|----------:|------:|-----:|
| Easy | 9 | 1 | 90% |
| Medium | 9 | 1 | 90% |
| Hard | 9 | 0 | 100% |
| **Total** | **27** | **2** | **93%** |

## Question Filter

| Difficulty | Filter Pass | Filter Fail | Pass% |
|------------|------------:|------------:|------:|
| Easy | 2 | 8 | 20% |
| Medium | 7 | 3 | 70% |
| Hard | 6 | 3 | 67% |
| **Total** | **15** | **14** | **52%** |

## Solver + Judge Results

| Difficulty | Solver OK | Solver Correct | Answerable | Support Covered | Composite |
|------------|----------:|---------------:|-----------:|----------------:|----------:|
| Easy | 2 | 1 (50%) | 100% | 100% | 0.750 |
| Medium | 7 | 2 (29%) | 100% | 100% | 0.760 |
| Hard | 6 | 4 (67%) | 100% | 100% | 0.881 |
| **Total** | **15** | **7 (47%)** | **100%** | **100%** | **0.807** |

## Top Filter Fail Reasons

| Reason | Count |
|--------|------:|
| path_coverage=covers 0 all events, need >= 1 [FAIL] | 5 |
| answer_consistency=no: extracted from text: asks=no ans=yes cons=no | 3 |
| generation_error | 2 |
| path_coverage=covers 0 prior events, need >= 1 [FAIL] | 2 |
| answer_consistency=no: skipped (early exit) | 2 |
| path_coverage=skipped (early exit) | 2 |
| answer_consistency=no: The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves. | 1 |
| answer_consistency=no: extracted from text: asks=yes ans=no cons=no | 1 |
| grammar=base: word repetition: take | 1 |
| grammar=repeat_question_mark | 1 |

## Path Coverage

| Difficulty | Avg Prior Coverage | Coverage Pass% |
|------------|-------------------:|---------------:|
| Easy | 0.3 | 30% |
| Medium | 0.8 | 70% |
| Hard | 2.1 | 89% |

## Hard Shortcut Analysis

- Hard items: 9
- Degraded (shortcut=yes AND needs_prior=no): 0
- shortcut_without_path=yes: 0
- needs_prior=no: 0

## Hard Per-Item Path Coverage

| item_id | question | prior_count | all_count | pass? | reason |
|---------|----------|------------:|----------:|-------|--------|
| 1 | During the uprising, when the Russians approached the Polish | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 10 | After the Russians were stopped at Silistra and their fleet  | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 13 | When did the warnings and the hotel manager notification occ | 0 | 0 | FAIL | skipped (early exit) |
| 18 | After the shopping center was especially crowded with many c | 3 | 3 | PASS | covers 3 prior events, need >= 2 [PASS] |
| 20 | When the uprising began and the Polish refused to negotiate, | 3 | 3 | PASS | covers 3 prior events, need >= 2 [PASS] |
| 21 | What event followed the western-most Allied corps surroundin | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 22 | After the Dutch began their campaign to destroy Portuguese p | 2 | 3 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 24 | After Operation Deny Flight began and provided close air sup | 3 | 4 | PASS | covers 3 prior events, need >= 2 [PASS] |
| 26 | After the Ottomans stopped the Russian advance at Silistra a | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |

## Good Examples (5)

### [Hard] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** began -> providing -> helped -> adapted
- **question:** After Operation Deny Flight began and provided close air support for UN troops, how did NATO's actions during the mission help demonstrate its adaptation to the post-Cold War era?
- **gold_answer:** NATO had adapted to the post-Cold War era and could operate in environments other than a major force on force engagement on the plains of Central Europe
- **solver_answer:** Engaged in combat, showed versatility.
- **judge_correct:** 1.0
- **composite:** 1.0

### [Medium] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** refused -> fight -> lost
- **question:** What happened to the Polish commander after he refused to negotiate with the Russians?
- **gold_answer:** lost his eye
- **solver_answer:** Lost an eye.
- **judge_correct:** 1.0
- **composite:** 0.967

### [Medium] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** began -> expanded -> bombed
- **question:** What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission scope was expanded?
- **gold_answer:** NATO aircraft first bombed ground targets in an operation near Goražde
- **solver_answer:** Bombing ground targets near Goražde.
- **judge_correct:** 1.0
- **composite:** 0.967

### [Hard] Myyrmanni bombing

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **path:** crowded -> took place -> investigated -> closed
- **question:** After the shopping center was especially crowded with many children present and the bombing took place, how was the investigation concluded?
- **gold_answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **solver_answer:** Without indictments, closed.
- **judge_correct:** 1.0
- **composite:** 0.967

### [Hard] Battle of Orthez

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **path:** Moving -> surrounded -> pushed -> battle
- **question:** What event followed the western-most Allied corps surrounding and isolating Bayonne and the remaining two Allied corps pushing Soult's army back to Orthez?
- **gold_answer:** the French marshal offered battle
- **solver_answer:** Soult offered battle.
- **judge_correct:** 1.0
- **composite:** 0.967


## Bad Examples (5)

### [Medium] 2006 state of emergency in the Philippines

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **path:** occurred -> revocation -> lifted
- **question:** 
- **filter_pass:** False
- **filter_reason:** generation_error
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Myyrmanni bombing

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **path:** took place -> released
- **question:** How were the 100 victims who did not require hospitalization after the Myyrmanni bombing treated?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] King David Hotel bombing

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **path:** Disguised -> operation
- **question:** What happened after the Irgun members had disguised themselves?
- **filter_pass:** False
- **filter_reason:** answer_consistency=no: The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves.
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Who's That Girl World Tour

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **path:** trained -> addressing -> wearing
- **question:** How did Marlene Stewart contribute to the costumes for the Who's That Girl Tour, and what was the impact of these costumes on the show?
- **filter_pass:** False
- **filter_reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Death of Joy Gardner

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **path:** attention -> inquest
- **question:** What did not happen despite the case becoming a cause célèbre?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?


## Conclusion

- Selected: 29 (Easy 10, Medium 10, Hard 9)
- Generated: 27/29 (93%)
- Filter pass: 15/29 (52%)
- Solver correct: 7/15 (47%)
- Path coverage fails: 9
- Hard shortcut fails: 0
- Answer consistency fails: 7