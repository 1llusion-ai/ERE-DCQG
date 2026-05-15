# QG Pilot Report (Strict 29 v2)

**Input:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl`
**Method:** PathQG-HardAware
**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)
**Total selected:** 29

**Traces:**
- full trace: `outputs\runs\qg_pilot_strict_29_v2/debug_traces/full_trace.jsonl`
- readable trace: `outputs\runs\qg_pilot_strict_29_v2/debug_traces/readable_trace.md`

## v1 vs v2 Comparison

| Metric | v1 | v2 | Change |
|--------|---:|---:|-------:|
| Generated | 28 | 27 | -1 |
| Filter pass | 7 | 10 | +3 |
| Path coverage fails | 14 | 14 | 0 |
| Hard degraded fails | 5 | 0 | -5 |
| Answer consistency fails | 5 | 5 | 0 |
| Judge JSON errors | 8 | 4 | -4 |
| Solver correct | - | 4/10 | - |

### Per-Difficulty Comparison

| Metric | v1 | v2 |
|--------|---:|---:|
| Easy filter pass | 3/10 | 5/10 |
| Medium filter pass | 1/10 | 5/10 |
| Hard filter pass | 3/9 | 0/9 |
| Easy solver correct | 0/0 | 2/5 |
| Medium solver correct | 0/0 | 2/5 |
| Hard solver correct | 0/0 | 0/0 |

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
| Easy | 5 | 5 | 50% |
| Medium | 5 | 5 | 50% |
| Hard | 0 | 9 | 0% |
| **Total** | **10** | **19** | **34%** |

## Solver + Judge Results

| Difficulty | Solver OK | Solver Correct | Answerable | Support Covered | Composite |
|------------|----------:|---------------:|-----------:|----------------:|----------:|
| Easy | 5 | 2 (40%) | 100% | 100% | 0.763 |
| Medium | 5 | 2 (40%) | 80% | 100% | 0.700 |
| Hard | 0 | - | - | - | - |
| **Total** | **10** | **4 (40%)** | **90%** | **100%** | **0.732** |

## Top Filter Fail Reasons

| Reason | Count |
|--------|------:|
| path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL] | 6 |
| path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL] | 4 |
| answer_consistency=no: extracted from text: asks=no ans=yes cons=no | 2 |
| path_coverage=covers 0 all events in the path, need >= 1 [FAIL] | 2 |
| generation_error | 2 |
| answer_consistency=no: The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves. | 1 |
| filter_exception: AttributeError: 'str' object has no attribute 'get' | 1 |
| path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL] | 1 |
| grammar=repeat_question_mark | 1 |
| answer_consistency=no: skipped (early exit) | 1 |
| path_coverage=skipped (early exit) | 1 |
| answer_consistency=no: extracted from text: asks=no ans=no cons=no | 1 |

## Path Coverage

| Difficulty | Avg Prior Coverage | Coverage Pass% |
|------------|-------------------:|---------------:|
| Easy | 0.7 | 60% |
| Medium | 0.7 | 50% |
| Hard | 1.0 | 11% |

## Hard Shortcut Analysis

- Hard items: 9
- Degraded (shortcut=yes AND needs_prior=no): 0
- shortcut_without_path=yes: 0
- needs_prior=no: 0

## Good Examples (5)

### [Medium] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** began -> expanded -> bombed
- **question:** What was the first significant action taken by NATO aircraft during Operation Deny Flight after the mission's scope was expanded?
- **gold_answer:** NATO aircraft first bombed ground targets in an operation near Goražde
- **solver_answer:** Bombing ground targets near Goražde.
- **judge_correct:** 1.0
- **composite:** 0.967

### [Easy] Death of Joy Gardner

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **path:** attention -> inquest
- **question:** What has not been conducted despite the case becoming a cause célèbre?
- **gold_answer:** no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held
- **solver_answer:** No coroner inquest was conducted.
- **judge_correct:** 1.0
- **composite:** 0.917

### [Medium] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** spanned -> providing -> taken
- **question:** What significant event occurred during the span of Deny Flight that led to increased tension between UN and NATO forces?
- **gold_answer:** UN peacekeepers were taken as hostages in response to NATO bombing
- **solver_answer:** UN peacekeepers taken as hostages.
- **judge_correct:** 1.0
- **composite:** 0.917

### [Easy] Cyclone Forrest

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **path:** reached -> produced
- **question:** What did the system do in Thailand after reaching its peak intensity?
- **gold_answer:** the system produced significant storm
- **solver_answer:** Produced significant storm surge.
- **judge_correct:** 1.0
- **composite:** 0.833

### [Easy] Death and state funeral of Raúl Alfonsín

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **path:** died -> funeral
- **question:** What significant event was organized by Vice President Julio Cobos after Raúl Alfonsín's death?
- **gold_answer:** a state funeral at the Palace of the Argentine National Congress
- **solver_answer:** onosons the Palace of the Argentine Congress.
- **judge_correct:** 0.0
- **composite:** 0.717


## Bad Examples (5)

### [Medium] 2006 state of emergency in the Philippines

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **path:** occurred -> revocation -> lifted
- **question:** What action did the government take on March 3, 2006, one week after proclaiming the state of emergency, and what immediate effects did this have on public activities and demonstrations?
- **filter_pass:** False
- **filter_reason:** answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Hard] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** uprising -> approached -> refused -> ordered
- **question:** During the uprising, when the Russians approached the Polish quarters and the commander tried to negotiate, what did Rogaliński decide to do?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 1 PRIOR events (all events EXCEPT the last one), need >= 2 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Battle of Malacca (1641)

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **path:** capture -> casualties -> rallied
- **question:** What event brought the Sultanate of Johor and the Dutch together in a strategic alliance? 
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Myyrmanni bombing

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **path:** took place -> released
- **question:** What happened to the victims after they were treated?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 all events in the path, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** refused -> fight -> lost
- **question:** What happened to the Polish commander after he refused to negotiate with the Russians?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 PRIOR events (all events EXCEPT the last one), need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?


## Conclusion

- Selected: 29 (Easy 10, Medium 10, Hard 9)
- Generated: 27/29 (93%)
- Filter pass: 10/29 (34%)
- Solver correct: 4/10 (40%)
- Path coverage fails: 14
- Hard shortcut fails: 0
- Answer consistency fails: 5