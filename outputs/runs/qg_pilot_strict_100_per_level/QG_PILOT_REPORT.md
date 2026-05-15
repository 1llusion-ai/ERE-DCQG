# QG Pilot Report (Strict 29 pilot_100)

**Input:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl`
**Method:** PathQG-HardAware
**Selected paths:** `outputs/runs/qg_pilot_strict_29/selected_paths.jsonl` (same as v1)
**Total selected:** 76

**Traces:**
- full trace: `outputs\runs\qg_pilot_strict_100_per_level/debug_traces/full_trace.jsonl`
- readable trace: `outputs\runs\qg_pilot_strict_100_per_level/debug_traces/readable_trace.md`

## pilot_100 vs pilot_100 Comparison

| Metric | pilot_100 | pilot_100 | Change |
|--------|---:|---:|-------:|
| Generated | 28 | 68 | +40 |
| Filter pass | 7 | 37 | +30 |
| Easy filter pass | 3/10 | 13/32 | |
| Medium filter pass | 1/10 | 18/35 | |
| Hard filter pass | 3/9 | 6/9 | |
| Path coverage fails | 14 | 27 | +13 |
| Hard path coverage fails | 2 | 1 | -1 |
| Hard degraded fails | 5 | 0 | -5 |
| Answer consistency fails | 5 | 10 | +5 |
| Judge JSON errors | 8 | 7 | -1 |
| Filter exceptions | 0 | 0 | 0 |
| Solver correct | 0/0 | 5/37 | |

### Per-Difficulty Comparison

| Metric | pilot_100 | pilot_100 |
|--------|---:|---:|
| Easy filter pass | 3/10 | 13/32 |
| Medium filter pass | 1/10 | 18/35 |
| Hard filter pass | 3/9 | 6/9 |
| Easy solver correct | 0/0 | 1/13 |
| Medium solver correct | 0/0 | 3/18 |
| Hard solver correct | 0/0 | 1/6 |

## Selection

| Difficulty | Available | Selected |
|------------|----------:|---------:|
| Easy | 32 | 32 |
| Medium | 35 | 35 |
| Hard | 9 | 9 |
| **Total** | **76** | **76** |

## Generation Success

| Difficulty | Generated | Error | Gen% |
|------------|----------:|------:|-----:|
| Easy | 28 | 4 | 88% |
| Medium | 31 | 4 | 89% |
| Hard | 9 | 0 | 100% |
| **Total** | **68** | **8** | **89%** |

## Question Filter

| Difficulty | Filter Pass | Filter Fail | Pass% |
|------------|------------:|------------:|------:|
| Easy | 13 | 19 | 41% |
| Medium | 18 | 17 | 51% |
| Hard | 6 | 3 | 67% |
| **Total** | **37** | **39** | **49%** |

## Solver + Judge Results

| Difficulty | Solver OK | Solver Correct | Answerable | Support Covered | Composite |
|------------|----------:|---------------:|-----------:|----------------:|----------:|
| Easy | 13 | 1 (8%) | 77% | 100% | 0.617 |
| Medium | 18 | 3 (17%) | 94% | 100% | 0.683 |
| Hard | 6 | 1 (17%) | 83% | 100% | 0.706 |
| **Total** | **37** | **5 (14%)** | **86%** | **100%** | **0.663** |

## Top Filter Fail Reasons

| Reason | Count |
|--------|------:|
| path_coverage=covers 0 all events, need >= 1 [FAIL] | 11 |
| path_coverage=covers 0 prior events, need >= 1 [FAIL] | 10 |
| generation_error | 8 |
| answer_consistency=no: skipped (early exit) | 5 |
| path_coverage=skipped (early exit) | 5 |
| answer_consistency=no: extracted from text: asks=yes ans=no cons=no | 2 |
| grammar=broken_grammar: What after | 1 |
| answer_consistency=no: extracted from text: asks=no ans=no cons=no | 1 |
| answer_consistency=no: brief | 1 |
| grammar=repeat_question_mark | 1 |
| grammar=base: bad start: in | 1 |
| grammar=base: word repetition: evacuated | 1 |
| path_coverage=covers 1 prior events, need >= 2 [FAIL] | 1 |
| grammar=base: word repetition: took | 1 |
| answer_consistency=no: The question does not ask about the target final event, which is 'collaborated.' It asks about Madonna's training for choreography, which is not directly related to the target event. | 1 |

## Path Coverage

| Difficulty | Avg Prior Coverage | Coverage Pass% |
|------------|-------------------:|---------------:|
| Easy | 0.5 | 44% |
| Medium | 0.8 | 54% |
| Hard | 2.0 | 89% |

## Hard Shortcut Analysis

- Hard items: 9
- Degraded (shortcut=yes AND needs_prior=no): 0
- shortcut_without_path=yes: 0
- needs_prior=no: 0

## Hard Per-Item Path Coverage

| item_id | question | prior_count | all_count | pass? | reason |
|---------|----------|------------:|----------:|-------|--------|
| 4 | After the Ottomans stopped the Russian advance at Silistra a | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 10 | When did the warnings lead the I on the Irgun on carried the | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 11 | After the Dutch began their campaign to destroy Portuguese p | 2 | 3 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 37 | After the Russians were stopped at Silistra and their fleet  | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 40 | When the uprising began and the Polish refused to negotiate, | 3 | 3 | PASS | covers 3 prior events, need >= 2 [PASS] |
| 42 | During the uprising, when the Russians approached the Polish | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 63 | After the western-most Allied corps surrounded and isolated  | 1 | 1 | FAIL | covers 1 prior events, need >= 2 [FAIL] |
| 66 | How was the Myyrmanni shopping mall when the bombing took pl | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |
| 68 | After the operation began and started providing close air su | 2 | 2 | PASS | covers 2 prior events, need >= 2 [PASS] |

## Good Examples (5)

### [Medium] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** spanned -> providing -> taken
- **question:** What significant event occurred during the span of Deny Flight that led to increased tension between UN and NATO forces?
- **gold_answer:** UN peacekeepers were taken as hostages in response to NATO bombing
- **solver_answer:** UN peacekeepers taken as hostages.
- **judge_correct:** 1.0
- **composite:** 0.967

### [Hard] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** uprising -> approached -> refused -> ordered
- **question:** During the uprising, when the Russians approached the Polish quarters and the commander tried to negotiate, what did Rogaliński decide to do next?
- **gold_answer:** ordered a charge of the Russians
- **solver_answer:** Rogaliński refused to negotiate.
- **judge_correct:** 1.0
- **composite:** 0.967

### [Medium] Cyclone Forrest

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **path:** enacted -> turned -> evacuation
- **question:** What event was successfully carried out, leading to the relocation of 600,000 people and preventing a potential disaster during Cyclone Forrest's approach to Bangladesh in 1992?
- **gold_answer:** prompted the evacuation of 600,000 people in Bangladesh in late November 1992
- **solver_answer:** Mass evacuation plans
- **judge_correct:** 1.0
- **composite:** 0.917

### [Medium] Battle of Orthez

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **path:** isolated -> overcome -> decided
- **question:** What decision did Marshal Soult make after his army was isolated and forced to retreat during the subsequent operations following the Battle of Orthez?
- **gold_answer:** Soult decided to abandon the large western port of Bordeaux and fall back east toward Toulouse
- **solver_answer:** Abandon Bordeaux, retreat east.
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

### [Hard] Battle of Malacca (1641)

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **path:** began -> launching -> took -> agreed
- **question:** After the Dutch began their campaign to destroy Portuguese power and launched small incursions against the Portuguese, what did they agree to do in line with their agreement with Johor in 1606?
- **gold_answer:** agreed not to seek territories or wage war with the Malay kingdoms
- **solver_answer:** Not seek territories.
- **judge_correct:** 0.0
- **composite:** 0.75

### [Medium] Cyclone Forrest

- **doc_id:** `8af040b8bfa2eec89d1197392f2fdac5`
- **path:** Originating -> Tracking -> storm
- **question:** What was the name of the tropical cyclone that prompted the evacuation of 600,000 people in Bangladesh in late November 1992, and how did it develop over time?
- **gold_answer:** also referred to as Tropical Storm
- **solver_answer:** Cyclone Forrest, developed from disturbed weather.
- **judge_correct:** 0.0
- **composite:** 0.75

### [Hard] Myyrmanni bombing

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **path:** crowded -> took place -> investigated -> closed
- **question:** How was the Myyrmanni shopping mall when the bombing took place, and what was the subsequent investigation like?
- **gold_answer:** closed in January 2003 without any indictments as Gerdt was the sole suspect
- **solver_answer:** Crowded; no indictments.
- **judge_correct:** 0.0
- **composite:** 0.75

### [Hard] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** uprising -> negotiate -> refused -> killed
- **question:** When the uprising began and the Polish refused to negotiate, the Russian commander did what?
- **gold_answer:** After a short hand-to-hand fight the Russian commander was killed
- **solver_answer:** was killed
- **judge_correct:** 0.0
- **composite:** 0.717

### [Medium] Battle of Malacca (1641)

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **path:** capture -> began -> destroyed
- **question:** What significant action did the Dutch and their allies take that marked the end of Portuguese power in the Malay archipelago after they began their campaign?
- **gold_answer:** This combined Dutch-Johor effort effectively destroyed the last bastion of Portuguese
- **solver_answer:** Captured Malacca in January 1641.
- **judge_correct:** 0.0
- **composite:** 0.717


## Bad Examples (10)

### [Easy] Operation Vengeance

- **doc_id:** `06f91ced00b41867979f3d5dc6996da2`
- **path:** attack -> blamed
- **question:** What was the intended outcome of the U.S. leaders' actions during the mission to kill Yamamoto?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** ordered -> wounded
- **question:** What after the Russian commander refused to negotiate and ordered a charge of the Russians, , what the outcome of the engagement was ? 
- **filter_pass:** False
- **filter_reason:** grammar=broken_grammar: What after; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Crimean War

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **path:** prepared -> granted -> recognize
- **question:** What significant realization did Russia's educated elites come to after the Crimean War, prompting a push for fundamental transformations? 
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Hard] Crimean War

- **doc_id:** `81c576926e0c52f158b210c244028f0b`
- **path:** stopped -> destroyed -> rushed -> forbade
- **question:** After the Ottomans stopped the Russian advance at Silistra and their fleet destroyed a Russian reinforcement attempt at Sinop, what did France and Britain do to prevent further Russian aggression?
- **filter_pass:** False
- **filter_reason:** answer_consistency=no: extracted from text: asks=no ans=no cons=no
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Battle of Orthez

- **doc_id:** `db50381e7d1dd4a41fb4ac60eaebe3a4`
- **path:** Moving -> attack
- **question:** What action did the Anglo-Portuguese Army take against the Imperial French army during the Battle of Orthez in February 1814?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Territorial era of Minnesota

- **doc_id:** `28a13a10cb57f8245b1f98270bad9860`
- **path:** diminished -> establish
- **question:** What economic resource replaced furs as the key economic activity in the area during the early 19th century?
- **filter_pass:** False
- **filter_reason:** answer_consistency=no: brief; path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Who's That Girl World Tour

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **path:** supported -> commending
- **question:** What did the reviewers think about the extravagant nature of the concert and Madonna's performance during the Who's That Girl World Tour?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 all events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] King David Hotel bombing

- **doc_id:** `f46091471f38006751fcdcda15d5775b`
- **path:** Disguised -> operation
- **question:** What was the name of the event where disguised Irgun members planted a bomb in the basement of the hotel's main building, causing the collapse of the western half of the southern wing? ?
- **filter_pass:** False
- **filter_reason:** grammar=repeat_question_mark; answer_consistency=no: skipped (early exit); path_coverage=skipped (early exit)
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Who's That Girl World Tour

- **doc_id:** `6dabade56742b6040cda6a5838176f6c`
- **path:** trained -> addressing -> wearing
- **question:** What kind of attire did the statue of Madonna in Pacentro feature, and how did her physical training contribute to this choice?
- **filter_pass:** False
- **filter_reason:** answer_consistency=no: extracted from text: asks=yes ans=no cons=no
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Cherry Valley massacre

- **doc_id:** `c0c67db40cd5e2e03645ff1116fafcfc`
- **path:** restrain -> took place -> descended on
- **question:** What actions did the Loyalists, British soldiers, Seneca, and Mohawks take that led to the unpreparedness of Cherry Valley's defenders during the attack?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0 prior events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?


## Conclusion

- Selected: 76 (Easy 32, Medium 35, Hard 9)
- Generated: 68/76 (89%)
- Filter pass: 37/76 (49%)
- Solver correct: 5/37 (14%)
- Path coverage fails: 27
- Hard shortcut fails: 0
- Answer consistency fails: 10