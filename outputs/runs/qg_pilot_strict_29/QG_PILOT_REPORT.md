# QG Pilot Report (Strict 29)

**Input:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl`
**Method:** PathQG-HardAware
**Total selected:** 29

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
| Easy | 10 | 0 | 100% |
| Medium | 9 | 1 | 90% |
| Hard | 9 | 0 | 100% |
| **Total** | **28** | **1** | **97%** |

## Question Filter

| Difficulty | Filter Pass | Filter Fail | Pass% |
|------------|------------:|------------:|------:|
| Easy | 3 | 7 | 30% |
| Medium | 1 | 9 | 10% |
| Hard | 3 | 6 | 33% |
| **Total** | **7** | **22** | **24%** |

## Solver + Judge Results

| Difficulty | Solver OK | Solver Correct | Answerable | Support Covered | Composite |
|------------|----------:|---------------:|-----------:|----------------:|----------:|
| Easy | 3 | 1 (33%) | 67% | 100% | 0.661 |
| Medium | 1 | 1 (100%) | 100% | 100% | 0.917 |
| Hard | 3 | 0 (0%) | 100% | 67% | 0.689 |
| **Total** | **7** | **2 (29%)** | **86%** | **86%** | **0.710** |

## Top Filter Fail Reasons

| Reason | Count |
|--------|------:|
| path_coverage=covers 1/3 events, need >= 2 [FAIL] | 8 |
| path_coverage=covers 0/2 events, need >= 1 [FAIL] | 3 |
| path_coverage=covers 1/4 events, need >= 2 [FAIL] | 2 |
| generation_error | 1 |
| answer_consistency=no: The question does not specifically ask about the target final event, but rather about the actions of the Irgun members after they had disguised themselves. | 1 |
| hard_degraded=can_answer_from_single_sentence=yes (sent=S25) | 1 |
| hard_degraded=can_answer_from_single_sentence=yes (sent=S1) | 1 |
| answer_consistency=no: brief | 1 |
| hard_degraded=can_answer_from_single_sentence=yes (sent=S6) | 1 |
| answer_consistency=no: The question does not ask about the target final event, which is about the agreement with Johor. The question is about the timing and nature of the Dutch attempt to capture Malacca. | 1 |
| hard_degraded=can_answer_from_single_sentence=yes (sent=S4) | 1 |
| hard_degraded=can_answer_from_single_sentence=yes (sent=S5) | 1 |
| answer_consistency=no: The question does not ask about the target final event, but rather about the economic resource that replaced furs after the decline of the fur trade. | 1 |
| grammar=base: no question mark | 1 |
| answer_consistency=no: skipped (early exit) | 1 |

## Path Coverage

| Difficulty | Avg Coverage | Coverage Pass% |
|------------|-------------:|---------------:|
| Easy | 0.8 | 60% |
| Medium | 1.1 | 10% |
| Hard | 2.0 | 78% |

## Hard Degradation

- Hard items: 9
- Degraded (single-sentence shortcut): 5
  - `81c576926e0c52f158b210c244028f0b`: can_answer_from_single_sentence=yes (sent=S25)
  - `f46091471f38006751fcdcda15d5775b`: can_answer_from_single_sentence=yes (sent=S1)
  - `e253b7fd1109bd5f87966022eea7762f`: can_answer_from_single_sentence=yes (sent=S6)
  - `3dcfd60153822a6a8f6a516f161fc506`: can_answer_from_single_sentence=yes (sent=S4)
  - `9fcf7e509cc4e59026333ba469e22ec3`: can_answer_from_single_sentence=yes (sent=S5)

## Good Examples (5)

### [Medium] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** spanned -> providing -> taken
- **question:** What significant event occurred during the span of Deny Flight that led to increased tension between NATO and the UN?
- **gold_answer:** UN peacekeepers were taken as hostages in response to NATO bombing
- **solver_answer:** NATO bombing led to two peacekeepers being taken hostages.
- **judge_correct:** 1.0
- **composite:** 0.917

### [Easy] Death of Joy Gardner

- **doc_id:** `dd2a791aa826766cf0d05dc8102f5c8e`
- **path:** attention -> inquest
- **question:** What did not happen despite the public attention brought to the case?
- **gold_answer:** no coroner's inquest or public inquiry into the circumstances of Gardner's death has been held
- **solver_answer:** No coroner's inquest or public inquiry.
- **judge_correct:** 1.0
- **composite:** 0.883

### [Hard] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** uprising -> negotiate -> refused -> killed
- **question:** During the uprising, when the Russian commander tried to negotiate but was refused, what happened next?
- **gold_answer:** After a short hand-to-hand fight the Russian commander was killed
- **solver_answer:** Polish unit charged.
- **judge_correct:** 0.0
- **composite:** 0.75

### [Hard] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** uprising -> approached -> refused -> ordered
- **question:** When the uprising began and the Russians approached the Polish quarters, what did Rogaliński decide to do?
- **gold_answer:** ordered a charge of the Russians
- **solver_answer:** Refused to negotiate
- **judge_correct:** 0.0
- **composite:** 0.717

### [Easy] Death and state funeral of Raúl Alfonsín

- **doc_id:** `04a82d4eac379a98efcd87ebdba0b0ce`
- **path:** died -> funeral
- **question:** What event did Vice President Julio Cobos arrange after Raúl Alfonsín died?
- **gold_answer:** a state funeral at the Palace of the Argentine National Congress
- **solver_answer:** three days of national mourning and a funeral at the Palace of the Argentineos Congress
- **judge_correct:** 0.0
- **composite:** 0.633


## Bad Examples (5)

### [Medium] 2006 state of emergency in the Philippines

- **doc_id:** `84ce009a07b987d60a79d92bc4d45744`
- **path:** occurred -> revocation -> lifted
- **question:** What action did the government take on March 3, 2006, one week after proclaiming the state of emergency, and what immediate effects did it have on public activities and demonstrations?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Battle of Malacca (1641)

- **doc_id:** `3dcfd60153822a6a8f6a516f161fc506`
- **path:** capture -> casualties -> rallied
- **question:** What alliance was formed as a result of the Battle of Malacca's outcome, involving the Sultanate of Johor and the Dutch?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Easy] Myyrmanni bombing

- **doc_id:** `e253b7fd1109bd5f87966022eea7762f`
- **path:** took place -> released
- **question:** What happened to the victims after they were treated?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 0/2 events, need >= 1 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Battle of Ciołków

- **doc_id:** `a24058769038462f489b0091ebb24597`
- **path:** refused -> fight -> lost
- **question:** What happened to the Polish commander after he refused to negotiate with the Russians?
- **filter_pass:** False
- **filter_reason:** path_coverage=covers 1/3 events, need >= 2 [FAIL]
- **solver_answer:** 
- **judge_correct:** ?

### [Medium] Operation Deny Flight

- **doc_id:** `9fcf7e509cc4e59026333ba469e22ec3`
- **path:** began -> expanded -> bombed
- **question:** 
- **filter_pass:** False
- **filter_reason:** generation_error
- **solver_answer:** 
- **judge_correct:** ?


## Conclusion

- Selected: 29 (Easy 10, Medium 10, Hard 9)
- Generated: 28/29 (97%)
- Filter pass: 7/29 (24%)
- Solver correct: 2/7 (29%)