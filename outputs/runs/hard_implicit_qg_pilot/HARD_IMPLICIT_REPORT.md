# Hard Implicit Chain Pilot Report

**Input:** `outputs/runs/path_filter_strict_pilot/paths.filtered.strict.jsonl` (9 Hard strict paths)
**Method:** PathQG-HardAware with implicit_chain prompt
**Filter:** v3 + hard_implicitness_check

## Summary

| Metric | Count |
|--------|------:|
| Generated | 9/9 |
| Filter Pass | 2/9 |
| Implicit Chain Pass | 8/9 |
| Implicit Chain Fail | 1/9 |
| Independent Judged | 2/2 |
| Pred Hard | 0/2 |
| Pred Medium | 1/2 |
| Pred Easy | 1/2 |
| PathDep Strong | 2/2 |
| PathDep Strong+Partial | 2/2 |
| Answerable | 2/2 |
| FinalConsistent | 2/2 |

## Old vs New Comparison (Same Hard Paths)

| Metric | Old Hard Prompt | New Implicit Hard |
|--------|---:|---:|
| Generated | 9 | 9 |
| Filter Pass | 6 | 2 |
| Judged | 6 | 2 |
| Independent Pred Hard | 0 | 0 |
| Independent Pred Medium | 2 | 1 |
| Independent Pred Easy | 4 | 1 |
| PathDep Strong | 5 | 2 |
| Answerable | 6 | 2 |
| FinalConsistent | 5 | 2 |

## Same-Path Comparison

| # | Doc ID | Old Question | Old Pred | New Question | New Pred | New PathDep |
|--:|--------|-------------|----------|-------------|----------|------------|
| 1 | 3dcfd6015382 | After the Dutch began their campaign to destroy Portuguese p... | Medium | What were the consequences of the Dutch and Johor forces' co... | - | - |
| 2 | 81c576926e0c | What event occurred after the Russian fleet destroyed a Turk... | Medium | What restriction did Russia face after the Ottoman forces st... | - | - |
| 3 | 81c576926e0c | - | - | What event followed the Russian fleet destroying a Turkish a... | - | - |
| 4 | 9fcf7e509cc4 | After Operation Deny Flight began and provided close air sup... | Easy | What significant changes did NATO undergo during Operation D... | - | - |
| 5 | a24058769038 | During the uprising, when the Russian commander tried to neg... | Easy | What impact did the Russian commander face after refusing to... | Easy | strong |
| 6 | a24058769038 | During the uprising, when the Russian force approached the P... | Easy | What outcome resulted from the Russian force approaching the... | Medium | strong |
| 7 | db50381e7d1d | After the Allied corps surrounded Bayonne and then pushed So... | Easy | What significant clash marked the end of the Allied forces' ... | - | - |
| 8 | e253b7fd1109 | - | - | What outcome concluded the sequence triggered by the crowded... | - | - |
| 9 | f46091471f38 | - | - | What impact did the warnings and calls have on the hotel's s... | - | - |

## Filter Failure Analysis

| Reason | Count |
|--------|------:|
| answer_consistency=no: extracted from text: asks=no ans=no cons=no | 2 |
| path_coverage=covers 0 prior events, need >= 2 [FAIL] | 2 |
| hard_implicit=3 prior triggers explicitly in question (max 2 allowed) | 1 |
| answer_consistency=no: extracted from text: asks=no ans=yes cons=no | 1 |
| path_coverage=covers 1 prior events, need >= 2 [FAIL] | 1 |

## Examples

### Good Examples (Pred Hard/Medium + PathDep Strong)

**[Hard -> Medium] doc_id=a24058769038**

- Question: What outcome resulted from the Russian force approaching the Polish quarters and the subsequent refusal to negotiate?
- Answer: ordered a charge of the Russians
- Path: uprising/Change_of_leadership -> approached/Arriving -> refused/Agree_or_refuse_to_act -> ordered/Arranging
- Steps: 2 | PathDep: strong | Answerable: yes
- Reason: The solver needs to connect the Russian force approaching the Polish quarters and the refusal to negotiate to understand that this led to the order to charge, requiring two pieces of information.

### Problem Examples

**[filter_fail] doc_id=81c576926e0c**

- Question: What restriction did Russia face after the Ottoman forces stopped the Russian advance and the British and French rushed forces to Gallipoli?
- Filter reason: answer_consistency=no: extracted from text: asks=no ans=no cons=no

**[filter_fail] doc_id=e253b7fd1109**

- Question: What outcome concluded the sequence triggered by the crowded shopping center taking place and being investigated?
- Filter reason: hard_implicit=3 prior triggers explicitly in question (max 2 allowed)

**[filter_fail] doc_id=f46091471f38**

- Question: What impact did the warnings and calls have on the hotel's safety on the day of the bombing?
- Filter reason: answer_consistency=no: extracted from text: asks=no ans=no cons=no

**[pred=Easy] doc_id=a24058769038**

- Question: What impact did the Russian commander face after refusing to negotiate with the Polish forces?
- Filter reason: all checks passed
- Judge reason: The question directly asks about the consequence of the Russian commander's refusal to negotiate, which is explicitly stated in one sentence of the context.

**[filter_fail] doc_id=81c576926e0c**

- Question: What event followed the Russian fleet destroying a Turkish attempt to reinforce Kars and the Ottoman forces being halted at Silistra?
- Filter reason: answer_consistency=no: extracted from text: asks=no ans=yes cons=no; path_coverage=covers 1 prior events, need >= 2 [FAIL]


## Interpretation

- **Failure**: No questions judged as Hard. See failure analysis above.
- Path dependency strong: 2/2
- Implicit chain pass rate: 8/9
- Compare with old Hard prompt: 0/6 judged Hard, 6/6 judged Easy/Medium