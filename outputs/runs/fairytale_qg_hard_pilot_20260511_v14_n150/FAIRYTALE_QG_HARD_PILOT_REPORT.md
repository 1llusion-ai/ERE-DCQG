# FairytaleQA Hard QG Pilot Report

Generated: 2026-05-11 21:21:02

## 1. Run Summary

| Field | Value |
|---|---|
| Methods | Direct, ICL, SelfRefine, Ours |
| Requested limit | 150 |
| Graph total | 250 |
| Graph valid | 219 |
| Selected candidates | 150 |
| Total generations | 600 |
| Target difficulty | Hard |

### Parse success by method

| Method | parse_ok | Total | Pct |
|---|---:|---:|---:|
| Direct | 134 | 150 | 89.3% |
| ICL | 133 | 150 | 88.7% |
| SelfRefine | 79 | 150 | 52.7% |
| Ours | 137 | 150 | 91.3% |

### 1b. Generation Robustness by Method

| Method | degenerate | repair_attempted | repair_success | quality_pass |
|---|---:|---:|---:|---:|
| Direct | 16 | 0 | 0 | 77 |
| ICL | 16 | 0 | 0 | 86 |
| SelfRefine | 0 | 0 | 0 | 51 |
| Ours | 45 | 81 | 8 | 104 |

## 2. Quality Pass by Method

| Method | quality_pass | strict_quality_pass | Total | Pct (loose) | Pct (strict) |
|---|---:|---:|---:|---:|---:|
| Direct | 77 | 11 | 150 | 51.3% | 7.3% |
| ICL | 86 | 10 | 150 | 57.3% | 6.7% |
| SelfRefine | 51 | 2 | 150 | 34.0% | 1.3% |
| Ours | 104 | 16 | 150 | 69.3% | 10.7% |

## 3. Blind Difficulty Distribution (quality-pass only)

| Method | Easy | Medium | Hard | JudgeError | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 10 | 58 | 9 | 0 | 77 |
| ICL | 6 | 67 | 13 | 0 | 86 |
| SelfRefine | 4 | 37 | 10 | 0 | 51 |
| Ours | 7 | 61 | 36 | 0 | 104 |

### 3b. Difficulty Judge Status by Method

| Method | judge_ok | judge_error | Total | Error rate |
|---|---:|---:|---:|---:|
| Direct | 77 | 0 | 77 | 0.0% |
| ICL | 86 | 0 | 86 | 0.0% |
| SelfRefine | 51 | 0 | 51 | 0.0% |
| Ours | 104 | 0 | 104 | 0.0% |

## 4. Hard Hit Rate by Method

Denominator: quality-pass AND difficulty_judge_status=ok

| Method | Hard hit | Wilson 95% CI |
|---|---|---|
| Direct | 9/77 (11.7%, 95%CI [6.3, 20.7%]) | |
| ICL | 13/86 (15.1%, 95%CI [9.1, 24.2%]) | |
| SelfRefine | 10/51 (19.6%, 95%CI [11.0, 32.5%]) | |
| Ours | 36/104 (34.6%, 95%CI [26.2, 44.2%]) | |

## 5. Evidence Dependency by Method (quality-pass, judge-ok only)

| Method | alone_sufficient=no | bridge_required=yes | removal in {ambiguous,unanswerable} | Total |
|---|---:|---:|---:|---:|
| Direct | 35 | 67 | 60 | 77 |
| ICL | 39 | 79 | 69 | 86 |
| SelfRefine | 19 | 45 | 36 | 51 |
| Ours | 55 | 94 | 78 | 104 |

### 5b. Target Evidence Coverage by Method (quality-pass, judge-ok only)

| Method | mean coverage | coverage>=0.67 | uses_all_target | Total |
|---|---:|---:|---:|---:|
| Direct | 0.383 | 0 | 0 | 77 |
| ICL | 0.385 | 1 | 0 | 86 |
| SelfRefine | 0.370 | 0 | 0 | 51 |
| Ours | 0.472 | 8 | 5 | 104 |

### 5c. Hard Realization Pass by Method (exact-id diagnostic)

Hard realization (legacy) = judge_ok AND num_judge_used>=3 AND uses_bridge in {yes,partial} AND coverage>=0.67 AND predicted=Hard

| Method | hard_realization_pass | quality-pass judge-ok | Rate |
|---|---:|---:|---:|
| Direct | 1 | 77 | 1.3% |
| ICL | 4 | 86 | 4.7% |
| SelfRefine | 3 | 51 | 5.9% |
| Ours | 16 | 104 | 15.4% |

### 5e. Hard Realization Pass v2 by Method

Denominator: quality-pass AND difficulty_judge_status=ok

hrp_v2 = predicted=Hard AND num_judge_used>=3 AND bridge_required=yes AND alone_sufficient=no AND semantic_evidence_match in {yes,partial}

| Method | hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 9/77 (11.7%, 95%CI [6.3, 20.7%]) | |
| ICL | 11/86 (12.8%, 95%CI [7.3, 21.5%]) | |
| SelfRefine | 10/51 (19.6%, 95%CI [11.0, 32.5%]) | |
| Ours | 31/104 (29.8%, 95%CI [21.9, 39.2%]) | |

### 5e2. Strict HRP-v2 by Method

strict_hrp_v2 = hard_realization_pass_v2=yes AND strict_quality_pass=true AND focus_match=yes

| Method | strict_hrp_v2 | Wilson 95% CI |
|---|---|---|
| Direct | 0/77 (0.0%, 95%CI [0.0, 4.8%]) | |
| ICL | 0/86 (0.0%, 95%CI [0.0, 4.3%]) | |
| SelfRefine | 0/51 (0.0%, 95%CI [0.0, 7.0%]) | |
| Ours | 4/104 (3.8%, 95%CI [1.5, 9.5%]) | |

### 5f. Semantic Evidence Match by Method (quality-pass, judge-ok)

| Method | yes | partial | no | judge_error | Total |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 71 | 6 | 0 | 77 |
| ICL | 0 | 83 | 3 | 0 | 86 |
| SelfRefine | 0 | 48 | 3 | 0 | 51 |
| Ours | 5 | 90 | 9 | 0 | 104 |

### 5d. Answer Focus Diagnostics (Ours)

#### Question focus distribution

| Focus | Count | Pct |
|---|---:|---:|
| bridge | 46 | 44.2% |
| state | 25 | 24.0% |
| motivation | 21 | 20.2% |
| outcome | 11 | 10.6% |
| count | 1 | 1.0% |

#### Focus match rate

- focus_match=yes: 69 / 104
- focus_match=no: 34 / 104
- focus_match=unknown: 1

#### Answer role -> question focus mapping

| answer_role | question_focus | count |
|---|---|---:|
| answer | bridge | 34 |
| answer | state | 25 |
| answer | motivation | 19 |
| answer | outcome | 11 |
| answer_bridge | bridge | 12 |
| answer_bridge | motivation | 2 |
| count_pattern | count | 1 |

#### Focus mismatch examples

**Mismatch 1:**
- Story: thomas-the-rhymer
- Question: What did Thomas do that caused the fairy queen to transform and demand seven years of service from him?
- Target answer: he asked for a kiss .
- answer_role=answer, question_focus=outcome

**Mismatch 2:**
- Story: lame-dog
- Question: Why did the youngest princess sit apart and weep during the wedding celebrations?
- Target answer: sad .
- answer_role=answer, question_focus=motivation

**Mismatch 3:**
- Story: the-fairies-of-merlin-crag
- Question: What did the countryman's action of lifting the peats from the other end of the moor potentially imply about his understanding of the previous incident?
- Target answer: he ruined her home .
- answer_role=answer, question_focus=bridge

#### Per question_focus metrics (all Ours)

| Focus | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 | focus_match=yes |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 63 | 46 | 17 | 15 | 2 | 33 |
| count | 2 | 1 | 1 | 1 | 1 | 1 |
| motivation | 31 | 21 | 5 | 4 | 0 | 19 |
| outcome | 20 | 11 | 6 | 5 | 0 | 6 |
| state | 34 | 25 | 7 | 6 | 1 | 21 |

#### Per answer_role metrics (all Ours)

| answer_role | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|
| answer | 129 | 89 | 30 | 26 | 3 |
| answer_bridge | 19 | 14 | 5 | 4 | 0 |
| count_pattern | 2 | 1 | 1 | 1 | 1 |

#### Per answer_node_type metrics (all Ours)

| node_type | N | quality_pass | Hard | hrp_v2 | strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|
| action | 42 | 33 | 10 | 9 | 2 |
| belief | 7 | 6 | 1 | 1 | 0 |
| consequence | 4 | 2 | 1 | 1 | 0 |
| count | 2 | 1 | 1 | 1 | 1 |
| description | 14 | 9 | 5 | 5 | 0 |
| emotion | 18 | 12 | 2 | 2 | 0 |
| goal | 2 | 2 | 2 | 1 | 0 |
| motivation | 11 | 7 | 1 | 1 | 0 |
| outcome | 16 | 9 | 5 | 4 | 0 |
| state | 34 | 23 | 8 | 6 | 1 |

### 5g. Unique Story Diversity and Cluster Diagnostic

#### Unique stories among predicted Hard (quality-pass, judge-ok)

| Method | unique stories | Hard count | stories |
|---|---:|---:|---|
| Direct | 9 | 9 | evil-one-kitta-grau, farquhar-macneill, gold-tree-and-silver-tree, master-girl, princess-glass-mountain, sheem-the-forsaken-boy, the-one-handed-girl, three-dogs, wunzh-the-father-of-indian-corn |
| ICL | 12 | 13 | evil-one-kitta-grau, farquhar-macneill, how-princess-pride-was-broken, murmur-goose-egg, princess-glass-mountain, story-of-princess-hase, the-black-bull-of-norroway, the-brown-bear-of-norway, the-escape-of-the-mouse, the-magic-bundle, the-one-handed-girl, wunzh-the-father-of-indian-corn |
| SelfRefine | 10 | 10 | faithful-and-unfaithful, master-girl, princess-glass-mountain, the-bones-of-djulung, the-crane-that-crossed-the-river, the-escape-of-the-mouse, the-fairies-of-merlin-crag, the-fairy-nurse, the-one-handed-girl, three-dogs |
| Ours | 29 | 36 | bokwewa-the-humpback, canonbie-dick-and-thomas-of-ercildoune, comrade, faithful-and-unfaithful, farquhar-macneill, farther-south-than-south-and-farther-north-than-north-and-in-great-hill-of-gold, habetrot-the-spinstress, little-lasse, murmur-goose-egg, sheem-the-forsaken-boy, soria-moria-castle, the-black-bull-of-norroway, the-brown-bear-of-norway, the-crane-that-crossed-the-river, the-elfin-knight, the-fairies-of-merlin-crag, the-fairy-nurse, the-fire-plume, the-little-spirit-or-boy-man, the-magic-bundle, the-one-handed-girl, the-rich-brother-and-the-poor-brother, the-seal-catcher-and-the-merman, the-three-crowns, thomas-the-rhymer, three-dogs, weendigoes-and-the-bone-dwarf, wunzh-the-father-of-indian-corn, youth-who-wanted-to-win-daughter-of-mother-in-corner |

#### Unique stories among hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 9 | 9 | evil-one-kitta-grau, farquhar-macneill, gold-tree-and-silver-tree, master-girl, princess-glass-mountain, sheem-the-forsaken-boy, the-one-handed-girl, three-dogs, wunzh-the-father-of-indian-corn |
| ICL | 10 | 11 | evil-one-kitta-grau, farquhar-macneill, princess-glass-mountain, story-of-princess-hase, the-black-bull-of-norroway, the-brown-bear-of-norway, the-escape-of-the-mouse, the-magic-bundle, the-one-handed-girl, wunzh-the-father-of-indian-corn |
| SelfRefine | 10 | 10 | faithful-and-unfaithful, master-girl, princess-glass-mountain, the-bones-of-djulung, the-crane-that-crossed-the-river, the-escape-of-the-mouse, the-fairies-of-merlin-crag, the-fairy-nurse, the-one-handed-girl, three-dogs |
| Ours | 25 | 31 | bokwewa-the-humpback, canonbie-dick-and-thomas-of-ercildoune, comrade, faithful-and-unfaithful, farquhar-macneill, farther-south-than-south-and-farther-north-than-north-and-in-great-hill-of-gold, habetrot-the-spinstress, little-lasse, murmur-goose-egg, sheem-the-forsaken-boy, soria-moria-castle, the-black-bull-of-norroway, the-brown-bear-of-norway, the-crane-that-crossed-the-river, the-elfin-knight, the-fairy-nurse, the-fire-plume, the-little-spirit-or-boy-man, the-magic-bundle, the-one-handed-girl, the-rich-brother-and-the-poor-brother, the-seal-catcher-and-the-merman, the-three-crowns, three-dogs, youth-who-wanted-to-win-daughter-of-mother-in-corner |

#### Unique stories among strict_hrp_v2 (quality-pass, judge-ok)

| Method | unique stories | strict_hrp_v2 count | stories |
|---|---:|---:|---|
| Direct | 0 | 0 |  |
| ICL | 0 | 0 |  |
| SelfRefine | 0 | 0 |  |
| Ours | 4 | 4 | faithful-and-unfaithful, the-one-handed-girl, three-dogs, youth-who-wanted-to-win-daughter-of-mother-in-corner |

#### Cluster diagnostic: three-dogs concentration

| Method | three-dogs in Hard | total Hard | three-dogs in hrp_v2 | total hrp_v2 | three-dogs in strict_hrp_v2 | total strict_hrp_v2 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 1 | 9 | 1 | 9 | 0 | 0 |
| ICL | 0 | 13 | 0 | 11 | 0 | 0 |
| SelfRefine | 1 | 10 | 1 | 10 | 0 | 0 |
| Ours | 1 | 36 | 1 | 31 | 1 | 4 |

## 6. Failure Reasons by Method

| Method | Failure reason | Count |
|---|---|---:|
| Direct | not answerable | 43 |
| Direct | wrong answer | 21 |
| Direct | not fluent | 7 |
| Direct | other | 2 |
| ICL | not answerable | 36 |
| ICL | wrong answer | 18 |
| ICL | not fluent | 8 |
| ICL | other | 2 |
| SelfRefine | not answerable | 81 |
| SelfRefine | wrong answer | 8 |
| SelfRefine | not fluent | 7 |
| SelfRefine | other | 3 |
| Ours | not answerable | 26 |
| Ours | not fluent | 10 |
| Ours | wrong answer | 6 |
| Ours | gen error: degenerate output | 2 |
| Ours | gen error: self-check failed: answer mism | 2 |

### 6b. Difficulty Judge Parse Failures

None.

## 7. Copy/Reference Diagnostics

| Method | Total | Copies source | Copy rate |
|---|---:|---:|---:|
| Direct | 150 | 0 | 0.0% |
| ICL | 150 | 0 | 0.0% |
| SelfRefine | 150 | 0 | 0.0% |
| Ours | 150 | 0 | 0.0% |

## 8. Examples

### Best Ours examples (quality-pass, predicted Hard)

**Example 1:**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.333, hard_realization=no, hrp_v2=yes
- Focus: answer_role=count_pattern, question_focus=count, focus_match=yes
- Semantic match: partial - The judge-identified sentences [17, 31, 32] cover the third exchange but miss the first and second exchanges which are covered by the target evidence sentences [4, 16, 31].

**Example 2:**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as her son continued to skip and dance all day, leading to him wanting more food and wearing out his clothes quickly?
- Target answer: unhappy .
- Quality: answerable=yes, asks_expected=yes, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.667, hard_realization=yes, hrp_v2=yes
- Focus: answer_role=answer, question_focus=state, focus_match=yes
- Semantic match: partial - The judge-identified sentences [3, 5, 6] cover the mother's worsening situation and the son's behavior leading to his clothes wearing out, but they miss the direct mention of the son's dancing and skipping behavior from sentence [2], which is part of the target evidence.

**Example 3:**
- Story: little-lasse
- Question: What was the main reason Lasse was frightened and began to cry as the boat drifted out to sea?
- Target answer: he did not have oars to row with .
- Quality: answerable=yes, asks_expected=partial, leakage=no
- Difficulty: predicted=Hard, alone_sufficient=no, bridge_required=yes
- Coverage: 0.500, hard_realization=no, hrp_v2=yes
- Focus: answer_role=answer_bridge, question_focus=motivation, focus_match=yes
- Semantic match: partial - The judge-identified sentences cover Lasse's situation of drifting out to sea and his reaction of crying, but they miss the explanation about the lack of oars, which is a key part of the reasoning chain for why he was frightened.

### Hard realization pass v2 examples

**HRP-v2 Example 1 (Direct):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences cover the final exchange but miss the earlier exchanges that establish the pattern, making the reasoning chain less complete.

**HRP-v2 Example 2 (SelfRefine):**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences cover the final exchange but miss the earlier exchanges that establish the pattern, which are crucial for the full reasoning chain.

**HRP-v2 Example 3 (Ours):**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences [17, 31, 32] cover the third exchange but miss the first and second exchanges which are covered by the target evidence sentences [4, 16, 31].

**HRP-v2 Example 4 (Ours):**
- Story: youth-who-wanted-to-win-daughter-of-mother-in-corner
- Question: How did the mother feel as her son continued to skip and dance all day, leading to him wanting more food and wearing out his clothes quickly?
- Target answer: unhappy .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences [3, 5, 6] cover the mother's worsening situation and the son's behavior leading to his clothes wearing out, but they miss the direct mention of the son's dancing and skipping behavior from sentence [2], which is part of the target evidence.

**HRP-v2 Example 5 (Ours):**
- Story: little-lasse
- Question: What was the main reason Lasse was frightened and began to cry as the boat drifted out to sea?
- Target answer: he did not have oars to row with .
- Predicted: Hard, num_used=3, bridge=yes, alone=no
- Semantic match: partial - The judge-identified sentences cover Lasse's situation of drifting out to sea and his reaction of crying, but they miss the explanation about the lack of oars, which is a key part of the reasoning chain for why he was frightened.

### Ours focus-match examples (quality-pass, focus_match=yes)

**Focus Example 1:**
- Story: three-dogs
- Question: How many times did the old man offer to exchange his dog for one of the youth's pigs?
- Target answer: three times .
- answer_role=count_pattern, question_focus=count, node_type=count

**Focus Example 2:**
- Story: three-dogs
- Question: What motivated the princes to plot against the youth and ultimately lead to his death?
- Target answer: they were jealous of the youth .
- answer_role=answer, question_focus=bridge, node_type=action

**Focus Example 3:**
- Story: thomas-the-rhymer
- Question: How did Thomas come to feel after the Queen of Fairies changed and mocked him?
- Target answer: scared .
- answer_role=answer, question_focus=state, node_type=state

### Best baseline examples (quality-pass)

**Direct Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

**ICL Example:**
- Story: three-dogs
- Question: Why did the princes decide to attack their comrade who had saved the king's daughters?
- Target answer: they were jealous of the youth .
- Predicted difficulty: Easy

**SelfRefine Example:**
- Story: three-dogs
- Question: Why did the young swineherd agree to exchange his pigs for dogs three times?
- Target answer: three times .
- Predicted difficulty: Hard

### Ours failure cases (grouped)

| Failure category | Count |
|---|---:|
| degenerate / parse failure | 17 |
| not answerable | 13 |
| not fluent | 9 |
| answer mismatch | 5 |
| other | 2 |

#### Ours failure examples

**Failure 1:**
- Story: three-dogs
- Question: What deeper reason drove the giant to make the youth send his dogs to fetch the 
- Reason: The question asks about the giant's deeper reason, but the story does not explicitly state that the troll will attack the youth. The giant's motive is to separate the youth from his protective dogs, but there's no mention of a troll attack.

**Failure 2:**
- Story: thomas-the-rhymer
- Question: What What he needed to do for the price of kissing her .
- Reason: The question is not grammatically correct and natural-sounding due to the repetition of 'What' and the lack of a clear verb or structure.

**Failure 3:**
- Story: the-one-handed-girl
- Question: (empty)
- Raw prefix: `{"{" 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 		
 
 
 
 
 
 		
 
 :"her unw ways arm ."
 }`
- Reason: empty question

### Baseline failure cases

**Failure 1 (ICL):**
- Story: three-dogs
- Question: Why why why did the youth do with the dog and the pig in the woods?
- Reason: The question is not grammatically correct and does not sound natural. It should be rephrased for clarity and grammatical correctness.

**Failure 2 (Direct):**
- Story: three-dogs
- Question: Why did the giant make the youth send his dogs to fetch the wine from the mounta
- Reason: The story does not mention a troll attacking the youth, nor does it provide a reason related to a troll attacking him. The question is based on an assumption not present in the text.

**Failure 3 (ICL):**
- Story: three-dogs
- Question: Why did the giant hand the youth a great tankard, and why did the dog go unwilli
- Reason: The question can be partially answered from the story, but it does not lead to the expected answer 'the troll will attack him.' The story does not mention the troll attacking the youth, and the question does not contain the exact phrase 'the troll will attack him.'

## 8b. Pairwise Difference Table (Ours - Baseline)

| Metric | Ours | Direct | ICL | SelfRefine | Ours-Direct | Ours-ICL | Ours-SelfRefine |
|---|---|---:|---:|---:|---|---|---|
| quality_pass | 69.3% (104/150) | 51.3% (77/150) | 57.3% (86/150) | 34.0% (51/150) | +18.0pp | +12.0pp | +35.3pp |
| Hard hit | 34.6% (36/104) | 11.7% (9/77) | 15.1% (13/86) | 19.6% (10/51) | +22.9pp | +19.5pp | +15.0pp |
| HRP-v2 | 29.8% (31/104) | 11.7% (9/77) | 12.8% (11/86) | 19.6% (10/51) | +18.1pp | +17.0pp | +10.2pp |
| unique HRP-v2 stories | 25 | 9 | 10 | 10 | +16 | +15 | +15 |

## 9. Success Criteria

| Criterion | Status | Value |
|---|---|---|
| Ours quality_pass >= 65% | PASS | 69.3% (104/150 (69.3%, 95%CI [61.5, 76.2%])) |
| Ours predicted Hard >= 25% | PASS | 34.6% (36/104 (34.6%, 95%CI [26.2, 44.2%])) |
| Ours eval_hrp_v2 >= 25% (quality-pass, judge-ok) | PASS | 29.8% (31/104 (29.8%, 95%CI [21.9, 39.2%])) |
| Ours strict_hrp_v2 >= 10% | FAIL | 3.8% (4/104 (3.8%, 95%CI [1.5, 9.5%])) |
| Ours unique HRP-v2 stories > each baseline | PASS | Ours=25, Direct=9, ICL=10, SelfRefine=10 |
| Ours Hard hit >= Direct/ICL/SelfRefine | PASS | Ours=0.35, Direct=0.12, ICL=0.15, SelfRefine=0.20 |

**Overall: SOME CRITERIA FAILED**
