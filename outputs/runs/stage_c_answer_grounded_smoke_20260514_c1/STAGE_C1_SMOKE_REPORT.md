# Stage C.1 Smoke Test: Answer-Grounded Pipeline with Repairs

Generated: 2026-05-14 10:31:10

Total candidates: 30
  Easy: 10
  Medium: 10
  Hard: 10

## 1. Pipeline Summary

| Stage | Pass | Total | Pct |
|---|---:|---:|---:|
| Evidence parse | 30 | 30 | 100.0% |
| Evidence valid | 30 | 30 | 100.0% |
| Graph valid | 30 | 30 | 100.0% |
| Graph role repair | 15 | 30 | 50.0% |
| QG question generated | 28 | 30 | 93.3% |
| QG parse OK | 26 | 30 | 86.7% |
| QG self-check pass | 9 | 30 | 30.0% |
| Original Q leakage | 0 | 30 | 0.0% |
| **End-to-end pass** | **9** | **30** | **30.0%** |

## 2. By Difficulty

| Difficulty | N | Ev Parse | Ev Valid | Graph Valid | Graph Repair | QG Gen | QG Parse | QG Self-Check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 10 | 10 | 10 | 10 | 1 | 9 | 8 | 4 |
| Medium | 10 | 10 | 10 | 10 | 7 | 10 | 10 | 4 |
| Hard | 10 | 10 | 10 | 10 | 7 | 9 | 8 | 1 |

## 3. Easy Diagnostics

| Metric | Count | Pct |
|---|---:|---:|
| Easy malformed | 1 | 10.0% |
| Easy banned frames | 3 | 30.0% |
| Easy parse OK | 8 | 80.0% |
| Easy self-check pass | 4 | 40.0% |

### Banned Frame Breakdown

| Frame | Count |
|---|---:|
| when (clause) | 2 |
| how did | 1 |
| after (clause) | 1 |

## 4. Hard Diagnostics

| Metric | Count | Pct |
|---|---:|---:|
| Hard self-check pass | 1 | 10.0% |
| Hard bridge_required!=yes | 0 | 0.0% |

### Hard Reasoning Operations

| Operation | Count |
|---|---:|
| causal_chain | 5 |
| motivation_chain | 4 |
| summary_synthesis | 1 |

## 5. Graph Role Repair Details

Total repairs: 15

- **a-lost-paradise** [Easy]: overrode role=context to answer for node with sentence_id=14
- **a-legend-of-knockmany** [Medium]: overrode role=context to answer for node with sentence_id=10
- **a-lost-paradise** [Medium]: overrode role=context to answer for node with sentence_id=3
- **bokwewa-the-humpback** [Medium]: overrode role=context to answer for node with sentence_id=1
- **canonbie-dick-and-thomas-of-ercildoune** [Medium]: overrode role=context to answer for node with sentence_id=3
- **child-of-mary** [Medium]: overrode role=context to answer for node with sentence_id=7
- **comrade** [Medium]: overrode role=context to answer for node with sentence_id=6
- **cuchulain-of-muirthemne** [Medium]: overrode role=context to answer for node with sentence_id=5
- **Snow-man** [Hard]: overrode role=anchor to answer for node with sentence_id=13
- **a-legend-of-knockmany** [Hard]: overrode role=context to answer for node with sentence_id=7
- **a-lost-paradise** [Hard]: overrode role=context to answer for node with sentence_id=4
- **bokwewa-the-humpback** [Hard]: overrode role=anchor to answer for node with sentence_id=3
- **brave-tin-soldier** [Hard]: overrode role=anchor to answer for node with sentence_id=0
- **comrade** [Hard]: overrode role=context to answer for node with sentence_id=2
- **cuchulain-of-muirthemne** [Hard]: overrode role=anchor to answer for node with sentence_id=3

## 6. QG Self-Check Details

### Easy

Pass: 4/10, Fail: 5/10

**Passing:**

- Snow-man: "What the young girl and a young man .?"
- bokwewa-the-humpback: "How only . grieving in his heart? "
- brother-sister: " What did the brother mother do to them? "
- canonbie-dick-and-thomas-of-ercildoune: "What kind of gold did the stranger give Canonbie Dick?"

**Failing:**

- a-legend-of-knockmany: "Who heard three whistles ?" | reason: 
- a-lost-paradise: "What Did the woman feel that they they" | reason: 
- child-of-mary: "How did the foster-mother feel when she saw the moon had slipped out?" | reason: 
- comrade: "What would the youth have after agreeing to travel with the man?" | reason: 
- cuchulain-of-muirthemne: "What time of day was it when Conchubar spoke to his people?" | reason: 

### Medium

Pass: 4/10, Fail: 6/10

**Passing:**

- Snow-man: "Why was the snow man motivated to slide along yonder on the ice?"
- a-legend-of-knockmany: "What did Far Rua intend to do to Finn?"
- a-lost-paradise: "What happened to the charcoal-burner and his wife that made their nights more and more frequent with"
- brave-tin-soldier: "What happened to the fish after it swallowed the tin soldier?"

**Failing:**

- bokwewa-the-humpback: "Why did Bokwewa point to Kwasynd?" | reason: 
- brother-sister: "What what motivated the old woman to hide her on eye ?" | reason: 
- canonbie-dick-and-thomas-of-ercildoune: "Why was Canonbie Dick startled by the approach of the venerable man?" | reason: 
- child-of-mary: "Why did the man ask his wife if she gave him a flat 'no' when he asked about letting the lady take t" | reason: needs only 1 sentence
- comrade: "Why did the youth feel surprised when he saw the corpse in the middle of the ice block?" | reason: answer mismatch
- cuchulain-of-muirthemne: "He wanted to get money out of Con con ." | reason: 

### Hard

Pass: 1/10, Fail: 8/10

**Passing:**

- comrade: "What led the youth to lose the shears after the princess's playful behavior?"

**Failing:**

- Snow-man: "What led the yard-dog to ultimately have to bite the master's son's leg?" | reason: answer mismatch
- a-legend-of-knockmany: "What motivated Finn to put his 'honest face into his own door' given that his wife was leading a lon" | reason: answer mismatch; focus mismatch
- a-lost-paradise: "What did the charcoal-b burner and his wife ultimately feel grateful after? ." | reason: 
- bokwewa-the-humpback: "What led Bokwewa to attempt to restore the woman to life after Kwasynd found her on the scaffold?" | reason: 
- brave-tin-soldier: "What did the little dancer do to ultimately lead the tin soldier into a tin heart? ." | reason: 
- brother-sister: "What motivated the brother to drink from the third brook despite his sister's warnings?" | reason: answer mismatch
- child-of-mary: "What motivated the queen to weep and plead when the foster-mother took the youngest child?" | reason: 
- cuchulain-of-muirthemne: "Why he love did ness . ultimately . motivated by . .?" | reason: 

## 7. Per-Candidate Table

| # | Story | Diff | Ev OK | Graph OK | Repair | QG | Parse | SC | Question |
|---|---|---|---|---|---|---|---|---|
| 1 | Snow-man | Easy | Y | Y | - | Y | Y | Y | What the young girl and a young man .? |
| 2 | a-legend-of-knockman | Easy | Y | Y | - | Y | Y | N | Who heard three whistles ? |
| 3 | a-lost-paradise | Easy | Y | Y | Y | Y | Y | N | What Did the woman feel that they they |
| 4 | bokwewa-the-humpback | Easy | Y | Y | - | Y | Y | Y | How only . grieving in his heart?  |
| 5 | brave-tin-soldier | Easy | Y | Y | - | N | N | N |  |
| 6 | brother-sister | Easy | Y | Y | - | Y | Y | Y |  What did the brother mother do to them?  |
| 7 | canonbie-dick-and-th | Easy | Y | Y | - | Y | Y | Y | What kind of gold did the stranger give Canonbie Dick? |
| 8 | child-of-mary | Easy | Y | Y | - | Y | Y | N | How did the foster-mother feel when she saw the moon had sli |
| 9 | comrade | Easy | Y | Y | - | Y | N | N | What would the youth have after agreeing to travel with the  |
| 10 | cuchulain-of-muirthe | Easy | Y | Y | - | Y | Y | N | What time of day was it when Conchubar spoke to his people? |
| 11 | Snow-man | Medium | Y | Y | - | Y | Y | Y | Why was the snow man motivated to slide along yonder on the  |
| 12 | a-legend-of-knockman | Medium | Y | Y | Y | Y | Y | Y | What did Far Rua intend to do to Finn? |
| 13 | a-lost-paradise | Medium | Y | Y | Y | Y | Y | Y | What happened to the charcoal-burner and his wife that made  |
| 14 | bokwewa-the-humpback | Medium | Y | Y | Y | Y | Y | N | Why did Bokwewa point to Kwasynd? |
| 15 | brave-tin-soldier | Medium | Y | Y | - | Y | Y | Y | What happened to the fish after it swallowed the tin soldier |
| 16 | brother-sister | Medium | Y | Y | - | Y | Y | N | What what motivated the old woman to hide her on eye ? |
| 17 | canonbie-dick-and-th | Medium | Y | Y | Y | Y | Y | N | Why was Canonbie Dick startled by the approach of the venera |
| 18 | child-of-mary | Medium | Y | Y | Y | Y | Y | N | Why did the man ask his wife if she gave him a flat 'no' whe |
| 19 | comrade | Medium | Y | Y | Y | Y | Y | N | Why did the youth feel surprised when he saw the corpse in t |
| 20 | cuchulain-of-muirthe | Medium | Y | Y | Y | Y | Y | N | He wanted to get money out of Con con . |
| 21 | Snow-man | Hard | Y | Y | Y | Y | Y | N | What led the yard-dog to ultimately have to bite the master' |
| 22 | a-legend-of-knockman | Hard | Y | Y | Y | Y | Y | N | What motivated Finn to put his 'honest face into his own doo |
| 23 | a-lost-paradise | Hard | Y | Y | Y | Y | Y | N | What did the charcoal-b burner and his wife ultimately feel  |
| 24 | bokwewa-the-humpback | Hard | Y | Y | Y | Y | N | N | What led Bokwewa to attempt to restore the woman to life aft |
| 25 | brave-tin-soldier | Hard | Y | Y | Y | Y | Y | N | What did the little dancer do to ultimately lead the tin sol |
| 26 | brother-sister | Hard | Y | Y | - | Y | Y | N | What motivated the brother to drink from the third brook des |
| 27 | canonbie-dick-and-th | Hard | Y | Y | - | N | N | N |  |
| 28 | child-of-mary | Hard | Y | Y | - | Y | Y | N | What motivated the queen to weep and plead when the foster-m |
| 29 | comrade | Hard | Y | Y | Y | Y | Y | Y | What led the youth to lose the shears after the princess's p |
| 30 | cuchulain-of-muirthe | Hard | Y | Y | Y | Y | Y | N | Why he love did ness . ultimately . motivated by . .? |

## 8. Success Criteria

| Criterion | Actual | Target | Status |
|---|---:|---:|---|
| Evidence valid >= 95% | 100.0% | >=95% | **PASS** |
| Graph valid >= 90% | 100.0% | >=90% | **PASS** |
| QG parse OK >= 85% | 86.7% | >=85% | **PASS** |
| QG self-check >= 50% | 30.0% | >=50% | **FAIL** |
| Easy self-check >= 40% | 40.0% | >=40% | **PASS** |
| Medium self-check >= 50% | 40.0% | >=50% | **FAIL** |
| Hard self-check >= 50% | 10.0% | >=50% | **FAIL** |
| Original Q leakage = 0 | 0 | 0 | **PASS** |