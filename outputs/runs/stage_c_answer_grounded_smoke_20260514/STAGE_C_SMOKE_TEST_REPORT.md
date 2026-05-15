# Stage C Smoke Test: Answer-Grounded Evidence → Graph → QG

Generated: 2026-05-14 02:25:07

Total candidates: 9

## 1. Pipeline Summary

| Stage | Pass | Total | Pct |
|---|---:|---:|---:|
| Evidence Plan parse | 9 | 9 | 100.0% |
| Evidence Plan valid | 9 | 9 | 100.0% |
| Graph valid | 8 | 9 | 88.9% |
| QG question generated | 8 | 9 | 88.9% |
| QG parse ok | 7 | 9 | 77.8% |
| QG self-check pass | 2 | 9 | 22.2% |
| **End-to-end pass** | **2** | **9** | **22.2%** |

## 2. By Difficulty

| Difficulty | Evidence Parse | Evidence Valid | Graph Valid | QG Question | QG Parse | QG Self-Check |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 0/3 |
| Medium | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| Hard | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 | 1/3 |

## 3. Per-Candidate Details

### 1. master-girl [Easy]

- **Original question:** why did the giant think he was a kind master ?
- **Target answer:** the giant gave seemingly simple tasks .
- **Evidence plan:** parse=OK, valid=yes, num_req=1
  - Required: [1]
  - Bridge: [2]
  - Reasoning: explicit_lookup
- **Graph:** valid=yes, nodes=1, edges=0
- **Generated question:** Who did the giant do simple tasks? 
- **QG status:** parse=OK, self_check=FAIL, attempts=4
  - Error: self-check failed: answer mismatch; graph_policy non-compliant (answer_only)
  - Graph policy: answer_only
  - Policy reason: Easy: using answer node N1 only

### 2. thomas-the-rhymer [Easy]

- **Original question:** why did thomas and the queen of fairies need to take the third road ?
- **Target answer:** the road lead to elf - land .
- **Evidence plan:** parse=OK, valid=yes, num_req=1
  - Required: [3]
  - Bridge: []
  - Reasoning: explicit_lookup
- **Graph:** valid=yes, nodes=1, edges=0
- **Generated question:** The the road that runs to elf - land .
- **QG status:** parse=OK, self_check=FAIL, attempts=4
  - Error: no question mark
  - Graph policy: answer_only
  - Policy reason: Easy: using answer node N1 only

### 3. the-fox-and-the-wolf [Easy]

- **Original question:** what did the fox and the wolf do after the wolf agreed with the fox ?
- **Target answer:** dug a hole , buried the keg , and went off hunting as usual .
- **Evidence plan:** parse=OK, valid=yes, num_req=1
  - Required: [7]
  - Bridge: [6]
  - Reasoning: explicit_lookup
- **Graph:** valid=yes, nodes=1, edges=0
- **Generated question:** What did the two animals do after digging a hole and burying the keg?
- **QG status:** parse=FAIL, self_check=FAIL, attempts=4
  - Error: degenerate output
  - Graph policy: answer_only
  - Policy reason: Easy: using answer node N1 only

### 4. three-dogs [Medium]

- **Original question:** how did the boy feel about the great dog ?
- **Target answer:** happy .
- **Evidence plan:** parse=OK, valid=yes, num_req=2
  - Required: [10, 11]
  - Bridge: []
  - Reasoning: local_inference
- **Graph:** valid=yes, nodes=2, edges=1
- **Generated question:** What made the youth feel that fortune had indeed favored him?
- **QG status:** parse=OK, self_check=FAIL, attempts=4
  - Error: degenerate output
  - Graph policy: two_node_relation
  - Policy reason: Medium: answer N2 + N1 via enables

### 5. how-boots-befooled-king [Medium]

- **Original question:** why were two or three men dawdling around the house every day ?
- **Target answer:** they wanted to marry the king 's daughter .
- **Evidence plan:** parse=OK, valid=yes, num_req=2
  - Required: [2, 3]
  - Bridge: [3]
  - Reasoning: local_inference
  - Contradictions: ['answer_sentence_id=2 in anchor, removed']
- **Graph:** valid=yes, nodes=2, edges=1
- **Generated question:** Why did so many lads keep coming to the king's house every day?
- **QG status:** parse=OK, self_check=PASS, attempts=3
  - Graph policy: two_node_relation
  - Policy reason: Medium fallback: answer N1 + N2 (no directed edge into answer)

### 6. youth-who-wanted-to-win-daughter-of-mother-in-corner [Medium]

- **Original question:** why did the youth have no great appetite for the food ?
- **Target answer:** it was rat food .
- **Evidence plan:** parse=OK, valid=yes, num_req=2
  - Required: [7, 9]
  - Bridge: []
  - Reasoning: local_inference
- **Graph:** valid=yes, nodes=2, edges=1
- **Generated question:** What caused the youth to have no great appetite for the food?
- **QG status:** parse=OK, self_check=FAIL, attempts=4
  - Error: degenerate output
  - Graph policy: two_node_relation
  - Policy reason: Medium: answer N2 + N1 via results_in

### 7. three-dogs [Hard]

- **Original question:** how many times did the boy make an exchange with the old gray-beard ?
- **Target answer:** three times .
- **Evidence plan:** parse=OK, valid=yes, num_req=3
  - Required: [6, 18, 32]
  - Bridge: []
  - Reasoning: summary_synthesis
  - Contradictions: ['Hard with bridge_required=no (expected yes)']
- **Graph:** valid=yes, nodes=3, edges=2
- **Generated question:** How many times did the young swine-herd exchange his pigs for dogs due to his admiration of the dogs?
- **QG status:** parse=OK, self_check=PASS, attempts=4
  - Graph policy: multi_node_chain
  - Policy reason: Hard: directed path N1 → N2 → N3 [pure_temporal_chain]

### 8. princess-glass-mountain [Hard]

- **Original question:** how many times did the princess's suitors attempt to go up glass mountain ?
- **Target answer:** three times .
- **Evidence plan:** parse=OK, valid=yes, num_req=3
  - Required: [7, 14, 17]
  - Bridge: [17]
  - Reasoning: summary_synthesis
  - Contradictions: ['answer_sentence_id=14 in anchor, removed']
- **Graph:** valid=no, nodes=3, edges=2
  - Reason: no answer-role node found
- **Generated question:** 
- **QG status:** parse=FAIL, self_check=FAIL, attempts=?
  - Graph policy: ?
  - Policy reason: ?

### 9. the-three-crowns [Hard]

- **Original question:** how did the men decide who would go down in the well ?
- **Target answer:** they chose the eldest prince .
- **Evidence plan:** parse=OK, valid=yes, num_req=3
  - Required: [6, 7, 8]
  - Bridge: [6]
  - Reasoning: motivation_chain
- **Graph:** valid=yes, nodes=3, edges=2
- **Generated question:** What motivated the group to ultimately let the eldest prince down into the well after the second daughter's sweetheart c
- **QG status:** parse=OK, self_check=FAIL, attempts=4
  - Error: self-check failed: answer mismatch
  - Graph policy: multi_node_chain
  - Policy reason: Hard: directed path N1 → N2 → N3
