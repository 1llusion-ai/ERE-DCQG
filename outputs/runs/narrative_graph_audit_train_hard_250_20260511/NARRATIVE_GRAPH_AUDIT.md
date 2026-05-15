# Narrative Evidence Graph Audit Report

Generated: 2026-05-11 18:26:28

## 1. Input Summary

| Field | Value |
|---|---|
| Input file | outputs/runs/fairytale_evidence_audit_train_implicit_2166_20260511/candidates.jsonl |
| Total Hard candidates available | 337 |
| Sampled for extraction | 250 |
| Model | Qwen/Qwen2.5-32B-Instruct |

### Distribution by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 78 |
| disambiguation | 17 |
| motivation_bridge | 77 |
| summary_synthesis | 78 |

### Distribution by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 77 |
| character_state | 17 |
| disambiguation | 16 |
| motivation | 64 |
| summary_inference | 76 |

### Distribution by attribute

| Attribute | Count |
|---|---:|
| action | 27 |
| causal relationship | 137 |
| character | 2 |
| feeling | 49 |
| outcome resolution | 14 |
| prediction | 21 |

## 2. Graph Extraction Success

| Metric | Count | Pct |
|---|---:|---:|
| parse_ok | 249 | 99.6% |
| graph_valid | 219 | 87.6% |
| graph_invalid | 31 | 12.4% |

### Validation fail reasons

| Reason | Count |
|---|---:|
| no answer-role node found | 17 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); no answer-role node found; e... | 2 |
| bridge sentences [5] have no bridge-role node | 2 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); bridge sentences [15] have n... | 1 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); bridge sentences [4, 5, 6, 7... | 1 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); bridge sentences [5, 6] have... | 1 |
| no connected path to answer node | 1 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); bridge sentences [4] have no... | 1 |
| bridge sentences [47] have no bridge-role node | 1 |
| bridge sentences [19] have no bridge-role node | 1 |

## 3. Graph Structure Statistics

| Metric | Value |
|---|---:|
| Avg nodes | 4.1 |
| Avg edges | 3.0 |
| Min nodes | 0 |
| Max nodes | 38 |
| Min edges | 0 |
| Max edges | 18 |

### Node type distribution

| Type | Count | Pct |
|---|---:|---:|
| action | 259 | 25.3% |
| state | 251 | 24.5% |
| description | 197 | 19.2% |
| emotion | 82 | 8.0% |
| belief | 74 | 7.2% |
| motivation | 60 | 5.9% |
| outcome | 53 | 5.2% |
| goal | 17 | 1.7% |
| consequence | 11 | 1.1% |
| problem | 9 | 0.9% |
| attempt | 8 | 0.8% |
| resolution | 3 | 0.3% |

### Edge relation distribution

| Relation | Count | Pct |
|---|---:|---:|
| temporal_before | 193 | 26.1% |
| causes | 186 | 25.1% |
| results_in | 160 | 21.6% |
| motivates | 84 | 11.4% |
| explains | 41 | 5.5% |
| supports_inference | 39 | 5.3% |
| enables | 27 | 3.6% |
| contrasts_with | 10 | 1.4% |

### Evidence role distribution

| Role | Count | Pct |
|---|---:|---:|
| bridge | 537 | 52.4% |
| answer | 193 | 18.8% |
| anchor | 133 | 13.0% |
| context | 125 | 12.2% |
| answer_bridge | 36 | 3.5% |

### Edge necessity distribution

| Necessity | Count | Pct |
|---|---:|---:|
| strong | 564 | 76.2% |
| partial | 169 | 22.8% |
| weak | 7 | 0.9% |

## 4. Coverage Diagnostics

| Metric | Covered | Total | Pct |
|---|---:|---:|---:|
| Required evidence sentences | 1022 | 1095 | 93.3% |
| Bridge sentences | 571 | 593 | 96.3% |
| Answer node present | 225 | 250 | 90.0% |

## 5. Detailed Graph Examples (up to 5)

### Example 1

**Story:** three-dogs
**Question:** how many times did the boy make an exchange with the old gray-beard ?
**Answer:** three times .
**Attribute:** action
**Reasoning:** summary_inference
**Necessity:** summary_synthesis
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [4, 16, 31]
**Bridge sentences:** [4, 16]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | action | 4 | bridge | that is why i have come , for i want to exchange my dog for  | old man, boy |
| N2 | action | 16 | bridge | that is why i have come , for i want to exchange my dog for  | old man, boy |
| N3 | outcome | 31 | answer | the youth was at once willing and agreed to close the bargai | youth, old man |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | The second exchange is a repetition of the first,  |
| N2 | N3 | results_in | strong | The repeated request for exchange leads to the you |

---

### Example 2

**Story:** three-dogs
**Question:** what will happen because the youth sent his dogs away ?
**Answer:** the troll will attack him .
**Attribute:** prediction
**Reasoning:** causal_chain
**Necessity:** causal_bridge
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [3, 4, 5]
**Bridge sentences:** [4]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | description | 3 | anchor | the giant replied : " on the mountain - top is a spring in w | giant |
| N2 | action | 4 | bridge | the youth answered : " if that be all that is lacking , one  | youth |
| N3 | motivation | 5 | answer | then the giant laughed in his false heart , for nothing suit | giant |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | causes | strong | The giant's request leads the youth to offer his d |
| N2 | N3 | results_in | strong | The youth's action of sending the dog satisfies th |

---

### Example 3

**Story:** three-dogs
**Question:** why did the two princes seize the youth by the throat and strangle him ?
**Answer:** they were jealous of the youth .
**Attribute:** causal relationship
**Reasoning:** motivation
**Necessity:** motivation_bridge
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [0, 1, 2, 3]
**Bridge sentences:** [0, 1]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | emotion | 0 | bridge | now when the princes learned that the youth had delivered th | princes, youth |
| N2 | goal | 1 | bridge | and they took counsel together as to how they might get the  | princes, youth |
| N3 | action | 2 | context | but they hid their evil plot till a favorable opportunity of | princes |
| N4 | action | 3 | answer | then they suddenly threw themselves on their comrade , seize | princes, youth |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | motivates | strong | The princes' jealousy motivates them to plot again |
| N2 | N3 | results_in | strong | The princes' goal to get the better of the youth r |
| N3 | N4 | temporal_before | partial | The princes hide their plot until an opportunity a |

---

### Example 4

**Story:** thomas-the-rhymer
**Question:** how did thomas feel after he heard the queen of fairies' demands ?
**Answer:** scared .
**Attribute:** feeling
**Reasoning:** character_state
**Necessity:** disambiguation
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [0, 3, 4]
**Bridge sentences:** [0]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | state | 0 | bridge | and, to the young man's horror, as soon as their lips had me | Thomas, Queen of Fairies |
| N2 | state | 3 | context | she saw the poor man's astonishment and terror, and she burs | Queen of Fairies, Thomas |
| N3 | state | 4 | answer | "i am not so fair to look on now as i was at first," she sai | Queen of Fairies, Thomas |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | causes | strong | The change in the Queen of Fairies causes Thomas's |
| N2 | N3 | results_in | strong | Thomas's terror leads to the Queen of Fairies conf |

---

### Example 5

**Story:** youth-who-wanted-to-win-daughter-of-mother-in-corner
**Question:** why did the mother think the youth's idea was not a bad idea ?
**Answer:** she could not afford to care for him anymore .
**Attribute:** causal relationship
**Reasoning:** summary_inference
**Necessity:** summary_synthesis
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [3, 4, 6, 7, 10]
**Bridge sentences:** [6, 7, 10]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | state | 3 | anchor | the longer this went on , the worse off his mother was. | mother |
| N2 | state | 4 | context | the youth was growing , and he wanted so much to eat that it | youth |
| N3 | state | 6 | bridge | at length it was too much for his mother. | mother |
| N4 | action | 7 | bridge | one day she told the young fellow that he ought at last to g | mother, youth |
| N5 | belief | 10 | answer_bridge | when the mother heard that she thought it might not be such  | mother |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | The worsening condition of the mother's situation  |
| N2 | N3 | causes | strong | The youth's growing need for food contributes to t |
| N3 | N4 | results_in | partial | The mother's inability to cope results in her deci |
| N4 | N5 | motivates | partial | The mother's suggestion to the youth motivates her |

---

## 6. Recommendation

### Success Criteria

| Criterion | Status | Value |
|---|---|---|
| parse_ok >= 95% | PASS | 249/250 (100%) |
| graph_valid >= 80% | PASS | 219/250 (88%) |
| required evidence coverage >= 90% | PASS | 1022/1095 (93%) |
| bridge sentence coverage >= 90% | PASS | 571/593 (96%) |
| answer node coverage >= 90% | PASS | 225/250 (90%) |
| avg nodes >= 3 | PASS | 4.1 |
| avg edges >= 2 | PASS | 3.0 |

**Overall: ALL CRITERIA PASS**

The narrative evidence graph schema is stable enough for a QG pilot.
- 219/250 (88%) graphs are valid
- 249/250 (100%) LLM responses parsed correctly
- Average 4.1 nodes and 3.0 edges per graph

Most common node type: **action** (259)
Most common edge relation: **temporal_before** (193)

**Recommended next steps:**
1. Use narrative graphs as QG scaffolding for Hard question generation
2. Focus on motivation_bridge and causal_bridge (most reliable necessity types)
3. Use causal_chain and motivation reasoning operations as primary targets
