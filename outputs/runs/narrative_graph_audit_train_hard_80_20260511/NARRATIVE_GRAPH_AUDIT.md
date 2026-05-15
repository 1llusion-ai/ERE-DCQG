# Narrative Evidence Graph Audit Report

Generated: 2026-05-11 02:15:44

## 1. Input Summary

| Field | Value |
|---|---|
| Input file | outputs/runs/fairytale_evidence_audit_train_implicit_500_20260510/candidates.jsonl |
| Total Hard candidates available | 75 |
| Sampled for extraction | 75 |
| Model | Qwen/Qwen2.5-32B-Instruct |

### Distribution by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 18 |
| disambiguation | 7 |
| motivation_bridge | 33 |
| summary_synthesis | 17 |

### Distribution by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 18 |
| character_state | 8 |
| disambiguation | 6 |
| motivation | 26 |
| summary_inference | 17 |

### Distribution by attribute

| Attribute | Count |
|---|---:|
| action | 13 |
| causal relationship | 41 |
| character | 2 |
| feeling | 12 |
| outcome resolution | 1 |
| prediction | 6 |

## 2. Graph Extraction Success

| Metric | Count | Pct |
|---|---:|---:|
| parse_ok | 75 | 100.0% |
| graph_valid | 69 | 92.0% |
| graph_invalid | 6 | 8.0% |

### Validation fail reasons

| Reason | Count |
|---|---:|
| no answer-role node found | 2 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); bridge sentences [3] have no... | 1 |
| bridge sentences [9] have no bridge-role node | 1 |
| bridge sentences [10] have no bridge-role node | 1 |
| only 1 nodes (need >= 3); only 1 edges (need >= 2); bridge sentences [3, 4] have... | 1 |

## 3. Graph Structure Statistics

| Metric | Value |
|---|---:|
| Avg nodes | 4.0 |
| Avg edges | 2.9 |
| Min nodes | 1 |
| Max nodes | 17 |
| Min edges | 1 |
| Max edges | 16 |

### Node type distribution

| Type | Count | Pct |
|---|---:|---:|
| description | 91 | 30.2% |
| action | 76 | 25.2% |
| state | 57 | 18.9% |
| motivation | 27 | 9.0% |
| belief | 17 | 5.6% |
| emotion | 16 | 5.3% |
| outcome | 7 | 2.3% |
| goal | 6 | 2.0% |
| problem | 3 | 1.0% |
| consequence | 1 | 0.3% |

### Edge relation distribution

| Relation | Count | Pct |
|---|---:|---:|
| temporal_before | 68 | 31.1% |
| causes | 52 | 23.7% |
| results_in | 44 | 20.1% |
| motivates | 28 | 12.8% |
| explains | 13 | 5.9% |
| enables | 6 | 2.7% |
| supports_inference | 5 | 2.3% |
| contrasts_with | 2 | 0.9% |
| prevents | 1 | 0.5% |

### Evidence role distribution

| Role | Count | Pct |
|---|---:|---:|
| bridge | 160 | 53.2% |
| answer | 60 | 19.9% |
| anchor | 42 | 14.0% |
| context | 28 | 9.3% |
| answer_bridge | 11 | 3.7% |

### Edge necessity distribution

| Necessity | Count | Pct |
|---|---:|---:|
| strong | 165 | 75.3% |
| partial | 53 | 24.2% |
| weak | 1 | 0.5% |

## 4. Coverage Diagnostics

| Metric | Covered | Total | Pct |
|---|---:|---:|---:|
| Required evidence sentences | 301 | 305 | 98.7% |
| Bridge sentences | 171 | 176 | 97.2% |
| Answer node present | 71 | 75 | 94.7% |

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
**Bridge sentences:** [4, 16, 31]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | action | 4 | bridge | that is why i have come , for i want to exchange my dog for  | old man, youth |
| N2 | action | 16 | bridge | that is why i have come , for i want to exchange my dog for  | old man, youth |
| N3 | outcome | 31 | answer_bridge | the youth was at once willing and agreed to close the bargai | youth, old man |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | The second exchange is a repetition of the first,  |
| N2 | N3 | results_in | strong | The third exchange results in the youth agreeing t |

---

### Example 2

**Story:** three-dogs
**Question:** why did the two princes seize the youth by the throat and strangle him ?
**Answer:** they were jealous of the youth .
**Attribute:** causal relationship
**Reasoning:** motivation
**Necessity:** motivation_bridge
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [0, 1, 3]
**Bridge sentences:** [1]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | emotion | 0 | anchor | now when the princes learned that the youth had delivered th | princes, youth |
| N2 | motivation | 1 | bridge | and they took counsel together as to how they might get the  | princes, youth |
| N3 | action | 3 | answer | then they suddenly threw themselves on their comrade , seize | princes, youth |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | motivates | strong | The princes' jealousy motivates them to plot again |
| N2 | N3 | causes | strong | The princes' plot leads to the youth being strangl |

---

### Example 3

**Story:** youth-who-wanted-to-win-daughter-of-mother-in-corner
**Question:** why did the mother think the youth's idea was not a bad idea ?
**Answer:** she could not afford to care for him anymore .
**Attribute:** causal relationship
**Reasoning:** causal_chain
**Necessity:** causal_bridge
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [3, 4, 6, 10]
**Bridge sentences:** [3, 4]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | state | 3 | bridge | the longer this went on, the worse off his mother was. | mother |
| N2 | state | 4 | bridge | the youth was growing, and he wanted so much to eat that it  | youth |
| N3 | state | 6 | context | at length it was too much for his mother. | mother |
| N4 | belief | 10 | answer | when the mother heard that she thought it might not be such  | mother |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N3 | causes | strong | the worsening condition of the mother leads to her |
| N2 | N1 | causes | strong | the youth's growing need for food exacerbates the  |
| N3 | N4 | motivates | partial | the mother's reaching a breaking point motivates h |

---

### Example 4

**Story:** werewolf
**Question:** what will the tiny old man do for the king's daughter ?
**Answer:** help her .
**Attribute:** prediction
**Reasoning:** disambiguation
**Necessity:** disambiguation
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [10, 11, 13]
**Bridge sentences:** [11]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | state | 10 | anchor | and while she sat there , lost in her thoughts , she heard a | king's daughter |
| N2 | description | 11 | bridge | why do you sit here so sad and lonely ? | king's daughter, tiny old man |
| N3 | state | 13 | answer | when she looked around there was nothing to be seen but a ti | king's daughter, tiny old man |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | The state of the king's daughter sitting and think |
| N2 | N3 | results_in | strong | The tiny old man's question leads to the king's da |

---

### Example 5

**Story:** youth-who-wanted-to-win-daughter-of-mother-in-corner
**Question:** how did the woman feel about her son singing and dancing ?
**Answer:** unhappy .
**Attribute:** feeling
**Reasoning:** character_state
**Necessity:** motivation_bridge
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [2, 3, 6]
**Bridge sentences:** [3]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | action | 2 | anchor | but he liked to sing and to dance , and that is what he did  | son |
| N2 | state | 3 | bridge | the longer this went on , the worse off his mother was. | mother |
| N3 | state | 6 | answer | at length it was too much for his mother. | mother |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | causes | strong | The son's continuous singing and dancing led to hi |
| N2 | N3 | results_in | strong | The worsening state of the mother eventually resul |

---

## 6. Recommendation

### Success Criteria

| Criterion | Status | Value |
|---|---|---|
| parse_ok >= 95% | PASS | 75/75 (100%) |
| graph_valid >= 80% | PASS | 69/75 (92%) |
| required evidence coverage >= 90% | PASS | 301/305 (99%) |
| bridge sentence coverage >= 90% | PASS | 171/176 (97%) |
| answer node coverage >= 90% | PASS | 71/75 (95%) |
| avg nodes >= 3 | PASS | 4.0 |
| avg edges >= 2 | PASS | 2.9 |

**Overall: ALL CRITERIA PASS**

The narrative evidence graph schema is stable enough for a QG pilot.
- 69/75 (92%) graphs are valid
- 75/75 (100%) LLM responses parsed correctly
- Average 4.0 nodes and 2.9 edges per graph

Most common node type: **description** (91)
Most common edge relation: **temporal_before** (68)

**Recommended next steps:**
1. Use narrative graphs as QG scaffolding for Hard question generation
2. Focus on motivation_bridge and causal_bridge (most reliable necessity types)
3. Use causal_chain and motivation reasoning operations as primary targets
