# Narrative Evidence Graph Audit Report

Generated: 2026-05-11 13:09:14

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
| graph_valid | 71 | 94.7% |
| graph_invalid | 4 | 5.3% |

### Validation fail reasons

| Reason | Count |
|---|---:|
| no answer-role node found | 3 |
| no connected path to answer node | 1 |

## 3. Graph Structure Statistics

| Metric | Value |
|---|---:|
| Avg nodes | 4.1 |
| Avg edges | 2.8 |
| Min nodes | 3 |
| Max nodes | 17 |
| Min edges | 2 |
| Max edges | 8 |

### Node type distribution

| Type | Count | Pct |
|---|---:|---:|
| description | 84 | 27.5% |
| state | 71 | 23.3% |
| action | 59 | 19.3% |
| motivation | 36 | 11.8% |
| emotion | 16 | 5.2% |
| belief | 16 | 5.2% |
| outcome | 9 | 3.0% |
| problem | 5 | 1.6% |
| goal | 4 | 1.3% |
| attempt | 3 | 1.0% |
| consequence | 1 | 0.3% |
| resolution | 1 | 0.3% |

### Edge relation distribution

| Relation | Count | Pct |
|---|---:|---:|
| temporal_before | 62 | 29.1% |
| causes | 52 | 24.4% |
| results_in | 36 | 16.9% |
| motivates | 31 | 14.6% |
| explains | 20 | 9.4% |
| enables | 8 | 3.8% |
| contrasts_with | 2 | 0.9% |
| supports_inference | 2 | 0.9% |

### Evidence role distribution

| Role | Count | Pct |
|---|---:|---:|
| bridge | 163 | 53.4% |
| answer | 60 | 19.7% |
| anchor | 41 | 13.4% |
| context | 28 | 9.2% |
| answer_bridge | 13 | 4.3% |

### Edge necessity distribution

| Necessity | Count | Pct |
|---|---:|---:|
| strong | 169 | 79.3% |
| partial | 43 | 20.2% |
| weak | 1 | 0.5% |

## 4. Coverage Diagnostics

| Metric | Covered | Total | Pct |
|---|---:|---:|---:|
| Required evidence sentences | 305 | 305 | 100.0% |
| Bridge sentences | 176 | 176 | 100.0% |
| Answer node present | 72 | 75 | 96.0% |

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
| N1 | action | 4 | bridge | the old man noticed the boy and began to propose an exchange | old man, boy |
| N2 | action | 16 | bridge | the old man noticed the boy again and proposed an exchange o | old man, boy |
| N3 | outcome | 31 | answer_bridge | the youth agreed to the exchange. | youth, old man |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | The second proposal follows the first. |
| N2 | N3 | causes | strong | The second proposal leads to the agreement. |

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
| N1 | problem | 3 | bridge | the longer this went on , the worse off his mother was. | mother |
| N2 | problem | 4 | bridge | the youth was growing , and he wanted so much to eat that it | youth |
| N3 | outcome | 6 | anchor | at length it was too much for his mother. | mother |
| N4 | belief | 10 | answer | when the mother heard that she thought it might not be such  | mother |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N3 | causes | strong | the worsening condition of the mother leads to the |
| N2 | N3 | causes | strong | the youth's growing need for food contributes to t |
| N3 | N4 | motivates | partial | the mother's inability to cope motivates her to re |

---

### Example 4

**Story:** silverwhite-lillwacker
**Question:** why was the courtier acclaimed as the greatest of heroes ?
**Answer:** he pretended to have saved the two princesses .
**Attribute:** causal relationship
**Reasoning:** disambiguation
**Necessity:** disambiguation
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [0, 9, 10]
**Bridge sentences:** [9]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | action | 0 | anchor | now when the battle was over and the youth had disappeared , | the courtier, the princess |
| N2 | action | 9 | bridge | he climbed down from his tree and forced the princess to pro | the courtier, the princess |
| N3 | outcome | 10 | answer | then they returned to the castle , where the courtier was ac | the courtier |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | The courtier's action of threatening the princess  |
| N2 | N3 | causes | strong | The courtier's forcing the princess to say he resc |

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
| graph_valid >= 80% | PASS | 71/75 (95%) |
| required evidence coverage >= 90% | PASS | 305/305 (100%) |
| bridge sentence coverage >= 90% | PASS | 176/176 (100%) |
| answer node coverage >= 90% | PASS | 72/75 (96%) |
| avg nodes >= 3 | PASS | 4.1 |
| avg edges >= 2 | PASS | 2.8 |

**Overall: ALL CRITERIA PASS**

The narrative evidence graph schema is stable enough for a QG pilot.
- 71/75 (95%) graphs are valid
- 75/75 (100%) LLM responses parsed correctly
- Average 4.1 nodes and 2.8 edges per graph

Most common node type: **description** (84)
Most common edge relation: **temporal_before** (62)

**Recommended next steps:**
1. Use narrative graphs as QG scaffolding for Hard question generation
2. Focus on motivation_bridge and causal_bridge (most reliable necessity types)
3. Use causal_chain and motivation reasoning operations as primary targets
