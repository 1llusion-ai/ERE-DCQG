# Narrative Evidence Graph Audit Report

Generated: 2026-05-10 19:29:26

## 1. Input Summary

| Field | Value |
|---|---|
| Input file | outputs/runs/fairytale_evidence_audit_train_implicit_500_20260510/candidates.jsonl |
| Total Hard candidates available | 75 |
| Sampled for extraction | 20 |
| Model | Qwen/Qwen2.5-32B-Instruct |

### Distribution by necessity_type

| Type | Count |
|---|---:|
| causal_bridge | 5 |
| disambiguation | 5 |
| motivation_bridge | 5 |
| summary_synthesis | 5 |

### Distribution by reasoning_operation

| Operation | Count |
|---|---:|
| causal_chain | 5 |
| character_state | 1 |
| disambiguation | 5 |
| motivation | 4 |
| summary_inference | 5 |

### Distribution by attribute

| Attribute | Count |
|---|---:|
| action | 4 |
| causal relationship | 12 |
| character | 1 |
| feeling | 2 |
| prediction | 1 |

## 2. Graph Extraction Success

| Metric | Count | Pct |
|---|---:|---:|
| parse_ok | 20 | 100.0% |
| graph_valid | 19 | 95.0% |
| graph_invalid | 1 | 5.0% |

### Validation fail reasons

| Reason | Count |
|---|---:|
| no answer-role node found | 1 |

## 3. Graph Structure Statistics

| Metric | Value |
|---|---:|
| Avg nodes | 3.2 |
| Avg edges | 2.2 |
| Min nodes | 3 |
| Max nodes | 5 |
| Min edges | 2 |
| Max edges | 4 |

### Node type distribution

| Type | Count | Pct |
|---|---:|---:|
| state | 24 | 36.9% |
| action | 18 | 27.7% |
| description | 8 | 12.3% |
| motivation | 4 | 6.2% |
| emotion | 3 | 4.6% |
| outcome | 2 | 3.1% |
| goal | 2 | 3.1% |
| problem | 2 | 3.1% |
| belief | 1 | 1.5% |
| consequence | 1 | 1.5% |

### Edge relation distribution

| Relation | Count | Pct |
|---|---:|---:|
| causes | 16 | 35.6% |
| temporal_before | 10 | 22.2% |
| results_in | 7 | 15.6% |
| motivates | 6 | 13.3% |
| explains | 4 | 8.9% |
| contrasts_with | 1 | 2.2% |
| enables | 1 | 2.2% |

### Evidence role distribution

| Role | Count | Pct |
|---|---:|---:|
| bridge | 29 | 44.6% |
| answer | 15 | 23.1% |
| anchor | 11 | 16.9% |
| context | 6 | 9.2% |
| answer_bridge | 4 | 6.2% |

### Edge necessity distribution

| Necessity | Count | Pct |
|---|---:|---:|
| strong | 42 | 93.3% |
| partial | 3 | 6.7% |

## 4. Coverage Diagnostics

| Metric | Covered | Total | Pct |
|---|---:|---:|---:|
| Required evidence sentences | 65 | 65 | 100.0% |
| Bridge sentences | 33 | 33 | 100.0% |
| Answer node present | 19 | 20 | 95.0% |

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
| N2 | action | 16 | bridge | the old man noticed the boy again and proposed the same exch | old man, boy |
| N3 | outcome | 31 | answer_bridge | the youth agreed to the exchange. | youth, old man |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | temporal_before | strong | the second proposal by the old man follows the fir |
| N2 | N3 | causes | strong | the old man's second proposal leads to the youth a |

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
| N1 | state | 3 | bridge | the longer this went on , the worse off his mother was . | mother |
| N2 | state | 4 | bridge | the youth was growing , and he wanted so much to eat that it | youth |
| N3 | state | 6 | context | at length it was too much for his mother . | mother |
| N4 | belief | 10 | answer | when the mother heard that she thought it might not be such  | mother |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N3 | causes | strong | the worsening condition of the mother leads to her |
| N2 | N1 | causes | strong | the youth's growing need for food exacerbates the  |
| N3 | N4 | motivates | partial | the mother's reaching a breaking point motivates h |

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
| N1 | N2 | temporal_before | strong | The courtier's actions in N1 lead to the forced pr |
| N2 | N3 | causes | strong | The courtier's forced promise in N2 results in his |

---

### Example 5

**Story:** werewolf
**Question:** how did the queen treat the princess after the king left ?
**Answer:** poorly .
**Attribute:** action
**Reasoning:** summary_inference
**Necessity:** summary_synthesis
**Valid:** True
**Validation:** all checks passed

**Required evidence sentences:** [1, 3, 4]
**Bridge sentences:** [3]

**Nodes:**

| ID | Type | Sent | Role | Text | Participants |
|---|---|---:|---|---|---|
| N1 | action | 1 | anchor | news came that the enemy had entered the land, and the king  | king |
| N2 | state | 3 | bridge | for no sooner had the king departed than the queen showed he | queen, king |
| N3 | state | 4 | answer | not a day went by without her scolding and threatening the p | queen, princess, queen's daugh |

**Edges:**

| Source | Target | Relation | Necessity | Reason |
|---|---|---|---|---|
| N1 | N2 | causes | strong | The king's departure triggered the queen's true na |
| N2 | N3 | results_in | strong | The queen's harsh and unkind nature resulted in he |

---

## 6. Recommendation

### Success Criteria

| Criterion | Status | Value |
|---|---|---|
| parse_ok >= 95% | PASS | 20/20 (100%) |
| graph_valid >= 80% | PASS | 19/20 (95%) |
| required evidence coverage >= 90% | PASS | 65/65 (100%) |
| bridge sentence coverage >= 90% | PASS | 33/33 (100%) |
| answer node coverage >= 90% | PASS | 19/20 (95%) |
| avg nodes >= 3 | PASS | 3.2 |
| avg edges >= 2 | PASS | 2.2 |

**Overall: ALL CRITERIA PASS**

The narrative evidence graph schema is stable enough for a QG pilot.
- 19/20 (95%) graphs are valid
- 20/20 (100%) LLM responses parsed correctly
- Average 3.2 nodes and 2.2 edges per graph

Most common node type: **state** (24)
Most common edge relation: **causes** (16)

**Recommended next steps:**
1. Use narrative graphs as QG scaffolding for Hard question generation
2. Focus on motivation_bridge and causal_bridge (most reliable necessity types)
3. Use causal_chain and motivation reasoning operations as primary targets
