# DCQG Direction Update: Necessity-Verified Difficulty Control

**Date:** 2026-05-10
**Decision:** Keep the research direction as difficulty-controlled question generation. Do not reduce the project to a two-level or structural-grounding-only paper. The method should shift from raw event-path hop count to necessity-verified evidence paths.

---

## 1. Current Finding

The latest QG-eligible Hard rescue result shows that event-path hop count alone is not a sufficient difficulty-control signal.

Run directory:

- `outputs/runs/hard_rescue_qg_eligible_20path_20260509/`

Verified result summary:

| Metric | Value |
|---|---:|
| QG-eligible Hard paths | 249/2598 (9.6%) |
| Fixed-policy qualified among QG-eligible | 126/249 (50.6%) |
| Test paths | 20 |
| Candidates | 120 |
| Generation errors | 8 (6.7%) |
| Quality filter pass | 21/120 (17.5%) |
| Blind Hard among all judged candidates | 0/112 |
| Blind Hard among quality-pass candidates | 0/21 |
| Quality-pass Blind Easy | 20/21 |
| Quality-pass Blind Medium | 1/21 |
| Quality-pass SSA=yes | 20/21 |

Interpretation:

- QG-eligible filtering removed most malformed answer paths.
- Grammar fixes reduced generation failures.
- But generated Hard-target questions still collapse to Easy/Medium under blind difficulty evaluation.
- The recurring blind judge explanation is that the answer can be found directly from a single answer sentence.

---

## 2. Why The Current Event-Hop Hypothesis Failed

The failed hypothesis was:

```text
Hard = 3-hop event path
```

The observed reality is:

```text
The document has a 3-hop event path,
but the generated question asks for the final event phrase,
and the final event phrase is usually locally identifiable in one sentence.
```

Therefore the current setup controls **graph hops**, but not necessarily **answering hops**.

The correct distinction:

```text
Graph hop:
An edge/path exists in the event graph.

Reasoning hop:
The solver must use this evidence step to identify the answer.
```

The current Hard failures are not proof that multi-hop cannot define difficulty. They show that event-path length must be verified for evidence necessity.

---

## 3. Why Multi-Hop QA Can Use Hop Count

Prior multi-hop QA work can use multi-hop structure as a difficulty signal when these conditions hold:

1. Each hop is necessary evidence for finding the answer.
2. The question does not directly point to the answer sentence.
3. The answer is not merely a locally extractable final event trigger or phrase.
4. Single-hop shortcuts are filtered out.

The current DCQG setup often violates these conditions because MAVEN-ERE event answers are final event phrases that appear in one local sentence.

So the issue is not:

```text
Multi-hop cannot define difficulty.
```

The issue is:

```text
The current event-path hop count has not been shown to be necessary for answering.
```

---

## 4. Revised Method Hypothesis

Use event graphs as candidate evidence structures, but make necessity verification the true difficulty signal.

Updated hypothesis:

```text
Hard = 3+ necessary evidence steps required to identify the answer.
```

Proposed framing:

```text
Difficulty-Controlled QG via Necessity-Verified Event Evidence Paths
```

Core rule:

```text
A Hard question must require anchor + bridge/disambiguator + answer evidence.
Removing any required bridge evidence should make the question unanswerable or ambiguous.
```

---

## 5. New Difficulty Definitions

| Level | Evidence requirement |
|---|---|
| Easy | Answer sentence alone is sufficient. |
| Medium | Requires an anchor sentence plus answer sentence, usually 2 evidence steps. |
| Hard | Requires anchor + bridge/disambiguator + answer; answer sentence alone is insufficient. |

Candidate fields to mine and verify:

- `answer_sentence_id`
- `anchor_sentence_ids`
- `bridge_sentence_ids`
- `evidence_span`
- `num_required_sentences`
- `answer_locality`: `single_sentence`, `two_sentence`, `multi_sentence`
- `reasoning_operation`: `bridge`, `contrast`, `temporal_order`, `causal_chain`, `disambiguation`, `comparison`
- `answer_sentence_alone_sufficient`: `yes`, `partial`, `no`
- `evidence_necessity`: `weak`, `partial`, `strong`

---

## 6. Immediate Next Action

Stop optimizing event-hop-only Hard prompts.

First build an evidence-necessity miner and run an audit over the validation set.

Minimum audit output:

- Easy candidate count
- Medium candidate count
- Hard evidence candidate count
- Hard candidates where `answer_sentence_alone_sufficient=no`
- Distribution by `reasoning_operation`
- Distribution by answer type
- Example traces for at least 10 candidate Hard cases

Only after this audit shows enough true Hard candidates should QG prompts be redesigned around evidence roles.

