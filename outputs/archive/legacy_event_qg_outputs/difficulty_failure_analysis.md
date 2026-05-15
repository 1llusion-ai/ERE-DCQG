# Difficulty Failure Analysis: Why Hard Isn't Harder

**Date**: 2026-04-29
**Data**: compare_PathQG_evaluated_retry_v2.jsonl (230 grammar-passed, 81E/83M/66H)
**Metric**: solver_correct — Easy 0.346, Medium 0.349, Hard 0.364 (higher = easier)

---

## Root Causes (7 patterns identified)

### 1. Shortcut Temporal Phrases (Pervasive)

Questions use template phrases that act as answer locators rather than reasoning constraints:

| Phrase | Freq | Problem |
|--------|------|---------|
| "What happened after..." | Very high | Solver scans for the next sentence |
| "What did X do after Y..." | High | Single-hop lookback from event Y |
| "What was the outcome..." | Medium | Points directly to last event |
| "What event occurred after..." | Medium | Same as above |
| "following the..." | High | Temporal anchor leaks position |

**Example (Hard #12)**: *"What happened after the attempts to establish coalition governments?"* — answer directly in sentence S7, no multi-hop needed.

### 2. Single-Sentence Answerability

Most questions can be answered by reading ONE sentence, regardless of path length:

- **Hard #18** (4 events): *"What activity did Dickens become nervous about after the Staplehurst rail crash?"* — S5 has "nervous when travelling by train"
- **Hard #19** (3 events): *"What stage of the European Cup was Manchester United advancing to after the crash?"* — S4 has "advance to the semi-finals"
- **Hard #10** (3 events): *"What was the purpose of American military interventions in Nicaragua?"* — S2 has "designed to stop..."

The solver never needs to chain across events. It finds one sentence and reads the answer.

### 3. Missing Intermediate Event Binding

Hard items have 3-4 path events but questions only reference 1 (rarely 2). The path is used to select context sentences but NOT enforced in question wording:

| Question | Path Events Used | Path Events Available |
|----------|-----------------|----------------------|
| "What activity did Dickens become nervous about after the crash?" | 1 (crash) | 3 |
| "What stage was MU advancing to after the crash?" | 1 (crash) | 3 |
| "What happened after the attempts to establish coalition?" | 1 (attempts) | 3 |

### 4. More Context = Easier (Paradox)

Hard items have 5-8 supporting sentences (due to Evidence Span in difficulty scoring). This gives the solver MORE surface area to find the answer:

- Easy avg sentences: ~3.5
- Medium avg sentences: ~4.5
- Hard avg sentences: ~5.5

The Evidence Span dimension of difficulty scoring (ES) is counterproductive — it adds "difficulty" points but makes questions easier to answer.

### 5. Gold Trigger Explicit in Context

The gold answer trigger is stated verbatim in the supporting sentences for almost all cases. The solver only needs to extract it, not reason about it.

**Example (Hard #16)**: Gold trigger is "win" and S14 says "Manchester United were trying to become the third club to **win** three successive English league titles."

### 6. Over-Specific Temporal Anchoring

Questions give away the exact temporal location of the answer:
- "by September 16 when..."
- "on November 17..."
- "during the police raid..."
- "before the crash..."

These anchors allow the solver to locate the exact sentence and extract the answer without understanding event relationships.

### 7. "After" Pattern Makes Last-Sentence Guessing Optimal

When questions use "What happened after X", the solver can guess from sentences near the end and be correct often — because the answer event IS near the end of the context window.

---

## Hard Item Categorization (20 samples)

| Category | Count | Description |
|----------|-------|-------------|
| Single-sentence answerable | 14/20 | Answer in one sentence, no chaining |
| Shortcut phrase ("after", "outcome") | 11/20 | Template phrases that give away position |
| Only 1 path event referenced | 15/20 | Question uses ≤1 event from path |
| Over-specific temporal anchor | 7/20 | "during X", "on date Y" |
| Answer verbatim in context | 18/20 | Gold trigger explicitly present |

---

## Fix Strategy

### Prompt changes (Task 2)
1. **Hard**: Require explicit binding of 2+ prior events in question
2. **Hard**: Ban shortcut phrases ("final outcome", "what happened after", "what action was taken")
3. **Hard**: Question must NOT be answerable from the last sentence alone
4. **Medium**: Require mention of start event + at least one intermediate event
5. **Easy**: Allow 1-hop, keep simple

### New judge (Task 3)
Add `path_faithfulness_judge` that checks:
- NeedIntermediateEvents (yes/no)
- EvidenceHopsUsed (1/2/3+)
- CanAnswerFromSingleSentence (yes/no)

Hard pass = all 3 conditions met.

### Evaluation target
Not "higher composite" but "difficulty monotonicity": solver_correct Easy > Medium > Hard.
