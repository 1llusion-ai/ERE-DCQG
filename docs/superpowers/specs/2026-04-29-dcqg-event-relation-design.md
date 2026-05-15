# DCQG: Event-Relation-Graph-Guided Controllable Question Generation

## 1. Problem Statement

Given a document-level event relation graph from the MAVEN-ERE dataset, generate questions at controlled difficulty levels (Easy / Medium / Hard) by sampling structured event reasoning paths and computing a four-dimensional difficulty score — without any model training. The difficulty signal comes entirely from graph structural features, not from human annotations or fine-tuning.

Goal: publish in a CCF-C venue or AI+education journal within one month of experiments.

## 2. Dataset

**MAVEN-ERE** (3 splits):
- Train: 2,913 documents
- Valid: 710 documents
- Test: 857 documents

Each document contains:
- `tokens`: tokenized sentences
- `sentences`: sentence-level text spans
- `events`: event mentions with trigger words, types, sentence IDs, mention IDs
- `causal_relations`: {CAUSE: [[src, tgt, ...], ...], PRECONDITION: [[src, tgt, ...], ...]}
- `temporal_relations`: {BEFORE: [...], AFTER: [...], SIMULTANEOUS: [...], INCLUDES: [...], IS_INCLUDED: [...]}
- `subevent_relations`: [[parent, child], ...]
- (Note: coreference annotation if available in MAVEN-ERE should be leveraged for Event Ambiguity)

## 3. Difficulty Dimensions

Four independent, interpretable dimensions:

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Path Length (PL)** | Number of hops in the sampled target reasoning path | 1 hop → 1, 2 hops → 2, ≥3 hops → 3 |
| **Relation Diversity (RD)** | Number of distinct relation types on the target path | 1 type → 1, 2 types → 2, 3 types → 3 |
| **Evidence Span (ES)** | Number of unique sentences spanned by the path | max(sent_id) - min(sent_id) + 1 → 1 (≤2), 2 (3-5), 3 (≥6) |
| **Event Ambiguity (EA)** | Presence of coreference chains, similar event types, or multiple candidate events near the path | none → 1, moderate → 2, high → 3 |

**Difficulty Score**: `D = PL + RD + ES + EA` (additive,总分4–12)

**Difficulty Level Thresholds** (can be calibrated by percentile on training set):
- Easy: D ∈ [4, 6]
- Medium: D ∈ [7, 9]
- Hard: D ∈ [10, 12]

Note: thresholds can later be refined via training set score distribution or kept as fixed rules — both approaches are valid for a CCF-C paper.

## 4. Path-Level Instance Extraction Pipeline

Instance unit is a **reasoning path** (not a single relation pair):

```
1. Build document-level event graph:
   - Nodes: events with their mention IDs, trigger words, types, sentence IDs
   - Edges: typed relations (causal CAUSE/PRECONDITION, temporal BEFORE/AFTER/SIMULTANEOUS/INCLUDES/IS_INCLUDED, subevent)

2. Sample target paths by difficulty rules:
   - Easy: single-hop subevent or temporal relation (no intermediate event)
   - Medium: 2-hop path, or single-hop causal/temporal with evidence spanning ≥2 sentences
   - Hard: ≥3-hop path OR mixed-relation path (≥2 relation types) with evidence spanning ≥3 sentences

3. For each sampled path [e1 → e2 → ... → ek]:
   - src_event = e1 (question subject)
   - tgt_event = ek (answer event)
   - intermediate_events = [e2, ..., ek-1]
   - path_relation_types = [r(e1,e2), r(e2,e3), ..., r(ek-1,ek)]
   - context_sentences = all sentences in document
   - path_sentence_ids = [sent_id(e1), ..., sent_id(ek)]

4. Compute four-dimensional score:
   - PL: k-1 (number of hops)
   - RD: distinct count of relation types on the path
   - ES: max(sent_ids) - min(sent_ids) + 1, bucketed into 1/2/3
   - EA: check for coreference chains linking events on path, similar event types in ±2 sentences, or multiple events matching partial path constraints
```

## 5. Question Generation Strategy

**Model**: Qwen2.5-7B (zero-shot, no fine-tuning)

**Input prompt structure** (three-shot prompting with path-level examples):

```
System: You are a question generation assistant. Given an event reasoning path, generate a question at the specified difficulty level.

[Easy Example]
Path: "declared bankruptcy" (subevent) → "closed offices"
Question: "What happened to the offices after the company declared bankruptcy?"
Difficulty: Easy

[Medium Example]
Path: "signed treaty" → "sent troops" → "ended conflict" (2 hops, temporal+BEFORE)
Question: "After the treaty was signed, what led to the end of the conflict?"
Difficulty: Medium

[Hard Example]
Path: "drought" → "crop failure" (causal) → "price surge" (causal) → "food shortage" (causal), 3 relation types, evidence spans 5 sentences
Question: "How did the drought ultimately lead to food shortages across the region?"
Difficulty: Hard

---

Now generate a question for:

Event Path: {src_trigger} → {intermediate_triggers} → {tgt_trigger}
Relation Types: {path_relation_types}
Difficulty: {Easy|Medium|Hard}
Evidence Span: {ES} sentences

Context:
{context_sentences}

Requirements:
- The question must be answerable from the context
- The difficulty level corresponds to the structural complexity of the path
- Generate natural, grammatically correct questions
- For Medium/Hard: include intermediate reasoning steps implicitly in the question
```

**Difficulty-aware prompt injection**: Path structure + difficulty label + shot examples are the only control signals — no training involved.

**Rule-based post-filtering**:
- Discard if generated question does not mention the target event trigger
- Discard if answer is not derivable from the path context
- Discard if format is invalid (empty, too short, no question mark)

## 6. Baselines

| Baseline | Description |
|----------|-------------|
| **Direct LLM** | Give context + target difficulty only, no event path information |
| **CoT LLM** | Give context + target difficulty + chain-of-thought reasoning prompt |
| **RelationTypeQG** | Give single relation type (or single relation pair) without path structure, map type directly to difficulty |
| **PathOnlyQG** | Give event path but without four-dimensional difficulty score or rule filter |
| **Ours (Full4Dim)** | Event path sampling + four-dimensional difficulty score + rule filter |
| **SingleDimQG** (ablation) | Use each dimension alone (PL-only, RD-only, ES-only, EA-only) to measure individual contribution |

Note: RandomQG is removed as it is not informative as a baseline.

## 7. Evaluation Framework (three layers)

### Layer 1: Automatic Rule Metrics
- **Format validity**: question ends with "?", length > 10 chars
- **Answer leakage**: target event trigger appears in question (may or may not be desired)
- **Evidence coverage**: all events on the path appear in the supporting context
- **Relation coverage**: all relation types on the path are represented in the question or answer
- **Difficulty rule consistency**: does the generated question's structure plausibly match the labeled difficulty level (heuristic check)

### Layer 2: LLM Judge (GPT-4o-mini)
- **Answerability** (1–5): can the question be answered from context
- **Relation relevance** (1–5): does the question address the given event relation
- **Evidence consistency** (1–5): does the answer align with the evidence
- **Difficulty alignment** (1–5): does the question's complexity match the specified difficulty
- **Fluency** (1–5): grammaticality and naturalness

### Layer 3: Solver Accuracy
- Use Qwen2.5-7B as solver to answer each generated question
- LLM judge (GPT-4o-mini) evaluates whether solver answer is semantically equivalent to the gold answer event
- Compute accuracy per difficulty level; expect: Easy > Medium > Hard (monotonic decreasing)
- This serves as the primary validation that difficulty levels are genuinely meaningful

## 8. Experiment Scale

To keep experiment runnable within timeline:

- **Total instances**: 900–1,500 (300–500 per difficulty), sampled from test set with balanced distribution
- **Human evaluation**: 100–150 instances (randomly sampled, ~30–50 per difficulty), two annotators, Cohen's κ ≥ 0.6 target
- **GPT-as-Judge**: run on all generated instances
- **Solver accuracy**: run on all instances

If temporal relations are too large (MAVEN-ERE has many), filter by relation type availability per difficulty target before sampling.

## 9. Expected Outcomes

1. **Primary**: Solver accuracy shows Easy > Medium > Hard (significant gap, e.g., Easy >80%, Hard <50%)
2. **Secondary**: GPT-as-Judge difficulty alignment scores are significantly higher for Ours vs. PathOnlyQG vs. Direct LLM
3. **Ablation**: SingleDimQG vs. Ours reveals which dimension contributes most to difficulty control
4. **Human eval**: Cohen's κ ≥ 0.6 on difficulty alignment

## 10. Timeline (4 weeks)

| Week | Tasks |
|------|-------|
| **Week 1** | Build event graph + path sampler + 4-dim scorer → generate instances → run all baselines |
| **Week 2** | GPT-as-Judge evaluation on all instances → solver accuracy → first-pass analysis |
| **Week 3** | Human evaluation (100–150 samples) → ablation tables/figures → refine thresholds |
| **Week 4** | Paper writing (related work, method, experiments, analysis) |

## 11. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Qwen2.5-7B generation quality low for Hard | Use CoT prompting; add reasoning step in prompt |
| Difficulty alignment scores not discriminating | Adjust scoring buckets; verify with human eval |
| Path sampling skews distribution | Balance Easy/Medium/Hard during sampling |
| Timeline slip | Parallelize human eval with baseline experiments |

## 12. Paper Outline (target: AI+Education or CCF-C NLP venue)

1. Introduction — motivation: event relation graph structure for difficulty control in QG
2. Related Work — QG, Controllable QG, Event Relation extraction, DCQG
3. Method — event graph building → path sampling → 4-dim scoring → zero-shot QG → rule filter
4. Experiments — baselines, GPT-as-Judge results, solver accuracy, ablation
5. Human Evaluation — annotation setup, Cohen's κ, results
6. Analysis & Discussion — which dimension matters most, error analysis
7. Conclusion & Future Work