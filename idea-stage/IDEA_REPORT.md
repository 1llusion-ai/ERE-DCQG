# Idea Discovery Report

**Direction**: Difficulty-Controlled Question Generation (DCQG) using LLMs + Event Relation Graphs
**Date**: 2026-04-29
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

## Executive Summary

The project already has a strong core idea (4-dimensional graph-structural difficulty scoring for event-relation-based QG) with working Stage 1/Stage 2 code. The literature confirms novelty: no prior work combines MAVEN-ERE event graphs with difficulty-controlled question generation. Below are **10 additional/extension ideas** that complement or expand the existing work, ranked by publishability, feasibility, and novelty.

## Literature Landscape Summary

See Phase 1 above. Key takeaway: the DCQG field is active but fragmented. CrossQG (EMNLP 2025) is the closest competitor but uses pure LLM prompting without structured knowledge. DiffQG uses KGs but not event-specific relations. The event-graph + difficulty-control intersection is open territory.

---

## Ranked Ideas

### Idea 1: CrossQG-Style Consistency Enhancement for Event-Graph QG — RECOMMENDED EXTENSION

**Type**: Method improvement (builds on existing code)
**Effort**: 1-2 weeks
**Venue target**: EMNLP 2026 Workshop / NLP-CC 2026

**What**: Adapt CrossQG's contrast enhancement and cross-filtering to the event-graph QG pipeline. Instead of relying solely on the 4-dim difficulty scorer at input time, add a post-generation consistency check:
1. Generate questions at all three difficulty levels for the same event path
2. Cross-filter: use GPT-4o-mini to rank which question is hardest/easiest
3. Keep only questions where the judge's ranking matches the intended difficulty ordering
4. Measure whether this filtering improves solver accuracy gap (Easy vs Hard)

**Why novel**: CrossQG was tested on standard QA datasets (SQuAD, HotpotQA), not on event-graph-generated questions. Combining structural difficulty signals with consistency-based filtering is a new combination.

**Risk**: Low. Builds directly on existing pipeline. If it works, it's a strong ablation. If not, it's a useful negative result for the paper.

**Pilot**: Run on 90 existing instances (30/level), compare filter-pass rate and solver accuracy gap vs. no-filter baseline.

---

### Idea 2: Learner-Aware Difficulty Personalization via Simulated Student Models

**Type**: New direction (complementary to existing work)
**Effort**: 2-3 weeks
**Venue target**: AIED 2026 / BEA Workshop @ ACL 2026

**What**: Extend DCQG from "absolute difficulty" to "learner-relative difficulty":
1. Build a simple simulated learner: track which event types, relation types, and evidence spans a learner has previously answered correctly/incorrectly
2. Use Item Response Theory (IRT) to estimate learner ability θ
3. Generate questions where difficulty is personalized: same event path may be "Hard" for learner A but "Medium" for learner B
4. Evaluate: does personalized difficulty improve learning efficiency (measured by simulated learner improvement rate)?

**Why novel**: All existing DCQG work uses absolute difficulty. Learner-relative difficulty is unexplored territory. IRT + event graph structure is a new combination.

**Data**: Can use MAVEN-ERE test split + simulated learner responses. No real student data needed for proof-of-concept.

**Risk**: Medium. Simulated learner behavior may not reflect real learners. But for a workshop paper, simulation-based validation is acceptable.

**Pilot**: Build simulated learner with 3 ability levels, generate 30 questions per learner, measure answer accuracy correlation with learner ability.

---

### Idea 3: Counterfactual Difficulty Calibration (inspired by DiffQG)

**Type**: Method improvement
**Effort**: 2-3 weeks
**Venue target**: CCF-C NLP venue / NLPCC 2026

**What**: Use counterfactual reasoning to verify that difficulty labels are causally meaningful:
1. For each generated Hard question, create a counterfactual Easy version by modifying only the path structure (e.g., shorten path from 3-hop to 1-hop) while keeping the same source and target events
2. Verify that the Easy counterfactual is actually easier to answer (via solver accuracy)
3. This provides causal evidence that the 4-dim scoring captures genuine difficulty, not just correlation

**Why novel**: DiffQG proposed counterfactual reasoning for KG-based QG but never applied it to event relation graphs. The causal verification angle is strong for reviewers.

**Risk**: Medium. Counterfactual path modification may produce unnatural paths. Need careful filtering.

**Pilot**: Take 50 Hard instances, generate counterfactual Easy versions (shorten path, reduce relation diversity), verify solver accuracy gap >20%.

---

### Idea 4: Multi-Agent Debate for Question Quality Verification

**Type**: Method improvement
**Effort**: 1-2 weeks
**Venue target**: NLP Workshop 2026

**What**: Replace single LLM judge with a multi-agent debate framework:
1. **Generator agent** (Qwen2.5-7B): generates question from event path
2. **Critic agent** (GPT-4o-mini): identifies flaws (ambiguity, answer leakage, difficulty mismatch)
3. **Refiner agent** (Qwen2.5-7B): revises question based on critic feedback
4. **Solver agent** (Qwen2.5-7B): attempts to answer; if solver succeeds on "Hard" or fails on "Easy", question is flagged
5. Iterate for up to 3 rounds

**Why novel**: Multi-agent QG has been explored for general QA but not for difficulty-controlled event-graph QG. The debate-then-refine loop is a natural fit for difficulty calibration.

**Risk**: Low-Medium. API costs increase with multiple agents, but only need to run on subset. Clear ablation: single-agent vs. multi-agent.

**Pilot**: 30 instances, compare multi-agent vs. single-agent on filter pass rate and difficulty alignment.

---

### Idea 5: Relation-Type-Aware Difficulty Prompting

**Type**: Method improvement (simple, high-impact)
**Effort**: 1 week
**Venue target**: CCF-C / BEA Workshop

**What**: The current prompt tells the LLM the difficulty level but doesn't explain *why* it's Hard. Add relation-type-aware reasoning hints:
- For causal paths: "This is Hard because it requires understanding a 3-step causal chain across 5 sentences"
- For temporal paths: "This is Medium because it requires tracking temporal order across 3 sentences"
- For mixed paths: "This is Hard because it combines causal and temporal reasoning"

**Why novel**: Simple but unexplored. CrossQG uses contrast between difficulty levels but doesn't explain *why* something is difficult. Relation-type-aware justification may significantly improve difficulty alignment.

**Risk**: Very low. Pure prompt engineering change. Easy to A/B test.

**Pilot**: Run on full 90-instance set, compare difficulty alignment scores with vs. without relation-type reasoning hints.

---

### Idea 6: Difficulty-Aware Distractor Generation for Event-Graph MCQs

**Type**: New direction
**Effort**: 2-3 weeks
**Venue target**: BEA Workshop / AIED 2026

**What**: Extend the pipeline from open-ended QG to MCQ generation:
1. Generate question + correct answer from event path (existing)
2. Generate 3 distractors by sampling alternative paths from the same event graph:
   - **Easy distractors**: events from a completely different part of the document
   - **Medium distractors**: events that are on alternative paths sharing some nodes
   - **Hard distractors**: events that are semantically similar or on parallel causal chains
3. Evaluate distractor quality via LLM judge and human eval

**Why novel**: Distractor generation is the hardest part of MCQ generation (per literature review). Using event graph structure to generate difficulty-calibrated distractors is unexplored.

**Risk**: Medium. Distractor quality is hard to guarantee. But even partial success is publishable at a workshop.

**Pilot**: 30 MCQ items, evaluate distractor plausibility and difficulty calibration.

---

### Idea 7: Cross-Dataset Difficulty Transfer (MAVEN-ERE → FairytaleQA)

**Type**: Generalization study
**Effort**: 2-3 weeks
**Venue target**: NLP Workshop / CCF-C

**What**: Test whether the 4-dim difficulty scoring framework generalizes to a different domain:
1. Apply the same graph-building + path-sampling + 4-dim scoring to FairytaleQA (which has narrative event structures and existing difficulty labels)
2. Measure correlation between 4-dim scores and FairytaleQA's human-annotated difficulty labels
3. If correlation is significant, this demonstrates domain-independent validity of the structural difficulty framework

**Why novel**: Generalization of structural difficulty metrics across domains is untested. FairytaleQA is the standard benchmark for narrative QG — showing cross-dataset validity strengthens the contribution.

**Risk**: Medium-High. FairytaleQA's event structure is different from MAVEN-ERE (narrative vs. news). May need adaptation. But negative result is still informative.

**Pilot**: Process 20 FairytaleQA stories, build event graphs, compute 4-dim scores, correlate with existing difficulty labels.

---

### Idea 8: Human Study — LLM-Generated vs. Human-Written Difficulty Perception

**Type**: Evaluation/HCI study
**Effort**: 2-3 weeks (mostly annotation time)
**Venue target**: BEA Workshop / AIED poster

**What**: A focused human study comparing:
1. LLM-generated questions at Easy/Medium/Hard (from existing pipeline)
2. Human-written questions on the same event paths (crowdsource)
3. Ask annotators to: (a) rank difficulty, (b) rate naturalness, (c) identify which is AI-generated
4. Measure: do humans perceive the same difficulty ordering as the 4-dim scorer?

**Why novel**: Most DCQG papers skip direct human difficulty perception studies. This would provide external validity evidence and could be a standalone workshop paper.

**Risk**: Medium. Requires ~$200-500 for crowdsourcing (e.g., Prolific). IRB may be needed depending on institution.

**Pilot**: 20 instances with 5 internal annotators as proof-of-concept.

---

### Idea 9: Lightweight Difficulty Prediction Without Generation

**Type**: New direction (regression task)
**Effort**: 1-2 weeks
**Venue target**: CCF-C short paper

**What**: Instead of generating questions, predict the difficulty distribution of questions that *could* be asked from a given event graph:
1. Build a lightweight classifier (RandomForest or small BERT) that takes event graph features (node count, edge type distribution, path length distribution, etc.) and predicts the difficulty profile
2. Evaluate: can the classifier predict which documents will yield Easy/Medium/Hard questions before generation?
3. Application: curriculum design — automatically identify passages suitable for different learner levels

**Why novel**: Most work focuses on generation. Difficulty *prediction* from graph structure is a different task with practical applications (content recommendation for adaptive learning).

**Risk**: Low. Simple ML task, quick to implement. Novelty is moderate but sufficient for a short paper.

**Pilot**: Extract features from 50 MAVEN-ERE documents, train classifier, evaluate precision/recall for difficulty level prediction.

---

### Idea 10: Difficulty-Controlled Question Generation as a Graph Traversal Problem

**Type**: Theoretical framing
**Effort**: 2-3 weeks
**Venue target**: CCF-C / NLP Workshop

**What**: Reframe DCQG as a constrained graph traversal problem:
1. Formalize difficulty dimensions as constraints on a walk through the event graph
2. Prove that Easy questions correspond to short walks with homogeneous edge types
3. Prove that Hard questions correspond to long walks with heterogeneous edge types crossing sentence boundaries
4. Use constrained random walk sampling instead of BFS-based path enumeration
5. This provides theoretical grounding for why the 4-dim scoring works

**Why novel**: No prior work provides a graph-theoretic formalization of question difficulty. This bridges NLP and graph theory.

**Risk**: Medium. Theoretical contribution alone may not be enough for a strong venue. Pair with experimental validation.

**Pilot**: Implement constrained random walk sampler, compare path distribution with BFS-based sampler.

---

## Comparison Matrix

| # | Idea | Novelty | Feasibility | Publishability | Builds on Existing | Risk |
|---|------|---------|-------------|----------------|---------------------|------|
| 1 | CrossQG-style filtering | Medium | High | High | Yes (direct extension) | Low |
| 2 | Learner-aware personalization | High | Medium | High | Partial | Medium |
| 3 | Counterfactual calibration | High | Medium | High | Yes | Medium |
| 4 | Multi-agent debate | Medium | High | Medium | Yes | Low |
| 5 | Relation-type-aware prompting | Medium | Very High | Medium | Yes (prompt change) | Very Low |
| 6 | Distractor generation | High | Medium | High | Partial | Medium |
| 7 | Cross-dataset transfer | High | Medium | Medium | Yes | Medium-High |
| 8 | Human perception study | Medium | Medium | Medium | Yes | Medium |
| 9 | Difficulty prediction | Medium | High | Medium | Partial | Low |
| 10 | Graph traversal formalization | High | Medium | High | Yes | Medium |

## Recommended Strategy

**Primary paper** (existing project + Ideas 1, 3, 5): The core event-graph DCQG paper with:
- 4-dim structural difficulty scoring ✓ (already implemented)
- Counterfactual difficulty calibration (Idea 3) as strong causal evidence
- Relation-type-aware prompting (Idea 5) as simple but effective improvement
- CrossQG-style consistency filtering (Idea 1) as post-generation quality control

**Standalone workshop paper** (Idea 2 or 6): Either learner-aware personalization or distractor generation as a separate submission.

**Short paper** (Idea 9): Difficulty prediction from graph structure — quick to execute, sufficient for CCF-C.

---

## Eliminated Ideas

None eliminated yet — all 10 ideas passed initial feasibility filter (no training required, uses public datasets, achievable within timeline).

---

## Novelty Check Results (Phase 3)

| Idea | Novelty | Closest Work | Differentiation |
|------|---------|-------------|-----------------|
| 1 (CrossQG filtering) | CONFIRMED (moderate) | CrossQG (EMNLP 2025) | First on event-graph QG |
| 2 (Learner-aware) | CONFIRMED (high) | SMART (EMNLP 2025), KAQG (2025) | No event graph + learner combo |
| 3 (Counterfactual) | CONFIRMED (moderate-high) | DiffQG (IP&M 2024) | Different KG + mechanism |

## Reviewer Assessment (Phase 4)

**Score**: 7/10 — solid workshop paper, needs strengthening for main conference

**Key weaknesses**: (1) arbitrary score calibration, (2) TEMPORAL dominance (86%), (3) single dataset, (4) weak EA dimension, (5) no SOTA DCQG baseline

**Mitigations applied** — see refined proposal.

## Refined Proposal (Phase 4.5)

**Problem anchor**: Event-graph structural features as sole difficulty signal for zero-shot QG

**Core + Extensions**:
- E1: Counterfactual calibration (causal evidence)
- E2: Relation-type-aware prompting (why-is-it-hard hints)
- E3: Consistency filtering (CrossQG-style post-generation)

**Venue strategy**:
- Primary: NLPCC / CCF-C NLP venue (core DCQG + E1 + E2)
- Workshop: BEA/AIED (Idea 2: learner-aware personalization)
- Short: CCF-C (Idea 9: difficulty prediction)

## Final Deliverables
- Full report: `idea-stage/IDEA_REPORT.md`
- Refined proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (existing)
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md` (existing)

## Next Steps
- [ ] Run experiments per `refine-logs/EXPERIMENT_PLAN.md`
- [ ] Implement E1 (counterfactual), E2 (prompting), E3 (filtering)
- [ ] Paper writing
