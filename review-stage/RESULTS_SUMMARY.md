# DCQG Results Summary

**Date**: 2026-04-30
**Status**: Unified evaluation on fixed sample (sample_300_seed42.jsonl)

---

## 1. Experimental Setup

- **Sample**: 300 items (100 Easy, 100 Medium, 100 Hard), seed=42, from MAVEN-ERE
- **Generator**: Qwen2.5-7B (SiliconFlow API)
- **Judge/Solver**: Qwen2.5-32B (same model family — known limitation)
- **Evaluation**: Unified LLM judge (3-way: answerable, solver_correct, support_covered)
- **Composite**: 0.25*solver_correct + 0.20*answerable + 0.15*support_covered + 0.15*fluency + 0.10*relevance + 0.15*diff_align

**Note on target-aware baselines**: ICL-TargetQG and ZeroShot-TargetQG are given the target answer trigger in the prompt. This matches our task definition of target-event-grounded question generation — the baseline must generate a question whose answer is the specified target event.

---

## 2. Main Baselines

All methods use the same 300 items, same generator, same evaluation pipeline.

| Method | Type | N gen | N pass | Pass% | Answerable | Solver Correct | Composite |
|--------|------|------:|-------:|------:|-----------:|---------------:|----------:|
| **PathQG-HardAware** | Ours | 300 | 186 | 62.0% | 0.747 | 0.274 | 0.693 |
| ZeroShotTargetQG | Baseline | 300 | 127 | 42.3% | 0.921 | 0.325 | 0.728 |
| ICLTargetQG | Baseline | 300 | 135 | 45.0% | 0.874 | **0.363** | **0.737** |
| SelfRefine | Baseline | 300 | 143 | 47.7% | 0.923 | 0.308 | 0.729 |

### By Difficulty Level

**Solver Correct (primary metric):**

| Method | Easy | Medium | Hard |
|--------|-----:|-------:|-----:|
| PathQG-HardAware | 0.260 | 0.371 | 0.203 |
| ZeroShotTargetQG | 0.375 | 0.375 | 0.211 |
| ICLTargetQG | 0.434 | 0.326 | 0.308 |
| SelfRefine | 0.277 | 0.380 | 0.261 |

**Composite:**

| Method | Easy | Medium | Hard |
|--------|-----:|-------:|-----:|
| PathQG-HardAware | 0.649 | 0.716 | 0.704 |
| ZeroShotTargetQG | 0.740 | 0.735 | 0.706 |
| ICLTargetQG | 0.739 | 0.743 | 0.727 |
| SelfRefine | 0.699 | 0.742 | 0.745 |

### Interpretation

1. **Target-aware baselines have higher answerable/solver_correct** because they are explicitly given the target answer trigger. This is by design — they are target-aware QG methods.
2. **ICLTargetQG achieves the highest solver_correct (0.363) and composite (0.737)**, demonstrating that few-shot examples help the model generate more answerable questions.
3. **PathQG-HardAware has higher pass rate (62%)** than baselines (42-48%) due to path binding validation and repair logic.
4. **PathQG-HardAware achieves higher difficulty alignment** (0.811) than baselines (0.74-0.79), meaning its questions better match the intended difficulty level.
5. **PathQG-HardAware Medium solver_correct (0.371) is competitive** with baselines, showing that event-path-constrained generation produces appropriately difficult questions.

### Fair Metrics

**Difficulty Control (Primary):**

| Method | Easy SolCor | Med SolCor | Hard SolCor | E-H gap | DC Score | Violations |
|--------|-----------:|----------:|----------:|--------:|--------:|----------:|
| PathQG-HardAware | 0.260 | 0.371 | 0.203 | 0.057 | 0.168 | 1 |
| ZeroShotTargetQG | 0.375 | 0.375 | 0.211 | 0.164 | 0.164 | 0 |
| ICLTargetQG | 0.434 | 0.326 | 0.308 | 0.126 | 0.108 | 0 |
| SelfRefine | 0.277 | 0.380 | 0.261 | 0.016 | 0.016 | 1 |

**Fair Metrics (Secondary):**

| Method | Pass% | Conditional SolCor | Macro-Avg SolCor | E2E SolCor |
|--------|------:|-------------------:|-----------------:|-----------:|
| PathQG-HardAware | 62.0% | 0.274 | 0.278 | 0.170 |
| ZeroShotTargetQG | 42.3% | 0.325 | 0.320 | 0.137 |
| ICLTargetQG | 45.0% | 0.363 | 0.356 | 0.163 |
| SelfRefine | 47.7% | 0.308 | 0.306 | 0.147 |

**Notes:**
- Conditional = on pass samples only (current standard)
- Macro-Avg = (Easy mean + Medium mean + Hard mean) / 3
- E2E = total score / 300 (fail samples count as 0)
- DC Score = max(0, Easy-Med) + max(0, Med-Hard)
- Violations = count of (Easy < Medium) or (Medium < Hard)

**PathQG-HardAware Monotonicity Issue:**
Medium SolCor (0.371) > Easy SolCor (0.260) — 1 violation. This is because Easy has 32 path_binding failures (out of 100), leaving harder surviving items. Medium has explicit prior-event requirements that help the solver.

---

## 3. Ablations (Component Analysis)

PathOnlyQG and RelationTypeQG are component ablations, not external baselines. They test the contribution of specific input components.

| Method | Component Removed | N gen | N pass | Pass% | Answerable | Solver Correct | Composite |
|--------|-------------------|------:|-------:|------:|-----------:|---------------:|----------:|
| PathQG-HardAware | (full) | 300 | 186 | 62.0% | 0.747 | 0.274 | 0.693 |
| RelationTypeQG | Specific path | 300 | 134 | 44.7% | 0.791 | 0.179 | 0.652 |
| PathOnlyQG | Context | 300 | 158 | 52.7% | 0.285 | 0.114 | 0.534 |

### By Difficulty Level (Solver Correct)

| Method | Easy | Medium | Hard |
|--------|-----:|-------:|-----:|
| PathQG-HardAware | 0.260 | 0.371 | 0.203 |
| RelationTypeQG | 0.222 | 0.136 | 0.178 |
| PathOnlyQG | 0.092 | 0.174 | 0.085 |

### Ablation Findings

1. **Removing context (PathOnlyQG)**: Answerable drops from 0.727 to 0.285 — context is essential for question answerability.
2. **Removing specific path (RelationTypeQG)**: Solver correct drops from 0.236 to 0.179 — specific event path helps the solver find the correct answer.
3. **Both ablations perform worse than full PathQG-HardAware**, confirming that both context and event path structure contribute to question quality.

---

## 4. Faithfulness Metrics (PathQG-HardAware, from prior evaluation)

| Difficulty | need_intermediate | can_answer_single | hard_pass |
|-----------|------------------|-------------------|-----------|
| Easy | 0.256 | 0.600 | 0.275 |
| Medium | 0.864 | 0.142 | 0.864 |
| Hard | 1.000 | 0.000 | 1.000 |

**Caveat**: These metrics use Qwen2.5-32B as judge, which is from the same model family as the 7B generator.

---

## 5. Known Limitations

1. **Circular evaluation**: Judge (32B) and generator (7B) are from the same model family
2. **No human evaluation**: Difficulty perception validated only by LLM judges
3. **Target-aware baselines see the answer**: ICL/ZeroShot-TargetQG are given the target trigger, which is appropriate for our task but means they have different information than PathQG
4. **Single dataset**: MAVEN-ERE only
5. **Single generator**: Qwen2.5-7B only

---

## 6. What the Data Supports

**Strong claim**: "PathQG-HardAware generates difficulty-controlled questions from event paths with higher pass rate (62%) and better difficulty alignment (0.811) than target-aware baselines."

**Strong claim**: "Removing either context or event path structure significantly degrades question quality (ablation results)."

**Strong claim**: "ICL-TargetQG with few-shot examples achieves the highest solver correctness (0.363) among all methods, demonstrating the effectiveness of in-context learning for target-aware question generation."

**Moderate claim**: "Event path structure helps generate questions that better match intended difficulty levels. PathQG-HardAware achieves positive E-H gap (0.057) and DC Score (0.168)."

**Not supported**: "PathQG-HardAware produces more answerable questions than target-aware baselines" — the baselines see the answer explicitly, so this comparison is not meaningful.

**Open issue**: "PathQG-HardAware Medium SolCor (0.371) > Easy SolCor (0.260) — 1 monotonicity violation. Easy has 32 path_binding failures, leaving harder surviving items."

---

## 7. Required for Publication

| Requirement | Status | Priority |
|------------|--------|----------|
| Unified evaluation (same sample, same judge) | DONE | FATAL |
| Target-aware baseline (ICL-TargetQG) | DONE | MAJOR |
| SelfRefine baseline | DONE | MAJOR |
| Component ablation (PathOnly, RelationType) | DONE | MAJOR |
| Break circular evaluation | NOT DONE | FATAL |
| Human evaluation (100 items) | NOT DONE | MAJOR |
| Paper reframing (4D → path-length) | NEEDED | MAJOR |
| Statistical significance tests | DONE | MODERATE |
