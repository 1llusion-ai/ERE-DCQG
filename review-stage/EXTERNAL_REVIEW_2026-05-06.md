# DCQG External Research Review

**Date**: 2026-05-06
**Reviewer**: GPT-5.4 (xhigh reasoning via Codex MCP)
**Review Type**: Multi-round critical review (3 rounds)
**Target Venue**: CCF C (COLING / AACL / EMNLP Findings-tier)

---

## Round 1: Mock Review

### Summary
The paper proposes event-path-constrained question generation over MAVEN-ERE, using path hop count as a proxy for Easy/Medium/Hard reasoning difficulty. The system improves filtered generation yield, but the central difficulty-control claim is not supported. Independent difficulty evaluation shows zero Hard questions judged as Hard, Medium mostly collapses to Easy, and solver-based difficulty is non-monotonic.

### Score: 4 / 10 | Confidence: 4 / 5

### Strengths
- The event-path framing is a reasonable structural prior for document-level QG.
- The traceable path constraint is practically useful: it gives debuggability and a clearer provenance story than pure prompt-based QG.
- The filtering pipeline and shortcut checks are valuable engineering contributions.
- The ablation suggests paths matter: PathOnly fails badly, while RelationType without explicit paths behaves differently.
- The paper is honest about failure modes, which could become a strong diagnostic contribution.

### Weaknesses
- The main claim fails. Hop count does not currently control perceived difficulty. Hard questions are never judged Hard.
- "Hard" path availability is too small. A strict Hard pool of 9 paths cannot support a credible difficulty-control paper.
- PathQG-HardAware is worse than ICLTargetQG on solver correctness and composite score.
- Solver correctness is hard to interpret without human evaluation.
- The independent judge sample is tiny, but the direction is devastating: all Hard items collapse to Easy/Medium.
- The hard-question template failure mode shows that path dependence is not the same as reasoning difficulty.
- Same-family generator/judge creates circularity risk.
- Single dataset and single generator make the claims fragile.

### Questions for Authors
- What exactly does "pass" measure, and why should it matter more than answerability or solver correctness?
- Are Hard questions genuinely multi-hop, or merely mentioning multiple events?
- How often can the answer be recovered from a single sentence or local cue?
- Did humans verify that generated questions require the intended event path?
- Are failures caused by the generator, the path selection, the difficulty definition, or the filters?
- Why is Medium solver accuracy higher than Easy?
- How stable are results across different generators and judges?

### What Would Move Toward Accept
- Drop or soften the strong difficulty-control claim.
- Add human difficulty/path-use evaluation.
- Demonstrate that path-grounded questions require more evidence or relation composition than baselines.
- Build a larger Hard pool.
- Replace hop count with a learned or composite difficulty predictor.
- Show that PathQG provides something baselines do not: traceability, controllable evidence coverage, lower shortcut rate, or better dataset construction yield.

---

## Round 2: Reframing Strategy

### Reframed Title
> Event-Path Constraints for Traceable Document-Level Question Generation: Evidence Control and the Limits of Hop-Based Difficulty

### Reframed Core Claim
> Event paths provide a useful structural interface for generating traceable document-level questions, but hop count alone is insufficient for reliable difficulty control.

### Claims to Remove
- "We achieve difficulty-controlled QG."
- "Hard questions require 3-hop reasoning."
- "PathQG-HardAware improves question quality" (unless new metrics prove it)

### Claims to Emphasize
- Path constraints improve traceability.
- Path constraints improve evidence coverage / path dependency.
- The framework enables debuggable generation.
- Hop-count difficulty fails in systematic, analyzable ways.
- The paper provides a diagnostic benchmark/protocol for future document-level difficulty-controlled QG.

### Minimum Experiment Package

**MUST-DO (fatal if missing):**

1. **Independent path-dependency evaluation** (150-200 items)
   - Labels: answerable, requires intended path, answerable from single sentence, uses multiple evidence sentences, shortcut exists
   - Can use GPT-4o-mini + small manual audit

2. **Human/manual audit** (60-90 items minimum)
   - Even one expert annotator is better than zero
   - Labels: answerable, difficulty, path required, shortcut

3. **Same-filter baseline comparison**
   - Report: raw generation, pass rate, final retained, answerability, path dependency, shortcut rate, solver correctness

4. **Difficulty failure analysis**
   - Categorize: single-sentence shortcut, temporal-template triviality, answer after final event cue, path mentioned but not needed, relation chain weak, generator drops constraint

5. **Statistical uncertainty**
   - Confidence intervals, bootstrap or binomial CIs, significance tests

**NICE-TO-DO:**
- Cross-generator test (GPT-4o-mini, 50-100 items)
- Cross-judge comparison (Qwen vs GPT-4o-mini, 100-150 items)
- More Hard path mining with relaxed constraints
- Evidence-distance analysis
- Error examples table (6-10 qualitative cases)

### How to Handle Hard Failure
Analyze it. Do not hide it.

> We find that 3-hop event paths increase structural complexity but do not reliably induce perceived reasoning difficulty. Most generated Hard-target questions collapse into sequence-completion questions answerable from local textual cues.

### Realistic Acceptance Probability
- Current form: **10-20%**
- Reframed with minimum experiments: **25-40%**

---

## Round 3: Claims Matrix and Paper Outline

### Results-to-Claims Matrix

| Experiment Outcome | Allowed Claim | Forbidden Claim |
|---|---|---|
| **E1 strong positive**: PathQG path-dependency clearly higher (31% vs 9-14%), significant | "Event-path constraints improve traceable/path-dependent document-level QG" | "Better questions overall" unless quality also improves |
| **E1 moderate positive**: PathQG higher, small margin | "Path constraints show evidence of improved path grounding" | "Substantial improvement" |
| **E1 null/negative**: PathQG not better | "Hop/path prompting alone does not reliably induce path dependency" (paper in danger) | Any traceability claim |
| **E2 strong positive**: Human difficulty matches, monotonic | "Hop count provides a useful coarse difficulty signal" | "Precise difficulty control" |
| **E2 partial**: Easy works, Medium/Hard collapse | "Hop count weakly separates simple questions but fails for higher difficulty" | "Difficulty-controlled QG" |
| **E2 negative**: No alignment | "Structural path complexity does not imply perceived difficulty" (key negative finding) | Any difficulty controllability claim |
| **E3 high agreement**: Qwen and GPT-4o-mini agree | "LLM-based evaluation is reasonably stable across judge families" | "Equivalent to human evaluation" |
| **E3 mixed**: Agreement on answerability, weak on difficulty | "Difficulty judging remains unstable; answerability/path-use are more reliable" | Strong claims based only on LLM difficulty scores |
| **E4 clear taxonomy**: Hard failures cluster into interpretable categories | "We identify systematic failure modes explaining why hop-count difficulty fails" | "We solve hard question generation" |
| **E4 messy**: No pattern | "Hard generation failures are heterogeneous and remain unresolved" | "Diagnostic framework" as major contribution |

**Best claim combination**: E1 strong + E2 partial/negative + E3 moderate + E4 clear taxonomy

### Paper Outline (8-page COLING/AACL format)

| Section | Pages | Description |
|---|---:|---|
| Abstract | 0.2 | Traceable document-level QG + hop-count limits. One positive, one negative result. |
| 1. Introduction | 0.8 | Document-level QG needs inspectable evidence. Event paths as structure. Hop-count as hypothesis. |
| 2. Related Work | 0.5 | Document-level QG, controllable QG, multi-hop QA/QG, event graphs, difficulty estimation. |
| 3. Task Formulation | 0.8 | Define document, event graph, path, target level, trace. Separate evidence controllability from difficulty alignment. |
| 4. Method | 1.0 | Graph construction, path sampling, prompt construction, generation, repair, filtering. Emphasize trace preservation. |
| 5. Setup | 0.8 | Dataset, sample, baselines, models, judges, metrics. Reveal small Hard pool. |
| 6. Main Results | 1.3 | **Positive story first**: pass rate, answerability, path-dependency, multi-sentence evidence, shortcut rate. |
| 7. Difficulty Limits | 1.1 | Human audit + independent judge. Difficulty confusion matrix. Hard collapse. |
| 8. Hard Failure Taxonomy | 0.9 | Systematic failure modes with examples. Turns negative result into contribution. |
| 9. Discussion | 0.4 | Future difficulty control: hop + evidence distance + relation type + shortcut detection + ambiguity. |
| 10. Limitations + Conclusion | 0.6 | Honest limitations. Balanced contribution restatement. |
| **Total** | **~8.4** | Compress Related Work, Discussion to fit 8 pages. |

### Key Figures/Tables Plan
- **Fig 1**: Pipeline diagram (document → event graph → path → question → trace)
- **Fig 2**: Motivating example (1-hop/2-hop/3-hop paths with generated questions)
- **Tab 1**: Dataset/path statistics (reveal Hard pool size)
- **Tab 2**: Main results (methods × metrics, path-dependency first)
- **Tab 3**: Difficulty confusion matrix
- **Tab 4**: Hard failure taxonomy
- **Tab 5**: Ablation results
- **Fig 3**: Path-dependency bar chart by method

---

## Reviewer Consensus

The idea is worth saving, but not as currently claimed. The honest paper is about the promise and failure modes of event-path-based control, with traceability as the main contribution and difficulty control as an open problem.

**Bottom line**: With the minimum experiment package (path-dependency eval, small human audit, same-filter baselines, Hard failure analysis, statistical CIs), this becomes a plausible CCF C submission. Without path-dependency evidence and at least a small manual audit, expect rejection.
