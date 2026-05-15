# Independent Difficulty Evaluation Report

**Input:** `outputs/runs/baseline_alignment_pilot/*_questions.filtered.jsonl`
**Judge Model:** gpt-4o-mini (AIHUBMIX)
**Items evaluated:** 62 (final_filter_pass=true)
**Balanced subset:** 24 (seed=42)
**Total API calls:** 124
**Parse errors:** 0
**Retries:** 0

**This evaluation is independent of the solver and does not use solver answers or solver correctness.**

## Method Comparison (Yield-Aware — All Valid Questions)

| Metric | PathQG-HardAware | ZeroShot-TargetQG | ICL-TargetQG | SelfRefine |
|--------|---:|---:|---:|---:|
| Judged | 36 | 11 | 7 | 8 |
| Judge OK | 36 | 11 | 7 | 8 |
| Diff Parse Errors | 0 | 0 | 0 | 0 |
| Path Parse Errors | 0 | 0 | 0 | 0 |
| Diff Accuracy | 47.2% | 72.7% | 71.4% | 62.5% |
| Spearman rho | 0.5564 | 0.6 | 0.5625 | N/A |
| Step Consistency | 47.2% | 72.7% | 71.4% | 62.5% |
| PathDep Strong | 11/36 (31%) | 1/11 (9%) | 1/7 (14%) | 0/8 (0%) |
| PathDep Strong+Partial | 11/36 (31%) | 1/11 (9%) | 1/7 (14%) | 0/8 (0%) |
| Answerable | 100.0% | 100.0% | 100.0% | 100.0% |
| FinalConsistent | 80.6% | 90.9% | 57.1% | 75.0% |

### Difficulty Confusion Matrix (Yield-Aware)


#### PathQG-HardAware

| Target \ Pred | Easy | Medium | Hard |
|---------------|-----:|-------:|-----:|
| Easy | 12 | 0 | 0 |
| Medium | 13 | 5 | 0 |
| Hard | 4 | 2 | 0 |

#### ZeroShot-TargetQG

| Target \ Pred | Easy | Medium | Hard |
|---------------|-----:|-------:|-----:|
| Easy | 8 | 1 | 0 |
| Medium | 2 | 0 | 0 |
| Hard | 0 | 0 | 0 |

#### ICL-TargetQG

| Target \ Pred | Easy | Medium | Hard |
|---------------|-----:|-------:|-----:|
| Easy | 5 | 1 | 0 |
| Medium | 1 | 0 | 0 |
| Hard | 0 | 0 | 0 |

#### SelfRefine

| Target \ Pred | Easy | Medium | Hard |
|---------------|-----:|-------:|-----:|
| Easy | 5 | 0 | 0 |
| Medium | 3 | 0 | 0 |
| Hard | 0 | 0 | 0 |

## Per-Difficulty Breakdown (Yield-Aware)

| Method | Target | N | Pred Easy | Pred Medium | Pred Hard | Diff Acc | Step Cons | PathDep Strong |
|--------|--------|--:|----------:|------------:|----------:|---------:|----------:|---------------:|
| PathQG-HardAware | Easy | 12 | 12 | 0 | 0 | 100% | 100% | 1 |
| PathQG-HardAware | Medium | 18 | 13 | 5 | 0 | 28% | 28% | 5 |
| PathQG-HardAware | Hard | 6 | 4 | 2 | 0 | 0% | 0% | 5 |
| ZeroShot-TargetQG | Easy | 9 | 8 | 1 | 0 | 89% | 89% | 1 |
| ZeroShot-TargetQG | Medium | 2 | 2 | 0 | 0 | 0% | 0% | 0 |
| ICL-TargetQG | Easy | 6 | 5 | 1 | 0 | 83% | 83% | 1 |
| ICL-TargetQG | Medium | 1 | 1 | 0 | 0 | 0% | 0% | 0 |
| SelfRefine | Easy | 5 | 5 | 0 | 0 | 100% | 100% | 0 |
| SelfRefine | Medium | 3 | 3 | 0 | 0 | 0% | 0% | 0 |

## Balanced Quality (Min-Count Per Difficulty)

**Seed:** 42
**Per-difficulty min counts:** Easy=5, Medium=1, Hard=0
**Excluded difficulties:** Hard (at least one method has 0 valid questions)

| Metric | PathQG-HardAware | ZeroShot-TargetQG | ICL-TargetQG | SelfRefine |
|--------|---:|---:|---:|---:|
| N balanced | 6 | 6 | 6 | 6 |
| Diff Accuracy | 83.3% | 83.3% | 66.7% | 83.3% |
| Spearman rho | N/A | N/A | 0.4857 | N/A |
| Step Consistency | 83.3% | 83.3% | 66.7% | 83.3% |
| PathDep Strong | 1/6 (17%) | 0/6 (0%) | 1/6 (17%) | 0/6 (0%) |
| Answerable | 100.0% | 100.0% | 100.0% | 100.0% |
| FinalConsistent | 100.0% | 100.0% | 66.7% | 83.3% |

## Key Observations

- Highest difficulty accuracy (yield-aware): **ZeroShot-TargetQG** (72.7%)
- Highest Spearman rho (yield-aware): **ZeroShot-TargetQG** (0.6)
- Highest path dependency strong (yield-aware): **PathQG-HardAware** (31%)
- Hard valid counts: PathQG-HardAware=6, ZeroShot-TargetQG=0, ICL-TargetQG=0, SelfRefine=0

## Interpretation

- **Valid yield advantage:** PathQG-HardAware produces more filter-passing questions (36 vs 7-11 for baselines).
- **Independent difficulty consistency:** Compare difficulty accuracy and Spearman across methods to assess whether PathQG's difficulty labels align with independent judge assessment.
- **Path dependency quality:** Strong path dependency indicates questions genuinely require understanding the event path, not just the final sentence.
- **Solver accuracy is not used here.** This evaluation is purely about difficulty prediction and path dependency.