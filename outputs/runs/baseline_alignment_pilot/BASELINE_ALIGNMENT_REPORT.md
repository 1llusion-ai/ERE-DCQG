# Baseline Alignment Pilot Report

**Input:** `outputs/runs/qg_pilot_strict_100_per_level/selected_paths.jsonl`
**Methods:** ZeroShot-TargetQG, ICL-TargetQG, SelfRefine, PathQG-HardAware
**Filter:** v3 quality filter pipeline
**Total paths:** 76

## Selection

| Difficulty | Selected |
|------------|---------:|
| Easy | 32 |
| Medium | 35 |
| Hard | 9 |
| **Total** | **76** |

## Method Comparison

| Metric | PathQG-HardAware | ZeroShot-TargetQG | ICL-TargetQG | SelfRefine |
|--------|---:|---:|---:|---:|
| Generated | 73 (96%) | 44 (58%) | 42 (55%) | 41 (54%) |
| Filter Pass | 36 (47%) | 11 (14%) | 7 (9%) | 8 (10%) |
| Path Coverage Fails | 31 | 62 | 69 | 67 |
| Answer Consistency Fails | 14 | 43 | 42 | 40 |
| Filter Exceptions | 0 | 0 | 0 | 0 |
| Judge JSON Errors | 8 | 2 | 6 | 6 |
| Solver Correct | 12/36 | 2/11 | 4/7 | 0/8 |

### Per-Difficulty Filter Pass

| Difficulty | PathQG-HardAware | ZeroShot-TargetQG | ICL-TargetQG | SelfRefine |
|------------|---:|---:|---:|---:|
| Easy | 12/32 | 9/32 | 6/32 | 5/32 |
| Medium | 18/35 | 2/35 | 1/35 | 3/35 |
| Hard | 6/9 | 0/9 | 0/9 | 0/9 |

### Per-Difficulty Solver Correct

| Difficulty | PathQG-HardAware | ZeroShot-TargetQG | ICL-TargetQG | SelfRefine |
|------------|---:|---:|---:|---:|
| Easy | 4/12 | 2/9 | 4/6 | 0/5 |
| Medium | 6/18 | 0/2 | 0/1 | 0/3 |
| Hard | 2/6 | 0/0 | 0/0 | 0/0 |

## Key Observations

- Highest filter pass: **PathQG-HardAware** (36/76, 47%)
- Hard filter pass: PathQG-HardAware=6/9, ZeroShot-TargetQG=0/9, ICL-TargetQG=0/9, SelfRefine=0/9

## Conclusion

- This is a baseline alignment pilot, NOT the final main experiment.
- solver_correct is auxiliary — judge calibration still needed.
- Primary comparison: filter pass, path coverage, answer consistency, difficulty consistency.