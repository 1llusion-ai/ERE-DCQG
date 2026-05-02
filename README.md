# DCQG: Difficulty-Controlled Question Generation via Event-Path Constraints

<p align="center">
  <b>Target-event-grounded question generation with document-level event relation paths for controllable reasoning difficulty</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/pipeline-end--to--end-brightgreen.svg" alt="Pipeline">
</p>

---

## Overview

**DCQG** is a training-free pipeline for generating questions at specified difficulty levels (Easy / Medium / Hard) using document-level event relation graphs. Given a document and a target final event, the system constructs an event graph, samples difficulty-labeled event paths by hop count, and generates questions that require multi-hop reasoning along those paths.

**Core idea**: Question difficulty can be controlled by the *structure* of the event path — not by prompt engineering alone. A 1-hop path yields an Easy question; a 3-hop path yields a Hard question that forces the solver to trace a causal or temporal chain across multiple sentences.

```
Document  ──►  Event Graph  ──►  Path Sampling (hop-based)  ──►  Prefilter  ──►  LLM Path Judge
                                                                              │
                                                                              ▼
Solver Eval  ◄──  Quality Filter  ◄──  PathQG-HardAware Generation  ◄────────┘
```

## Highlights

- **Zero-shot, training-free** — no fine-tuning; uses off-the-shelf LLMs via OpenAI-compatible APIs
- **Hop-based difficulty control** — Easy (1-hop), Medium (2-hop), Hard (3-hop) from directed event paths
- **Difficulty-aware prompting** — Hard questions must explicitly bind 2+ prior events; shortcut phrases are banned and repaired
- **Full-chain trace debugging** — every pipeline stage is traced end-to-end for reproducible diagnosis
- **Modular architecture** — 7 subpackages (`graph`, `path`, `generation`, `question_filter`, `evaluation`, `tracing`, `utils`) with clean dependency boundaries

## Installation

```bash
git clone https://github.com/<your-username>/DCQG.git
cd DCQG
```

**Python 3.10+** is required. No third-party dependencies — the pipeline uses only the Python standard library (`urllib.request`, `json`, `re`, etc.).

### API Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# SiliconFlow (used for generation, solver, and judges)
SILICONFLOW_API_URL=https://api.siliconflow.cn/v1/chat/completions
SILICONFLOW_API_KEY=your_key_here
MODEL=Qwen/Qwen2.5-7B-Instruct
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct

# AiHubMix (used for LLM path filtering)
AIHUBMIX_API_URL=https://aihubmix.com/v1/chat/completions
AIHUBMIX_KEY=your_key_here
AIHUBMIX_MODEL=gpt-4o-mini
```

All endpoints must be **OpenAI-compatible** (`/v1/chat/completions`). You can substitute any compatible provider (OpenAI, vLLM, Ollama, etc.).

### Data

Download the [MAVEN-ERE](https://github.com/thunlp/MAVEN-ERE) dataset and place the raw data at:

```
data/raw/maven_ere/valid.jsonl
```

Each line should be a JSON document with `id`, `title`, `sentences`, `events`, and relation fields (`causal_relations`, `temporal_relations`, `subevent_relations`).

## Quick Start

### End-to-End Pipeline

Run the full pipeline on 50 documents (skip expensive stages for speed):

```bash
python -m scripts.run_pipeline \
  --raw_data data/raw/maven_ere/valid.jsonl \
  --output_dir outputs/runs/quick_start \
  --limit 50 --skip_path_judge --skip_solver
```

### Smoke Test (3 items, full chain)

```bash
python -m scripts.run_smoke_test --limit 3
```

### Individual Stages

```bash
# Stage 1: Build event graphs
python -m scripts.01_build_graph --input data/raw/maven_ere/valid.jsonl --limit 10

# Stage 2: Sample difficulty-labeled paths
python -m scripts.02_sample_paths --input data/raw/maven_ere/valid.jsonl --limit 100

# Stage 3: Filter paths (prefilter + LLM judge)
python -m scripts.03_filter_paths --input outputs/runs/latest/paths.raw.jsonl --skip_llm_judge

# Stage 4: Generate questions
python -m scripts.04_generate_questions --input outputs/runs/latest/paths.filtered.jsonl

# Stage 5: Evaluate (quality filter + solver + judge)
python -m scripts.05_evaluate --input outputs/runs/latest/questions.raw.jsonl --skip_solver
```

## Pipeline Details

### Stage 1 — Event Graph Construction

Builds a directed event graph per document. Nodes are event mentions (with trigger word, type, sentence ID). Edges are typed relations (CAUSE, TEMPORAL, SUBEVENT) with subtypes (e.g., `CAUSE/PRECONDITION`, `TEMPORAL/BEFORE`). Only outgoing edges are followed for path sampling.

### Stage 2 — Path Sampling

Samples paths of length 1, 2, and 3+ hops from the event graph, labeled as Easy, Medium, and Hard respectively. Each path is enriched with:
- Supporting sentences (path event sentences ± 1 sentence window)
- Answer phrase extraction (clause-aware, from the final event's sentence)
- Relation subtype sequence

### Stage 3 — Path Filtering

**Deterministic prefilter** checks:
- Weak/broad final triggers (e.g., "happened", "occurred")
- Answer phrase validity (complete clause, not truncated)
- Single-sentence risk (Hard questions must span multiple sentences)
- Relation composition (temporal-only paths flagged)

**LLM path judge** (optional, gpt-4o-mini) evaluates whether the path supports a meaningful multi-hop question.

### Stage 4 — Question Generation (PathQG-HardAware)

Difficulty-specific prompts with few-shot examples and hard constraints:
- **Easy**: 1-hop, simple "what happened after X?" pattern
- **Medium**: must reference 1 prior event, connect 2+ sentences
- **Hard**: must explicitly mention 2+ prior events; banned shortcut phrases ("final outcome", "what happened after the incident")

Failed questions are automatically repaired with targeted fix prompts. Path binding is verified lexically (stem matching + substring containment).

### Stage 5 — Evaluation

**Quality filter pipeline** (5 sequential filters):
1. Grammar filter (question mark, valid start word, no repetition)
2. Weak trigger check
3. Answer phrase validation
4. Path coverage (lexical overlap between question and path events)
5. Hard degradation check (can the Hard question be answered from a single sentence?)

**Solver + Judge evaluation**:
- **Solver** answers the question from context (Qwen2.5-32B)
- **LLM Judge** scores: answerability, solver correctness, support coverage, fluency, path relevance, difficulty alignment

## Results

Evaluation on 300 fixed samples (100 per difficulty level, seed=42) from MAVEN-ERE valid split. Generator: Qwen2.5-7B-Instruct. Solver/Judge: Qwen2.5-32B-Instruct.

| Method | Pass% | Answerable | Solver Correct | Difficulty Alignment |
|:-------|------:|-----------:|---------------:|---------------------:|
| ZeroShotTargetQG | 42.3% | 0.921 | 0.325 | 0.747 |
| ICLTargetQG | 45.0% | 0.874 | 0.363 | 0.728 |
| SelfRefine | 47.7% | 0.923 | 0.308 | 0.729 |
| **PathQG-HardAware** | **62.0%** | 0.747 | 0.274 | **0.811** |

**Ablation** (component contribution):

| Variant | Answerable | Solver Correct |
|:--------|-----------:|---------------:|
| Full (path + context) | 0.747 | 0.274 |
| PathOnly (no context) | 0.285 | — |
| RelationType (no specific path) | — | 0.179 |

Key findings:
- PathQG-HardAware achieves the **highest pass rate** (62%) and **best difficulty alignment** (0.811)
- Removing context collapses answerability (0.747 → 0.285)
- Removing specific path structure collapses solver correctness (0.274 → 0.179)
- ICL-TargetQG achieves highest solver correctness (0.363), indicating room for improvement

## Project Structure

```
DCQG/
├── dcqg/                          # Main Python package
│   ├── graph/                     # Event graph construction
│   │   └── event_graph.py
│   ├── path/                      # Path sampling, filtering, validation
│   │   ├── sampler.py             # Hop-based difficulty sampling
│   │   ├── answer_extraction.py   # Clause-aware answer phrase extraction
│   │   ├── diagnostics.py         # Deterministic prefilter
│   │   ├── selector.py            # Answer phrase validation
│   │   ├── llm_filter.py          # LLM path quality judge
│   │   └── direction.py           # Path binding check, Hard validation
│   ├── generation/                # Question generation
│   │   ├── generator.py           # PathQG-HardAware with retry
│   │   ├── prompts.py             # Difficulty-aware few-shot prompts
│   │   ├── baselines.py           # ZeroShot, ICL, SelfRefine, ablations
│   │   ├── parser.py              # LLM response parsing
│   │   ├── repair.py              # Failed question repair prompts
│   │   └── faithfulness.py        # Path-faithfulness judge
│   ├── question_filter/           # Post-generation quality filters
│   │   ├── grammar.py             # Grammar + structural checks
│   │   ├── consistency.py         # Answer-event consistency judge
│   │   ├── path_coverage.py       # Lexical path coverage
│   │   ├── shortcut.py            # Hard degradation detection
│   │   └── pipeline.py            # Unified filter pipeline
│   ├── evaluation/                # Solver and judge
│   │   ├── solver.py              # Question answering solver
│   │   ├── judge.py               # LLM judge v2 (3-way scoring)
│   │   ├── metrics.py             # Fair metrics computation
│   │   └── report.py              # Comparison table formatting
│   ├── tracing/                   # Full-chain debug trace
│   │   ├── record.py              # TraceRecord data structure
│   │   ├── writer.py              # JSONL trace writer
│   │   └── render.py              # Readable markdown trace
│   └── utils/                     # Shared utilities
│       ├── config.py              # .env loader
│       ├── api_client.py          # OpenAI-compatible API client
│       ├── text.py                # Stemming, normalization, fuzzy match
│       └── jsonl.py               # JSONL I/O
├── scripts/                       # Entry points
│   ├── 01_build_graph.py          # Stage 1: graph construction
│   ├── 02_sample_paths.py         # Stage 2: path sampling
│   ├── 03_filter_paths.py         # Stage 3: path filtering
│   ├── 04_generate_questions.py   # Stage 4: question generation
│   ├── 05_evaluate.py             # Stage 5: evaluation
│   ├── run_smoke_test.py          # End-to-end smoke test
│   ├── run_quality_pilot.py       # Quality-focused pilot
│   └── run_pipeline.py            # Full pipeline orchestrator
├── data/                          # Dataset (not in repo)
│   └── raw/maven_ere/
├── outputs/                       # Experiment outputs (not in repo)
├── .env.example                   # API configuration template
└── README.md
```

## Baselines

The repository includes implementations of the following baselines and ablations:

| Method | Description |
|:-------|:------------|
| **ZeroShotTargetQG** | Context + target answer + difficulty definition, no examples |
| **ICLTargetQG** | Context + target answer + difficulty + few-shot examples |
| **SelfRefine** | ZeroShot + critique + revise (3 API calls per item) |
| **PathQG-HardAware** | Event path + context + difficulty-aware constraints (ours) |
| PathOnlyQG | Event path only, no context (ablation) |
| RelationTypeQG | Context + relation types, no specific path (ablation) |

Run baselines:

```bash
python -m scripts.run_quality_pilot --n_per_level 30
```

## Debugging

Every pipeline stage writes trace data. For any unexpected result, inspect the trace in this order:

```
raw sentence/event/offset
 → graph node and directed edge
 → sampled path and relation direction
 → answer phrase extraction
 → prefilter result
 → LLM path judge prompt/raw/parsed
 → QG prompt/raw/repair
 → quality filter
 → solver answer and judge raw response
```

Traces are written to `outputs/<run>/traces/`:
- `full_trace.jsonl` — one JSON line per item, all fields
- `readable_trace.md` — human-readable markdown with expandable prompt/response blocks

## Citation

If you find this work useful, please cite:

```bibtex
@misc{dcqg2026,
  title={Difficulty-Controlled Question Generation via Event-Path Constraints},
  author={},
  year={2026},
  howpublished={\url{https://github.com/<your-username>/DCQG}}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [MAVEN-ERE](https://github.com/thunlp/MAVEN-ERE) for the document-level event relation dataset
- [CrossQG](https://aclanthology.org/2025.findings-emnlp.151/) for difficulty-aware QG inspiration
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) for the open-weight language models
