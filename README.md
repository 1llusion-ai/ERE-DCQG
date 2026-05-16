# DCQG: Evidence-Necessity-Aware Difficulty-Controlled Question Generation

<p align="center">
  <b>Difficulty control via evidence-inference necessity — not by surface features or graph hop count</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/pipeline-end--to--end-brightgreen.svg" alt="Pipeline">
</p>

---

## Overview

**DCQG** is a pipeline for generating reading comprehension questions whose answering difficulty (Easy / Medium / Hard) is **controllable**, where difficulty is defined by a two-dimensional combination of answer explicitness and necessary evidence scope — grounded in established reading comprehension theory (Kintsch, Coh-Metrix).

Unlike prior work that defines difficulty via surface features (question length, word count) or structural features (graph hop count), DCQG operationalizes difficulty through **evidence-inference load**: how many sentences must a reader synthesize, and must the answer be directly found or inferred?

```
FairytaleQA Data
       │
       ▼
Stage 1: Evidence Audit (No-Vote Pipeline)
  Selector → Blind Verifier → Removal Verifier
       │
       ▼
Stage 2: Multi-Task Classifier Training
  DeBERTa-v3 (difficulty head + evidence head)
       │
       ▼
Stage 3: QG + Reranking
  4 methods × K=5 candidates → classifier rerank
       │
       ▼
Stage 4: Evaluation
  Human eval (primary) + classifier consistency + LLM judge
```

## Difficulty Definition

| Difficulty | Definition |
|:-----------|:-----------|
| **Easy** | The answer can be directly found in the text; obtaining the answer requires relying on only one necessary evidence sentence. |
| **Medium** | **Case 1:** The answer cannot be directly found in the text; obtaining the answer requires relying on one necessary evidence sentence and making a simple inference. **Case 2:** The answer can be directly found in the text; however, obtaining the answer requires synthesizing information from multiple necessary evidence sentences. |
| **Hard** | The answer cannot be directly found in the text; obtaining the answer requires synthesizing information from multiple necessary evidence sentences and making at least one inference. |

Canonical source: `dcqg/difficulty/definitions.py`.

## Highlights

- **Evidence-necessity difficulty**: Two-axis definition (answer explicitness × evidence scope) grounded in reading comprehension theory
- **Automated evidence audit**: No-Vote pipeline (Selector → Blind Verifier → Removal Verifier) produces evidence-necessity labels without human annotation
- **Interpretable classifier**: Multi-task DeBERTa-v3 jointly predicts difficulty AND identifies required evidence sentences
- **Dual-use design**: Same classifier serves as both evaluation instrument and generation-time reranker
- **No learner data required**: Unlike IRT-based methods, the pipeline works from existing QA datasets

## Installation

```bash
git clone https://github.com/1llusion-ai/ERE-DCQG.git
cd ERE-DCQG
```

**Python 3.10+** is required.

### API Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# SiliconFlow (used for evidence audit, generation, and judges)
SILICONFLOW_API_URL=https://api.siliconflow.cn/v1/chat/completions
SILICONFLOW_API_KEY=your_key_here
MODEL=Qwen/Qwen3-32B
JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct

# AiHubMix (used for classifier evaluation and diagnostics)
AIHUBMIX_API_URL=https://aihubmix.com/v1/chat/completions
AIHUBMIX_API_KEY=your_key_here
AIHUBMIX_MODEL=gpt-4o-mini
```

All endpoints must be **OpenAI-compatible** (`/v1/chat/completions`).

### Data

The pipeline uses [FairytaleQA](https://github.com/WorkInTheDark/FairytaleQA) (Xu et al., 2022), a narrative reading comprehension dataset with explicit/implicit question annotations. Data is loaded via HuggingFace:

```python
from dcqg.datasets.fairytaleqa_loader import load_fairytaleqa
df = load_fairytaleqa(split="train")
```

## Quick Start

### Stage 1 — Evidence Audit (No-Vote)

```bash
python -u -m scripts.run_evidence_no_vote_pilot \
  --split train --implicit_limit 500 --sample_mode random --seed 42 \
  --batch_size 5 --timeout 300 --model Qwen/Qwen3-32B \
  --output_dir outputs/runs/no_vote_pilot
```

### Stage 2 — Classifier Training

```bash
python -m scripts.train_classifier \
  --data outputs/runs/evidence_audit/train_dataset.jsonl \
  --output_dir outputs/runs/classifier
```

### Stage 3 — Question Generation

```bash
python -m scripts.run_fairytale_qg_pilot \
  --candidates_path outputs/runs/candidates.jsonl \
  --output_dir outputs/runs/qg_pilot
```

### Stage 4 — Evaluation

```bash
# CrossQG-style full evaluation
python -m scripts.run_crossqg_eval \
  --selection_mode story_matched \
  --output_dir outputs/runs/crossqg

# Reranking evaluation
python -m scripts.run_reranking_eval \
  --k5_dir outputs/runs/k5 \
  --classifier_path outputs/runs/classifier
```

## Project Structure

```
DCQG/
├── dcqg/                              # Main Python package
│   ├── difficulty/                     # Difficulty definitions + classifier
│   │   ├── definitions.py              # Canonical Easy/Medium/Hard definitions
│   │   ├── classifier.py               # MultiTaskDifficultyClassifier
│   │   ├── data.py                     # Training data construction
│   │   └── reranker.py                 # DifficultyReranker
│   ├── path/                           # Evidence audit
│   │   ├── no_vote_evidence.py         # No-Vote pipeline (Selector/Verifier)
│   │   ├── fairytale_evidence_audit.py # Utilities + old auditor
│   │   └── answer_grounded_evidence.py # Evidence planner + validator
│   ├── graph/                          # Narrative evidence graph
│   │   └── narrative_graph.py          # Graph extraction for QG scaffolding
│   ├── generation/                     # Question generation
│   │   ├── fairytale_qg.py             # 4 QG methods (Direct/ICL/SelfRefine/Ours)
│   │   └── parser.py                   # LLM response parsing
│   ├── question_filter/                # Quality filters
│   │   └── grammar.py                  # Grammar + structural checks
│   ├── evaluation/                     # Solver and judge
│   │   ├── judge.py                    # LLM judge, solver, evaluate_item
│   │   ├── metrics.py                  # Fair metrics computation
│   │   └── report.py                   # Comparison table formatting
│   ├── datasets/                       # Data loaders
│   │   └── fairytaleqa_loader.py       # FairytaleQA (HuggingFace)
│   └── utils/                          # Shared utilities
│       ├── config.py                   # .env loader
│       ├── api_client.py               # OpenAI-compatible API client
│       ├── text.py                     # Normalization, fuzzy match
│       └── jsonl.py                    # JSONL I/O
├── scripts/                            # Entry points
│   ├── run_evidence_no_vote_pilot.py   # Stage 1: No-Vote evidence audit
│   ├── train_classifier.py             # Stage 2: classifier training
│   ├── select_test_samples.py          # Stage 2: test sample selection
│   ├── run_fairytale_qg_pilot.py       # Stage 3: QG pilot
│   ├── run_crossqg_eval.py             # Stage 3/4: CrossQG evaluation
│   ├── run_k5_generation.py            # Stage 3: K=5 generation
│   ├── run_reranking_eval.py           # Stage 3/4: reranking evaluation
│   ├── run_human_eval.py               # Stage 4: human eval prep
│   ├── compute_table1.py               # Stage 4: classifier vs judges table
│   ├── compute_table3.py               # Stage 4: reranking results table
│   ├── run_llm_judge_difficulty.py     # Stage 4: LLM judge diagnostic
│   ├── audit_fairytale_candidate_suitability.py  # Candidate quality audit
│   └── audit_fairytale_target_calibration.py     # Label calibration audit
├── review-stage/                       # Project documentation
│   └── PROJECT_STATUS.md               # Full research log
├── refine-logs/                        # Design documents
│   ├── FINAL_PROPOSAL.md               # Complete experiment plan
│   └── NO_VOTE_EVIDENCE_PROMPTS.md     # Frozen prompt snapshot
├── .env.example                        # API configuration template
└── README.md
```

## QG Methods

| Method | Description | Uses Graph? |
|:-------|:------------|:-----------:|
| **Direct** | Story + answer + difficulty definition | No |
| **ICL** | Direct + few-shot examples | No |
| **SelfRefine** | Direct → critique → revise | No |
| **Ours** | Evidence graph scaffold + difficulty constraints | Yes |

All methods use the same Qwen-32B API, same difficulty definitions, and same K=5 candidate pool for fair comparison.

## Evaluation Design

| Level | Method | Purpose |
|:------|:-------|:-------|
| **Primary** | Human evaluation (100 samples, blind) | Ground-truth difficulty assessment |
| **Consistency** | Classifier on held-out fold | Reproducible self-consistency check |
| **Diagnostic** | LLM judge (GPT-4o-mini, Qwen-32B) | Method comparison and debugging |

Reranker-evaluator circularity is addressed by: (1) human eval as ground truth, (2) cross-validation, (3) classifier eval reported as consistency check only.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{dcqg2026,
  title={Evidence-Necessity-Aware Difficulty-Controlled Question Generation},
  author={},
  year={2026},
  howpublished={\url{https://github.com/1llusion-ai/ERE-DCQG}}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [FairytaleQA](https://github.com/WorkInTheDark/FairytaleQA) (Xu et al., 2022) for the narrative QA dataset
- [CrossQG](https://aclanthology.org/2025.findings-emnlp.151/) for difficulty-aware QG inspiration
- [Qwen](https://github.com/QwenLM/Qwen) for open-weight language models
- [DeBERTa-v3](https://github.com/microsoft/DeBERTa) for the classifier backbone
