# DCQG: Difficulty-Controlled Question Generation via Evidence Necessity

<p align="center">
  <b>Evidence-necessity-aware difficulty-controlled question generation on narrative QA</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/dataset-FairytaleQA-orange.svg" alt="Dataset">
</p>

---

## Overview

**DCQG** generates reading comprehension questions whose answering difficulty (Easy / Medium / Hard) is reliably controllable. Difficulty is defined by a two-dimensional combination of **answer explicitness** and **necessary evidence scope** — not by surface features or graph hop count.

| Difficulty | Definition |
|:-----------|:-----------|
| **Easy** | Answer directly found in text; 1 necessary evidence sentence |
| **Medium** | Answer inferred from 1 sentence, OR direct answer requiring multiple sentences |
| **Hard** | Answer not directly found + multiple evidence sentences + complex reasoning |

**Core idea**: A multi-task classifier trained on counterfactual-verified evidence-necessity labels provides reproducible difficulty assessment and enables generation-time reranking that improves difficulty control for any QG method.

## Method

```
FairytaleQA train QA pairs (2166 implicit + 600 explicit)
  |
  v
No-vote evidence labeling (Qwen3-32B: selector + blind verifier + removal verifier)
  |
  v
Human annotation (200 samples, 2 annotators, Cohen's kappa)
  |
  v
Multi-task DeBERTa classifier (difficulty head + evidence head, 5-fold CV)
  |
  v
4 QG methods (Direct, ICL, SelfRefine, Ours) x K=5 candidates
  |
  v
Classifier reranking by P(target_difficulty)
  |
  v
Human evaluation (100 samples, reranked vs K=1, blind)
```

## Highlights

- **Evidence-necessity labels** — difficulty grounded in whether the answer can be found from a single sentence or requires multi-sentence evidence synthesis
- **Counterfactual verification** — each evidence sentence is verified by removal: "if this sentence is removed, can the answer still be determined?"
- **Multi-task classifier** — jointly predicts difficulty and identifies required evidence sentences, serving as both evaluation instrument and reranking signal
- **No-vote pipeline** — simplified labeling without majority voting, using blind verification and leave-one-out necessity checks
- **4 QG methods compared** — Direct, ICL, SelfRefine, and narrative-evidence-graph-based generation

## Installation

```bash
git clone https://github.com/1llusion-ai/ERE-DCQG.git
cd ERE-DCQG
pip install -r requirements.txt
```

### API Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys. The pipeline uses OpenAI-compatible endpoints.

### Data

FairytaleQA is loaded from HuggingFace at runtime — no manual download needed.

## Quick Start

### No-vote evidence labeling

```bash
python -u -m scripts.run_evidence_no_vote_pilot \
  --split train --implicit_limit 500 --sample_mode random --seed 42 \
  --batch_size 5 --timeout 300 --model Qwen/Qwen3-32B
```

### CrossQG evaluation (story-matched)

```bash
python -m scripts.run_crossqg_eval \
  --selection_mode story_matched --candidates_per_level_per_story 1 --max_stories 106
```

### Classifier training

```bash
python -m scripts.train_classifier --data_path outputs/runs/labels/train_dataset.jsonl
```

## Project Structure

```
DCQG/
├── dcqg/                              # Main Python package
│   ├── datasets/                      # Data loaders
│   │   └── fairytaleqa_loader.py      # FairytaleQA from HuggingFace
│   ├── path/                          # Evidence labeling pipeline
│   │   ├── no_vote_evidence.py        # No-vote selector + verifiers
│   │   ├── answer_grounded_evidence.py # CoT evidence analysis
│   │   ├── counterfactual_verify.py   # Remove-then-check necessity
│   │   ├── self_consistency.py        # Multi-run aggregation
│   │   └── fairytale_evidence_audit.py # Evidence audit for FairytaleQA
│   ├── graph/                         # Narrative evidence graph
│   │   └── narrative_graph.py         # Graph extraction (12 node types, 10 edge relations)
│   ├── generation/                    # Question generation
│   │   ├── fairytale_qg.py            # 4 QG methods (Direct, ICL, SelfRefine, Ours)
│   │   └── parser.py                  # JSON response parsing
│   ├── difficulty/                    # Difficulty modeling
│   │   ├── definitions.py             # Easy/Medium/Hard definitions
│   │   ├── classifier.py              # Multi-task DeBERTa classifier
│   │   ├── data.py                    # [Sn] tokenization + dataset
│   │   └── reranker.py                # Classifier-based reranking
│   ├── evaluation/                    # Evaluation
│   │   └── judge.py                   # LLM judge, quality judge
│   ├── question_filter/               # Quality filters
│   │   └── grammar.py                 # Grammar and structural checks
│   └── utils/                         # Shared utilities
│       ├── config.py                  # .env loader
│       ├── api_client.py              # OpenAI-compatible API client
│       ├── jsonl.py                   # JSONL I/O
│       └── text.py                    # Text normalization
├── scripts/                           # Entry points
│   ├── run_evidence_no_vote_pilot.py  # No-vote labeling (main)
│   ├── run_crossqg_eval.py            # CrossQG evaluation
│   ├── run_k5_generation.py           # K=5 candidate generation
│   ├── run_reranking_eval.py          # Reranking evaluation
│   ├── train_classifier.py            # Classifier training
│   ├── run_human_eval.py              # Human evaluation
│   └── run_llm_judge_difficulty.py    # LLM judge difficulty eval
├── review-stage/                      # Project status
│   └── PROJECT_STATUS.md              # Source of truth for pipeline state
└── refine-logs/                       # Research documents
    ├── FINAL_PROPOSAL.md              # Method thesis
    ├── EXPERIMENT_PLAN.md             # Experiment block definitions
    └── NO_VOTE_EVIDENCE_PROMPTS.md    # Frozen labeling prompts
```

## Citation

```bibtex
@misc{dcqg2026,
  title={Evidence-Necessity-Aware Multi-Task Classification for Difficulty-Controlled Question Generation},
  author={},
  year={2026},
  howpublished={\url{https://github.com/1llusion-ai/ERE-DCQG}}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- [FairytaleQA](https://huggingface.co/datasets/WorkInTheDark/FairytaleQA) for the narrative QA dataset
- [CrossQG](https://aclanthology.org/2025.findings-emnlp.151/) for difficulty-aware QG inspiration
- [DeBERTa-v3](https://github.com/microsoft/DeBERTa) for the classifier backbone
