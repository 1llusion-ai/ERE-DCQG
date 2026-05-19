# DeBERTa Multi-Task Classifier Training

This classifier is the reranking/evaluation-side model for FairytaleQA QA
difficulty.  It uses one encoder backbone with two heads:

- difficulty head: `Easy`, `Medium`, `Hard`;
- evidence head: binary prediction over numbered sentence markers `[S0]`,
  `[S1]`, ...

Input format:

```text
Question: ...
Answer: ...
Context:
[S0] ...
[S1] ...
```

The default backbone is `microsoft/deberta-v3-base`.

## Environment

Create the local Windows virtual environment:

```powershell
.\scripts\setup_training_env.ps1 -VenvPath .venv-deberta -Torch cpu
```

For a CUDA 12.1 server:

```powershell
.\scripts\setup_training_env.ps1 -VenvPath .venv-deberta -Torch cu121
```

Activate:

```powershell
.\.venv-deberta\Scripts\Activate.ps1
```

## Prepare Human Labels

After Label Studio export:

```powershell
python -m scripts.prepare_classifier_data `
  --input C:\path\to\label_studio_export.json `
  --output outputs/runs/classifier_data/train.jsonl
```

Multiple files can be merged:

```powershell
python -m scripts.prepare_classifier_data `
  --input outputs/runs/implicit/label_studio_export.json `
  --input outputs/runs/explicit/label_studio_export.json `
  --output outputs/runs/classifier_data/train_merged.jsonl
```

By default, only human annotations are used. `--allow_model_labels` is only for
debugging weak labels before final annotation is ready.

## Local Smoke Test

This checks the full training loop with a tiny HuggingFace backbone:

```powershell
python -m scripts.train_classifier `
  --smoke_test `
  --device cpu `
  --output_dir outputs/models/smoke_deberta_multitask
```

## Full Training

```powershell
python -m scripts.train_classifier `
  --data_path outputs/runs/classifier_data/train_merged.jsonl `
  --output_dir outputs/models/deberta_v3_base_multitask `
  --model_name microsoft/deberta-v3-base `
  --split_strategy story `
  --n_folds 5 `
  --epochs 8 `
  --batch_size 8 `
  --lr 2e-5 `
  --lambda_evidence 0.3 `
  --evidence_pos_weight 3.0
```

Use `--split_strategy story` for final experiments so the same story does not
leak across train and validation folds.

Outputs:

- `normalized_records.jsonl`: exact records consumed by training;
- `run_config.json`: command/config snapshot;
- `fold_*/model/`: saved model, tokenizer, and config;
- `fold_*/metrics.json`: fold metrics;
- `cv_summary.json`: cross-validation summary.
