"""
Stage 1 (cleaned): tune on train, formal sample from valid.
Generates:
  - outputs/inspect_samples.md
  - outputs/sampled_paths_preview.jsonl
  - outputs/path_sampling_report.json
"""
import sys
import subprocess
from pathlib import Path


def run_command(cmd, desc):
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        sys.exit(1)
    print(f"Done: {desc}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="event_qg/data/raw")
    parser.add_argument("--tune_split", default="train")
    parser.add_argument("--tune_docs", type=int, default=20)
    parser.add_argument("--eval_split", default="valid")  # test has no relation annotations
    parser.add_argument("--num_docs", type=int, default=50)
    parser.add_argument("--samples_per_level", type=int, default=30)
    parser.add_argument("--output_dir", default="event_qg/outputs")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Inspect data (show train structure for reference)
    run_command(
        f"python {base_dir / 'inspect_data.py'} "
        f"--data_dir {args.data_dir} --split {args.tune_split} "
        f"--num_docs 10 --output {args.output_dir}/inspect_samples.md",
        "Step 1: Inspect data (train sample)"
    )

    # Step 2: Graph building on eval split
    run_command(
        f"python {base_dir / 'graph_builder.py'} "
        f"--data_dir {args.data_dir} --split {args.eval_split} "
        f"--num_docs {args.num_docs} --output_dir {args.output_dir}",
        "Step 2: Build event graphs (eval split)"
    )

    # Step 3: Path sampling (tune on train, formal on eval split)
    run_command(
        f"python {base_dir / 'path_sampler.py'} "
        f"--data_dir {args.data_dir} "
        f"--tune_split {args.tune_split} --tune_docs {args.tune_docs} "
        f"--split {args.eval_split} --num_docs {args.num_docs} "
        f"--samples_per_level {args.samples_per_level} "
        f"--output_dir {args.output_dir}",
        "Step 3: Sample paths (tune=train, eval=valid)"
    )

    print("\n=== Stage 1 Complete ===")
    print(f"Outputs in: {output_dir}")
    print("  - inspect_samples.md")
    print("  - graph_building_report.json")
    print("  - sampled_paths_preview.jsonl")
    print("  - path_sampling_report.json")


if __name__ == "__main__":
    main()