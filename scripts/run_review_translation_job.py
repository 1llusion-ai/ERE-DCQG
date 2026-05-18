"""Run sharded Chinese translation for review CSVs and export Label Studio tasks."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a review CSV with parallel shards and merge outputs.",
    )
    parser.add_argument("--input", required=True, help="Input review CSV.")
    parser.add_argument("--output", required=True, help="Merged translated CSV.")
    parser.add_argument("--tasks_output", required=True, help="Label Studio JSON output.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_batch_chars", type=int, default=18000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no_difficulty_hint", action="store_true")
    parser.add_argument("--skip_full_context", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    shard_dir: Path,
    stem: str,
    shards: int,
) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = math.ceil(len(rows) / shards)
    shard_inputs = []
    for shard_index in range(shards):
        start = shard_index * chunk_size
        end = min(start + chunk_size, len(rows))
        shard_rows = rows[start:end]
        shard_path = shard_dir / f"{stem}.shard{shard_index:02d}.input.csv"
        write_csv(shard_path, shard_rows, fieldnames)
        shard_inputs.append(shard_path)
    return shard_inputs


def launch_shards(args: argparse.Namespace, shard_inputs: list[Path], shard_dir: Path) -> list[tuple[subprocess.Popen, Path]]:
    processes: list[tuple[subprocess.Popen, Path]] = []
    for shard_input in shard_inputs:
        shard_output = shard_dir / shard_input.name.replace(".input.csv", ".output.csv")
        log_out = shard_dir / shard_input.name.replace(".input.csv", ".out.log")
        log_err = shard_dir / shard_input.name.replace(".input.csv", ".err.log")
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "scripts.translate_review_csv",
            "--input",
            str(shard_input),
            "--output",
            str(shard_output),
            "--resume",
            "--model",
            args.model,
            "--batch_size",
            str(args.batch_size),
            "--max_batch_chars",
            str(args.max_batch_chars),
            "--timeout",
            str(args.timeout),
            "--retries",
            str(args.retries),
        ]
        if args.max_tokens:
            cmd.extend(["--max_tokens", str(args.max_tokens)])
        if args.no_difficulty_hint:
            cmd.append("--no_difficulty_hint")
        if args.skip_full_context:
            cmd.append("--skip_full_context")
        out_f = log_out.open("a", encoding="utf-8")
        err_f = log_err.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            cwd=Path(__file__).resolve().parent.parent,
        )
        processes.append((process, shard_output))
        print(f"Launched shard {shard_input.name}: pid={process.pid}", flush=True)
    return processes


def wait_for_shards(processes: list[tuple[subprocess.Popen, Path]]) -> None:
    pending = {process.pid: process for process, _ in processes}
    while pending:
        finished = []
        for pid, process in pending.items():
            code = process.poll()
            if code is not None:
                print(f"Shard pid={pid} exited with code {code}", flush=True)
                if code != 0:
                    raise SystemExit(f"Shard pid={pid} failed with code {code}")
                finished.append(pid)
        for pid in finished:
            pending.pop(pid, None)
        if pending:
            time.sleep(10)


def merge_shards(shard_outputs: list[Path], output_path: Path) -> None:
    merged_rows: list[dict[str, str]] = []
    merged_fields: list[str] = []
    for path in shard_outputs:
        rows, fieldnames = read_csv(path)
        if not merged_fields:
            merged_fields = fieldnames
        else:
            for field in fieldnames:
                if field not in merged_fields:
                    merged_fields.append(field)
        merged_rows.extend(rows)
    write_csv(output_path, merged_rows, merged_fields)
    print(f"Merged {len(merged_rows)} rows into {output_path}", flush=True)


def export_tasks(output_path: Path, tasks_output: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.export_label_studio_tasks",
        "--input",
        str(output_path),
        "--output",
        str(tasks_output),
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, check=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    tasks_output = Path(args.tasks_output)
    source_path = output_path if output_path.exists() else input_path
    rows, fieldnames = read_csv(source_path)
    if not rows:
        raise SystemExit(f"No rows found in {source_path}")

    shard_dir = output_path.parent / "translation_shards" / output_path.stem
    shard_inputs = split_rows(
        rows,
        fieldnames,
        shard_dir=shard_dir,
        stem=output_path.stem,
        shards=args.shards,
    )
    processes = launch_shards(args, shard_inputs, shard_dir)
    wait_for_shards(processes)
    merge_shards([output for _, output in processes], output_path)
    export_tasks(output_path, tasks_output)
    print(f"Saved Label Studio tasks to {tasks_output}", flush=True)


if __name__ == "__main__":
    main()
