#!/usr/bin/env python3
"""Run the real GPU executable; merge only actual measurements into a CSV."""

import argparse
import csv
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build/cuda_sorts"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096, 65536, 1048576])
    parser.add_argument("--patterns", nargs="+", default=["random", "sorted", "reverse", "few-unique", "extremes"])
    parser.add_argument("--algorithms", default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()
    if not args.exe.is_file():
        parser.error(f"Executable not found: {args.exe}; build the project first")
    if any(n < 0 or n > (1 << 26) for n in args.sizes) or args.repeats < 1:
        parser.error("Sizes must be in [0,2^26], and repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a previous results file if any invocation fails.
    rows = []
    with tempfile.TemporaryDirectory(prefix="cuda-sort-benchmark-") as directory:
        for index, (n, pattern) in enumerate((n, p) for n in args.sizes for p in args.patterns):
            csv_path = Path(directory) / f"run_{index}.csv"
            command = [str(args.exe.resolve()), "--algo", args.algorithms, "--n", str(n),
                       "--pattern", pattern, "--seed", str(args.seed), "--device", str(args.device),
                       "--repeats", str(args.repeats), "--csv", str(csv_path)]
            print(f"Running N={n}, pattern={pattern}", flush=True)
            subprocess.run(command, check=True)
            with csv_path.open(newline="", encoding="utf-8") as source:
                rows.extend(csv.DictReader(source))
    if not rows:
        parser.error("No benchmark rows produced")
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} real benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
