"""Pre-process surrogate data: convert 124-dim absolute states to 128-dim relative.

Reads batch_*.pt files from the source directory, converts states and
next_states via raw_to_relative(), recomputes deltas, and writes the
results to an output directory. Non-state fields (actions, t_start,
episode_ids, step_ids, forces) are passed through unchanged.

Usage:
    python -m aprx_model_elastica.preprocess_relative
    python -m aprx_model_elastica.preprocess_relative \
        --input-dir data/surrogate_rl_step \
        --output-dir data/surrogate_rl_step_rel128 \
        --workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from aprx_model_elastica.state import raw_to_relative


def convert_batch(src: Path, dst: Path) -> int:
    """Convert one batch file. Returns number of transitions processed."""
    data = torch.load(src, map_location="cpu", weights_only=True)

    states_rel = raw_to_relative(data["states"])
    next_states_rel = raw_to_relative(data["next_states"])

    out = {
        "states": states_rel,
        "next_states": next_states_rel,
        "actions": data["actions"],
        "t_start": data["t_start"],
        "episode_ids": data["episode_ids"],
        "step_ids": data["step_ids"],
    }
    if "forces" in data:
        out["forces"] = data["forces"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return len(states_rel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert surrogate data from 124-dim absolute to 128-dim relative states"
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/surrogate_rl_step",
        help="Source directory with 124-dim batch_*.pt files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/surrogate_rl_step_rel128",
        help="Destination directory for 128-dim batch files",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--verify", action="store_true", default=False,
        help="Verify round-trip accuracy on first batch after conversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(input_dir.glob("batch_*.pt"))
    if not files:
        print(f"No batch_*.pt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Skip already-converted files
    todo = []
    for f in files:
        dst = output_dir / f.name
        if not dst.exists():
            todo.append(f)

    print(f"Input:   {input_dir} ({len(files)} files)")
    print(f"Output:  {output_dir}")
    print(f"To convert: {len(todo)} files ({len(files) - len(todo)} already done)")

    if not todo:
        print("Nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    total_transitions = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_batch, f, output_dir / f.name): f
            for f in todo
        }
        for future in as_completed(futures):
            src = futures[future]
            try:
                n = future.result()
                total_transitions += n
            except Exception as e:
                print(f"  FAILED {src.name}: {e}", file=sys.stderr)
                continue
            done += 1
            if done % 1000 == 0 or done == len(todo):
                elapsed = time.monotonic() - t0
                rate = done / elapsed
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(
                    f"  {done}/{len(todo)} files "
                    f"({total_transitions:,} transitions, "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

    elapsed = time.monotonic() - t0
    print(f"\nDone: {done} files, {total_transitions:,} transitions in {elapsed:.1f}s")

    # Verify round-trip on first file
    if args.verify and done > 0:
        first_src = todo[0]
        first_dst = output_dir / first_src.name
        print(f"\nVerifying round-trip on {first_src.name}...")

        from aprx_model_elastica.state import relative_to_raw

        original = torch.load(first_src, map_location="cpu", weights_only=True)
        converted = torch.load(first_dst, map_location="cpu", weights_only=True)

        reconstructed = relative_to_raw(converted["states"])
        max_err = (reconstructed - original["states"]).abs().max().item()
        print(f"  Max round-trip error: {max_err:.2e}")
        assert max_err < 1e-5, f"Round-trip error too large: {max_err}"
        print(f"  State shape: {original['states'].shape} -> {converted['states'].shape}")
        print("  Verification passed.")


if __name__ == "__main__":
    main()
