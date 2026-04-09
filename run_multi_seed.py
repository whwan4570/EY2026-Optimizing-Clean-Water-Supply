"""
멀티 시드 자동 실행 + 평균 submission 생성.

seeds=[42,43,44]에 대해 benchmark_model.main()을 순차 실행한 뒤
submission_seed42.csv, submission_seed43.csv, submission_seed44.csv를 만들고
average_submissions 로직으로 최종 submission.csv를 생성합니다.

사용법:
  python run_multi_seed.py
  python run_multi_seed.py --seeds 3 11 29 47 89 137 251 509 877 1301   # 시드 여러 개

의존: benchmark_model.py (RANDOM_SEED 패치), average_submissions.py 또는 동일 로직.

"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
KEY_COLS = ["Latitude", "Longitude", "Sample Date"]


def run_benchmark(seed: int) -> Path:
    """benchmark_model.main()을 주어진 시드로 실행하고 submission.csv를 seed 파일로 복사."""
    import benchmark_model as bm
    bm.RANDOM_SEED = seed
    bm.main()
    src = DATA_DIR / "submission.csv"
    dst = DATA_DIR / f"submission_seed{seed}.csv"
    if src.exists():
        shutil.copy(src, dst)
        print(f"  → {dst.name}  [Seed {seed} 완료]")
        return dst
    print(f"  [Seed {seed}] submission.csv 미생성")
    raise FileNotFoundError(f"benchmark did not produce {src}")


def average_submissions(seeds: list[int], out_path: str = "submission2.csv") -> Path:
    """submission_seed{s}.csv들을 읽어 타깃 컬럼만 평균 내어 submission.csv 저장."""
    import pandas as pd
    files = [DATA_DIR / f"submission_seed{s}.csv" for s in seeds]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {[f.name for f in missing]}")
    dfs = [pd.read_csv(f) for f in files]
    key_cols = [c for c in KEY_COLS if c in dfs[0].columns]
    out = dfs[0][key_cols].copy()
    for c in TARGET_COLS:
        if c in dfs[0].columns:
            out[c] = sum(d[c] for d in dfs) / len(dfs)
    out.to_csv(DATA_DIR / out_path, index=False)
    print(f"Saved: {DATA_DIR / out_path} (mean of seeds {seeds})")
    return DATA_DIR / out_path


def main():
    parser = argparse.ArgumentParser(description="Run benchmark for multiple seeds and average submissions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds to run (default: 42 43 44)")
    parser.add_argument("--out", type=str, default="submission.csv", help="Final submission filename")
    args = parser.parse_args()
    seeds = args.seeds
    print(f"Running benchmark for seeds: {seeds}")
    for s in seeds:
        print(f"\n--- Seed {s} ---")
        run_benchmark(s)
    print("\nAveraging submissions...")
    average_submissions(seeds, out_path=args.out)
    print("진짜 끝남 Done.")


if __name__ == "__main__":
    main()
