"""
다중 시드 제출 평균 (LB+0.1 권장).
benchmark_model.py를 RANDOM_SEED=42, 43, 44로 각각 실행한 뒤
submission.csv를 submission_seed42.csv, submission_seed43.csv, submission_seed44.csv로
복사해 두었다면, 이 스크립트로 세 타깃 컬럼만 평균 내어 submission.csv를 생성합니다.

사용 예 (PowerShell):
  python benchmark_model.py
  copy submission.csv submission_seed42.csv
  # benchmark_model.py에서 RANDOM_SEED=43으로 변경 후 실행
  copy submission.csv submission_seed43.csv
  # RANDOM_SEED=44로 변경 후 실행
  copy submission.csv submission_seed44.csv
  python average_submissions.py
"""
import pandas as pd
from pathlib import Path

TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DEFAULT_SEEDS = [42, 43, 44]
# benchmark_model.py 제출 형식: Latitude, Longitude, Sample Date + 타깃 3개
KEY_COLS = ["Latitude", "Longitude", "Sample Date"]


def main(seeds=None, out_path="submission.csv"):
    seeds = seeds or DEFAULT_SEEDS
    base = Path(__file__).parent
    data_dir = base / "data"
    files = [data_dir / f"submission_seed{s}.csv" for s in seeds]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("다음 파일이 없습니다. 먼저 각 시드로 benchmark_model.py를 실행하고 submission을 복사하세요:")
        for f in missing:
            print(f"  {f.name}")
        return

    dfs = [pd.read_csv(f) for f in files]
    # 행 식별: 키 컬럼 (Latitude, Longitude, Sample Date)
    key_cols = [c for c in KEY_COLS if c in dfs[0].columns]
    if not key_cols:
        raise KeyError(f"제출 파일에 키 컬럼이 없습니다. 예상: {KEY_COLS}, 실제: {list(dfs[0].columns)}")
    out = dfs[0][key_cols].copy()
    for c in TARGET_COLS:
        if c not in dfs[0].columns:
            print(f"경고: 컬럼 '{c}' 없음, 건너뜀")
            continue
        out[c] = sum(d[c] for d in dfs) / len(dfs)

    out.to_csv(data_dir / out_path, index=False)
    print(f"저장: {data_dir / out_path} (시드 {seeds} 평균)")


if __name__ == "__main__":
    main()
