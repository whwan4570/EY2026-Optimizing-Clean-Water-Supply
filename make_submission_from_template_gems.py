"""
submission_template에 맞춰 GEMS 데이터를 붙여 제출 파일 생성.
- template 행 순서 유지, GEMS 값으로 Total Alkalinity / Electrical Conductance / Dissolved Reactive Phosphorus 채움.
실행: python make_submission_from_template_gems.py
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KEY_COLS = ["Latitude", "Longitude", "Sample Date"]
TARGET_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
GEMS_TO_TARGET = {"gems_Alk_Tot": "Total Alkalinity", "gems_EC": "Electrical Conductance", "gems_DRP": "Dissolved Reactive Phosphorus"}


def main():
    template_path = DATA_DIR / "submission_template.csv"
    gems_path = DATA_DIR / "gems_features_validation.csv"
    if not template_path.exists():
        print(f"없음: {template_path}")
        return
    if not gems_path.exists():
        print(f"없음: {gems_path}. create_enriched_dataset.py 로 생성 후 실행.")
        return

    sub = pd.read_csv(template_path)
    gems = pd.read_csv(gems_path)

    if len(sub) != len(gems):
        print(f"행 수 불일치: template {len(sub)} vs GEMS {len(gems)}. 키로 merge합니다.")
        merged = sub[KEY_COLS].merge(
            gems[KEY_COLS + list(GEMS_TO_TARGET.keys())],
            on=KEY_COLS,
            how="left",
            suffixes=("", "_g"),
        )
        for gcol, tcol in GEMS_TO_TARGET.items():
            if gcol in merged.columns:
                sub[tcol] = merged[gcol].values
    else:
        # 행 순서 1:1 가정 (compare_ec_gems_submission에서 200/200 일치 확인됨)
        key_match = (
            (sub["Latitude"].astype(str) == gems["Latitude"].astype(str))
            & (sub["Longitude"].astype(str) == gems["Longitude"].astype(str))
            & (sub["Sample Date"].astype(str) == gems["Sample Date"].astype(str))
        )
        if not key_match.all():
            print("경고: template과 GEMS 행 순서가 다릅니다. 키로 merge합니다.")
            merged = sub[KEY_COLS].merge(
                gems[KEY_COLS + list(GEMS_TO_TARGET.keys())],
                on=KEY_COLS,
                how="left",
            )
            for gcol, tcol in GEMS_TO_TARGET.items():
                if gcol in merged.columns:
                    sub[tcol] = merged[gcol].values
        else:
            for gcol, tcol in GEMS_TO_TARGET.items():
                sub[tcol] = gems[gcol].values

    out_path = DATA_DIR / "submission_gems_baseline.csv"
    sub.to_csv(out_path, index=False)
    print(f"저장: {out_path} ({len(sub)}행)")
    print(f"  Total Alkalinity: min={sub['Total Alkalinity'].min():.4f}, max={sub['Total Alkalinity'].max():.4f}, mean={sub['Total Alkalinity'].mean():.4f}")
    print(f"  Electrical Conductance: min={sub['Electrical Conductance'].min():.4f}, max={sub['Electrical Conductance'].max():.4f}, mean={sub['Electrical Conductance'].mean():.4f}")
    print(f"  Dissolved Reactive Phosphorus: min={sub['Dissolved Reactive Phosphorus'].min():.6f}, max={sub['Dissolved Reactive Phosphorus'].max():.4f}, mean={sub['Dissolved Reactive Phosphorus'].mean():.4f}")


if __name__ == "__main__":
    main()
