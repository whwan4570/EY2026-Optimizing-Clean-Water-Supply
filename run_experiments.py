"""
여러 설정을 한 번에 돌려서 CV 결과(TA/EC/DRP R², Average R²)를 테이블로 비교합니다.
실행: python run_experiments.py

옵션:
- 메트릭 비교만: 기본 (BENCHMARK_RETURN_METRICS=1, submission1.csv 생성 안 함)
- 각 실험용 submission_*.csv + 하이브리드 제출 생성: MAKE_SUBMISSIONS=True 로 설정
"""
import os
import sys
from typing import Optional
from pathlib import Path

# 실험 시 메트릭만 반환하도록 플래그 설정 (main 진입 전)
os.environ["BENCHMARK_RETURN_METRICS"] = "1"

import benchmark_model as bm


# 각 실험 전에 초기화할 기본값 (benchmark_model.py 상단과 동일하게)
DEFAULTS = {
    "USE_SPATIAL_FEATURES": True,
    "USE_DRP_SIMPLE_FEATURES": False,
    "USE_REGIONAL_STANDARDIZATION": True,
}

EXPERIMENTS = [
    ("default", "현재 기본 설정", {}),
    ("no_latlon", "위경도 제거 (LB 일반화)", {"USE_SPATIAL_FEATURES": False}),
    ("no_gems", "GEMS prior 제거", {}),  # USE_GEMS=0 은 아래에서 env로 설정
    ("drp_simple", "DRP 단순 피처만", {"USE_DRP_SIMPLE_FEATURES": True}),
    ("no_regional_std", "Regional standardization 끔", {"USE_REGIONAL_STANDARDIZATION": False}),
]

# 실험 후 각 설정별 submission_*.csv 및 하이브리드 제출 생성 여부
MAKE_SUBMISSIONS = True


def run_one(name: str, description: str, overrides: dict, use_gems_env: Optional[str] = None) -> Optional[dict]:
    """한 실험 실행 후 메트릭 반환. 실패 시 None."""
    for k, v in DEFAULTS.items():
        if hasattr(bm, k):
            setattr(bm, k, v)
    for k, v in overrides.items():
        if hasattr(bm, k):
            setattr(bm, k, v)
    if use_gems_env is not None:
        os.environ["USE_GEMS"] = use_gems_env
    try:
        out = bm.main()
        if out is None:
            return None
        return {"name": name, "description": description, **out}
    except Exception as e:
        print(f"[{name}] 실패: {e}", file=sys.stderr)
        return None
    finally:
        if "USE_GEMS" in os.environ and use_gems_env is not None:
            del os.environ["USE_GEMS"]


def run_submission(name: str, overrides: dict, use_gems_env: Optional[str] = None) -> Optional[Path]:
    """
    한 실험 설정으로 benchmark_model.main()을 submission 모드로 실행하고,
    생성된 data/submission1.csv 를 프로젝트 루트의 submission_<name>.csv 로 복사.
    """
    # 설정 초기화 + override 적용
    for k, v in DEFAULTS.items():
        if hasattr(bm, k):
            setattr(bm, k, v)
    for k, v in overrides.items():
        if hasattr(bm, k):
            setattr(bm, k, v)

    # env 설정
    prev_benchmark = os.environ.pop("BENCHMARK_RETURN_METRICS", None)
    if use_gems_env is not None:
        os.environ["USE_GEMS"] = use_gems_env

    try:
        bm.main()  # 제출 모드: submission1.csv 생성
    except Exception as e:
        print(f"[submission:{name}] 실패: {e}", file=sys.stderr)
        return None
    finally:
        # 환경 복구
        if prev_benchmark is not None:
            os.environ["BENCHMARK_RETURN_METRICS"] = prev_benchmark
        if "USE_GEMS" in os.environ and use_gems_env is not None:
            del os.environ["USE_GEMS"]

    src = bm.DATA_DIR / "submission1.csv"
    if not src.exists():
        print(f"[submission:{name}] data/submission1.csv 가 없습니다.", file=sys.stderr)
        return None
    dest = Path(__file__).parent / f"submission_{name}.csv"
    dest.write_bytes(src.read_bytes())
    print(f"[submission:{name}] 저장: {dest}")
    return dest


def main():
    print("=" * 70)
    print("실험 러너: 여러 설정을 순차 실행 후 Average R² 비교")
    print("=" * 70)

    results = []
    for name, description, overrides in EXPERIMENTS:
        use_gems_env = "0" if name == "no_gems" else None
        print(f"\n>>> 실험: {name} ({description})")
        row = run_one(name, description, overrides, use_gems_env=use_gems_env)
        if row is not None:
            results.append(row)

    if not results:
        print("수집된 결과 없음.")
        return

    # 테이블 출력
    import pandas as pd
    rows = []
    for r in results:
        ss = r["results_summary"]
        ta = ss[ss["Parameter"] == "Total Alkalinity"]["R2_CV_mean"].values
        ec = ss[ss["Parameter"] == "Electrical Conductance"]["R2_CV_mean"].values
        drp = ss[ss["Parameter"] == "Dissolved Reactive Phosphorus"]["R2_CV_mean"].values
        rows.append({
            "실험": r["name"],
            "설명": r["description"],
            "R²_TA": ta[0] if len(ta) else None,
            "R²_EC": ec[0] if len(ec) else None,
            "R²_DRP": drp[0] if len(drp) else None,
            "Average_R²(LB proxy)": r["lb_cv_mean"],
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("결과 비교 (Average R² = Final LB score proxy)")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    best = df.loc[df["Average_R²(LB proxy)"].idxmax()]
    worst = df.loc[df["Average_R²(LB proxy)"].idxmin()]
    print(f"  최고 Average R²: {best['실험']} ({best['Average_R²(LB proxy)']:.4f})")
    print(f"  최저 Average R²: {worst['실험']} ({worst['Average_R²(LB proxy)']:.4f})")
    if worst["Average_R²(LB proxy)"] < 0:
        print(f"  ※ {worst['실험']} 실험이 음수 R² — 해당 설정은 제출/하이브리드에 쓰지 않는 것을 권장합니다.")
    print("=" * 70)

    # 옵션: 각 실험용 submission_*.csv 및 하이브리드 제출 생성
    if not MAKE_SUBMISSIONS:
        return

    print("\n각 실험 설정으로 submission_*.csv 생성 중...")
    submission_paths = {}
    for name, description, overrides in EXPERIMENTS:
        use_gems_env = "0" if name == "no_gems" else None
        p = run_submission(name, overrides, use_gems_env=use_gems_env)
        if p is not None:
            submission_paths[name] = p
    created = list(submission_paths.keys())
    missing = [n for n, _, _ in EXPERIMENTS if n not in submission_paths]
    if missing:
        print(f"\n[제출 실패한 실험: {missing}]", file=sys.stderr)
    print(f"[제출 생성됨: {created}]")

    # 하이브리드 제출: TA/EC = default, DRP = no_gems (가장 강력한 조합)
    root = Path(__file__).parent
    path_default = submission_paths.get("default") or root / "submission_default.csv"
    path_no_gems = submission_paths.get("no_gems") or root / "submission_no_gems.csv"
    if path_default.exists() and path_no_gems.exists():
        import pandas as pd
        # no_gems 실험 성능이 나쁘면 경고 (results는 이미 수집됨)
        no_gems_row = next((r for r in results if r["name"] == "no_gems"), None)
        if no_gems_row is not None and no_gems_row["lb_cv_mean"] < 0:
            print(f"\n[경고] no_gems 실험 Average R²={no_gems_row['lb_cv_mean']:.4f} (음수). 하이브리드 DRP가 LB에서 불리할 수 있습니다.", file=sys.stderr)
        # 하이브리드 Final LB proxy: (default TA R² + default EC R² + no_gems DRP R²) / 3
        def _r2(df, exp, col):
            v = df.loc[df["실험"] == exp, col].values
            return float(v[0]) if len(v) else None
        r_ta = _r2(df, "default", "R²_TA")
        r_ec = _r2(df, "default", "R²_EC")
        r_drp_ng = _r2(df, "no_gems", "R²_DRP")
        if r_ta is not None and r_ec is not None and r_drp_ng is not None:
            hybrid_lb_proxy = (r_ta + r_ec + r_drp_ng) / 3.0
            print(f"\n  [하이브리드] Final LB score proxy (TA/EC=default, DRP=no_gems): {hybrid_lb_proxy:.4f}")
        print("\n하이브리드 제출 (TA/EC=default, DRP=no_gems) 생성...")
        sub_def = pd.read_csv(path_default)
        sub_ng = pd.read_csv(path_no_gems)
        if len(sub_def) != len(sub_ng):
            print(f"[submission:hybrid] 경고: default({len(sub_def)}행) vs no_gems({len(sub_ng)}행) 행 수 불일치", file=sys.stderr)
        sub_hybrid = sub_def.copy()
        sub_hybrid["Dissolved Reactive Phosphorus"] = sub_ng["Dissolved Reactive Phosphorus"].values
        hybrid_path = root / "submission_hybrid_defaultTAEC_noGemsDRP.csv"
        sub_hybrid.to_csv(hybrid_path, index=False)
        print(f"[submission:hybrid] 저장: {hybrid_path}")
    else:
        missing = []
        if not path_default.exists():
            missing.append("submission_default.csv")
        if not path_no_gems.exists():
            missing.append("submission_no_gems.csv")
        print(f"[submission:hybrid] 생성 불가: 다음 파일 없음 — {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
