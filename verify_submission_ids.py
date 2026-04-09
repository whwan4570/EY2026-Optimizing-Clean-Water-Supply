"""
제출 파일이 sample_submission(submission_template)과 ID 정렬/매핑 100% 일치하는지 검증합니다.
ID = (Latitude, Longitude, Sample Date)
"""
import argparse
from pathlib import Path

import pandas as pd


# 기본 경로 (스크립트 기준 data/ 폴더)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
ID_COLS = ["Latitude", "Longitude", "Sample Date"]
PRED_COLS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    return pd.read_csv(path)


def compare_ids(
    template: pd.DataFrame,
    submission: pd.DataFrame,
    float_rtol: float = 1e-9,
    float_atol: float = 1e-12,
) -> tuple[bool, list[str]]:
    """ID 컬럼 순서·값 일치 여부 검사. (모두 일치 여부, 메시지 리스트)"""
    msgs = []
    ok = True

    for c in ID_COLS:
        if c not in template.columns:
            msgs.append(f"[오류] 템플릿에 ID 컬럼 없음: {c}")
            ok = False
        if c not in submission.columns:
            msgs.append(f"[오류] 제출 파일에 ID 컬럼 없음: {c}")
            ok = False
    if not ok:
        return False, msgs

    n_t, n_s = len(template), len(submission)
    if n_t != n_s:
        msgs.append(f"[오류] 행 수 불일치: 템플릿={n_t}, 제출={n_s}")
        ok = False

    # 행별 ID 비교 (순서 동일 가정)
    for i in range(min(n_t, n_s)):
        for c in ID_COLS:
            vt = template[c].iloc[i]
            vs = submission[c].iloc[i]
            if c == "Sample Date":
                if str(vt).strip() != str(vs).strip():
                    msgs.append(f"행 {i+1} ({c}): 템플릿={repr(vt)}, 제출={repr(vs)}")
                    ok = False
                    break
            else:
                try:
                    t_f, s_f = float(vt), float(vs)
                    if not (abs(t_f - s_f) <= (float_atol + float_rtol * abs(t_f))):
                        msgs.append(f"행 {i+1} ({c}): 템플릿={vt}, 제출={vs}")
                        ok = False
                        break
                except (TypeError, ValueError):
                    if vt != vs:
                        msgs.append(f"행 {i+1} ({c}): 템플릿={repr(vt)}, 제출={repr(vs)}")
                        ok = False
                        break
        if not ok and len(msgs) >= 10:  # 최대 10개까지만
            msgs.append("... (이후 생략)")
            break

    if ok and n_t == n_s:
        msgs.append(f"ID 일치: {n_t}행 모두 (Latitude, Longitude, Sample Date) 순서·값 100% 일치")
    return ok, msgs


def check_pred_columns(submission: pd.DataFrame) -> list[str]:
    """예측 컬럼 존재 여부."""
    msgs = []
    for c in PRED_COLS:
        if c not in submission.columns:
            msgs.append(f"[경고] 제출 파일에 예측 컬럼 없음: {c}")
        elif submission[c].isna().all():
            msgs.append(f"[경고] 제출 파일 예측 컬럼 전부 NaN: {c}")
    return msgs


def main():
    parser = argparse.ArgumentParser(description="제출 파일 ID가 템플릿과 100% 일치하는지 검증")
    parser.add_argument(
        "--template",
        type=Path,
        default=DATA_DIR / "submission_template.csv",
        help="샘플/템플릿 CSV 경로 (기본: data/submission_template.csv)",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=None,
        help="검증할 제출 CSV 경로 (기본: data/submission.csv → 없으면 data/submission1.csv)",
    )
    parser.add_argument("--rtol", type=float, default=1e-9, help="Lat/Lon 상대 허용 오차")
    parser.add_argument("--atol", type=float, default=1e-12, help="Lat/Lon 절대 허용 오차")
    args = parser.parse_args()

    if args.submission is None:
        for name in ["submission.csv", "submission1.csv"]:
            p = DATA_DIR / name
            if p.exists():
                args.submission = p
                break
        if args.submission is None:
            args.submission = DATA_DIR / "submission.csv"

    print("=" * 60)
    print("제출 ID 정렬/매핑 검증 (template vs submission)")
    print("=" * 60)
    print(f"  템플릿:   {args.template}")
    print(f"  제출:     {args.submission}")
    print(f"  ID 컬럼:  {ID_COLS}")
    print()

    template = load_csv(args.template)
    submission = load_csv(args.submission)

    id_ok, id_msgs = compare_ids(
        template, submission, float_rtol=args.rtol, float_atol=args.atol
    )
    for m in id_msgs:
        print(m)

    pred_msgs = check_pred_columns(submission)
    for m in pred_msgs:
        print(m)

    print()
    if id_ok and not any("오류" in m for m in id_msgs):
        print("결과: ID 정렬/매핑 100% 일치")
    else:
        print("결과: 불일치 있음 (위 메시지 확인)")
    print("=" * 60)
    return 0 if id_ok else 1


if __name__ == "__main__":
    exit(main())
