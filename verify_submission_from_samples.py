"""
submission_from_samples.csv가 submission_template과 행·열이 동일한지 검증.
실행: python verify_submission_from_samples.py
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATE = DATA_DIR / "submission_template.csv"
OUTPUT = DATA_DIR / "submission_from_samples.csv"


def main():
    if not TEMPLATE.exists():
        print(f"없음: {TEMPLATE}")
        return
    if not OUTPUT.exists():
        print(f"없음: {OUTPUT}. make_submission_from_samples.py 먼저 실행.")
        return
    t = pd.read_csv(TEMPLATE)
    o = pd.read_csv(OUTPUT)
    ok = True
    print("1) 행 수:", len(t), "vs", len(o), "->", "OK" if len(t) == len(o) else "불일치")
    if len(t) != len(o):
        ok = False
    print("2) 열 순서/이름:", list(t.columns) == list(o.columns) and "OK" or "불일치")
    if list(t.columns) != list(o.columns):
        ok = False
    lat_eq = (t["Latitude"].astype(str).values == o["Latitude"].astype(str).values).all()
    lon_eq = (t["Longitude"].astype(str).values == o["Longitude"].astype(str).values).all()
    date_eq = (t["Sample Date"].astype(str).values == o["Sample Date"].astype(str).values).all()
    print("3) 행 순서( Lat/Lon/Date 동일 ):", "OK" if (lat_eq and lon_eq and date_eq) else "불일치")
    if not (lat_eq and lon_eq and date_eq):
        ok = False
    nan_count = o[["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]].isna().sum()
    print("4) TA/EC/DRP 결측:", nan_count.to_dict(), "->", "OK" if nan_count.sum() == 0 else "결측 있음")
    if nan_count.sum() != 0:
        ok = False
    print()
    if ok:
        print("검증 통과: template과 행·열·순서 일치, 결측 없음.")
    else:
        print("검증 실패: 위 항목 확인.")
    print()
    print("참고: LB 점수가 매우 낮다면, '가장 가까운 GEMS 지점 값'이 제출 위치의 정답과 다르기 때문일 수 있음 (R² 음수 = 평균보다 못한 예측).")


if __name__ == "__main__":
    main()
