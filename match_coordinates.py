"""
samples.csv와 metadata.xlsx를 GEMS Station Number로 매칭하여
Longitude, Latitude를 추가하는 스크립트

사용 전 openpyxl 설치 필요: pip install openpyxl
"""

import pandas as pd
from pathlib import Path


def match_coordinates(
    samples_path: str = "samples.csv",
    metadata_path: str = "metadata.xlsx",
    output_path: str = "samples_with_coordinates.csv",
) -> pd.DataFrame:
    """
    samples.csv의 GEMS Station Number에 따라 metadata.xlsx에서
    Longitude, Latitude를 매칭하여 반환합니다.

    Args:
        samples_path: samples.csv 파일 경로
        metadata_path: metadata.xlsx 파일 경로
        output_path: 결과 저장 경로

    Returns:
        Longitude, Latitude가 추가된 DataFrame
    """
    # paths
    base_dir = Path(__file__).parent
    samples_file = base_dir / samples_path
    metadata_file = base_dir / metadata_path

    if not samples_file.exists():
        raise FileNotFoundError(f"samples 파일을 찾을 수 없습니다: {samples_file}")

    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata 파일을 찾을 수 없습니다: {metadata_file}")

    # samples.csv 읽기 (세미콜론 구분자)
    samples_df = pd.read_csv(samples_file, sep=";", encoding="utf-8")

    # metadata.xlsx의 Station_Metadata 시트 읽기
    try:
        metadata_df = pd.read_excel(
            metadata_file,
            sheet_name="Station_Metadata",
            engine="openpyxl",
        )
    except ImportError as e:
        raise ImportError(
            "metadata.xlsx 읽기에 openpyxl이 필요합니다. "
            "다음 명령으로 설치하세요: pip install openpyxl"
        ) from e

    # GEMS Station Number 기준으로 필요한 컬럼만 선택
    station_cols = ["GEMS Station Number", "Latitude", "Longitude"]
    # 컬럼명이 다를 수 있으므로 유연하게 처리
    available_cols = [c for c in station_cols if c in metadata_df.columns]
    if "GEMS Station Number" not in available_cols:
        raise ValueError(
            f"metadata에 'GEMS Station Number' 컬럼이 없습니다. "
            f"현재 컬럼: {metadata_df.columns.tolist()}"
        )

    station_coords = metadata_df[available_cols].drop_duplicates(
        subset="GEMS Station Number", keep="first"
    )

    # GEMS Station Number로 매칭 (merge)
    merged_df = samples_df.merge(
        station_coords,
        on="GEMS Station Number",
        how="left",
    )

    # 결과 저장
    output_file = base_dir / output_path
    merged_df.to_csv(output_file, sep=";", index=False, encoding="utf-8")
    print(f"결과 저장 완료: {output_file}")

    # 매칭 통계
    matched = merged_df["Latitude"].notna().sum() if "Latitude" in merged_df.columns else 0
    total = len(merged_df)
    print(f"총 {total}개 행 중 {matched}개 행에 좌표가 매칭되었습니다.")

    return merged_df


if __name__ == "__main__":
    result = match_coordinates(
        samples_path="samples.csv",
        metadata_path="metadata.xlsx",
        output_path="samples_with_coordinates.csv",
    )
    print("\n처음 5행 미리보기:")
    print(result.head().to_string())
