
"""
04_create_silhouette_splits.py

processed_silhouette_dataset_clean.csv를 train/val/test로 분할합니다.

입력:
  data/processed/xray_silhouette_dataset/processed_silhouette_dataset_clean.csv

출력:
  data/splits/silhouette_train.csv
  data/splits/silhouette_val.csv
  data/splits/silhouette_test.csv

실행:
  python src/prepare/04_create_silhouette_splits.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import (
    XRAY_SILHOUETTE_DIR,
    SILHOUETTE_TRAIN_CSV,
    SILHOUETTE_VAL_CSV,
    SILHOUETTE_TEST_CSV,
    SPLIT_DIR,
    SEED,
)


def main():
    default_csv = XRAY_SILHOUETTE_DIR / "processed_silhouette_dataset_clean.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(default_csv))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"clean silhouette csv가 없습니다: {csv_path}")

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train + val + test 비율의 합은 1이어야 합니다.")

    df = pd.read_csv(csv_path)

    required = {"sample_id", "silhouette_image_path", "HVA", "IMA"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"clean silhouette csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    # 혹시 warning 컬럼이 있으면 warning 아닌 것만 사용
    if "mask_quality_warning" in df.columns:
        before = len(df)
        df = df[~df["mask_quality_warning"].astype(bool)].reset_index(drop=True)
        print(f"mask_quality_warning 제외: {before} -> {len(df)}")

    # source_filename 단위로 분할 (leakage 방지)
    # 동일 X-ray에서 left/right 두 발이 각각 다른 split에 들어가지 않도록
    files = np.array(sorted(df["source_filename"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)

    n_files = len(files)
    n_train_f = int(round(n_files * args.train))
    n_val_f = int(round(n_files * args.val))

    train_files = set(files[:n_train_f])
    val_files = set(files[n_train_f:n_train_f + n_val_f])
    test_files = set(files[n_train_f + n_val_f:])

    train_df = df[df["source_filename"].isin(train_files)].reset_index(drop=True)
    val_df = df[df["source_filename"].isin(val_files)].reset_index(drop=True)
    test_df = df[df["source_filename"].isin(test_files)].reset_index(drop=True)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SILHOUETTE_TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(SILHOUETTE_VAL_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(SILHOUETTE_TEST_CSV, index=False, encoding="utf-8-sig")

    print("Silhouette split 생성 완료 (source_filename 기준)")
    print("=" * 70)
    print(f"Input clean CSV: {csv_path}")
    print(f"전체 샘플: {len(df)}")
    print(f"고유 source_filename: {n_files}")
    print(f"Train files: {len(train_files)}, 샘플: {len(train_df)} -> {SILHOUETTE_TRAIN_CSV}")
    print(f"Val   files: {len(val_files)}, 샘플: {len(val_df)} -> {SILHOUETTE_VAL_CSV}")
    print(f"Test  files: {len(test_files)}, 샘플: {len(test_df)} -> {SILHOUETTE_TEST_CSV}")

    print("\nHVA 분포")
    print(f"Train mean/std: {train_df['HVA'].mean():.2f} / {train_df['HVA'].std():.2f}")
    print(f"Val   mean/std: {val_df['HVA'].mean():.2f} / {val_df['HVA'].std():.2f}")
    print(f"Test  mean/std: {test_df['HVA'].mean():.2f} / {test_df['HVA'].std():.2f}")

    print("\nIMA 분포")
    print(f"Train mean/std: {train_df['IMA'].mean():.2f} / {train_df['IMA'].std():.2f}")
    print(f"Val   mean/std: {val_df['IMA'].mean():.2f} / {val_df['IMA'].std():.2f}")
    print(f"Test  mean/std: {test_df['IMA'].mean():.2f} / {test_df['IMA'].std():.2f}")


if __name__ == "__main__":
    main()
