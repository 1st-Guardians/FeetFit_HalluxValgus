"""
processed_axis_dataset.csv를 train/val/test로 분할합니다.

Data leakage 방지:
  동일 X-ray에서 left/right 두 발이 각각 다른 split에 들어가지 않도록
  source_filename 단위로 분할합니다.

실행:
  python src/prepare/02_create_axis_splits.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_AXIS_CSV, SPLIT_DIR, AXIS_TRAIN_CSV, AXIS_VAL_CSV, AXIS_TEST_CSV, SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(PROCESSED_AXIS_CSV))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"processed axis csv가 없습니다: {csv_path}")

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train + val + test 비율의 합은 1이어야 합니다.")

    df = pd.read_csv(csv_path)
    df = df[df["valid_axis"].astype(bool)].reset_index(drop=True)

    # source_filename 단위로 분할 (leakage 방지)
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

    train_df.to_csv(AXIS_TRAIN_CSV, index=False, encoding="utf-8-sig")
    val_df.to_csv(AXIS_VAL_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(AXIS_TEST_CSV, index=False, encoding="utf-8-sig")

    print("Axis split 생성 완료 (source_filename 기준)")
    print("=" * 60)
    print(f"전체 valid 샘플: {len(df)}")
    print(f"고유 source_filename: {n_files}")
    print(f"Train files: {len(train_files)}, 샘플: {len(train_df)} -> {AXIS_TRAIN_CSV}")
    print(f"Val   files: {len(val_files)}, 샘플: {len(val_df)} -> {AXIS_VAL_CSV}")
    print(f"Test  files: {len(test_files)}, 샘플: {len(test_df)} -> {AXIS_TEST_CSV}")

    print("\nHVA 분포")
    print(f"Train mean/std: {train_df['HVA'].mean():.2f} / {train_df['HVA'].std():.2f}")
    print(f"Val   mean/std: {val_df['HVA'].mean():.2f} / {val_df['HVA'].std():.2f}")
    print(f"Test  mean/std: {test_df['HVA'].mean():.2f} / {test_df['HVA'].std():.2f}")


if __name__ == "__main__":
    main()