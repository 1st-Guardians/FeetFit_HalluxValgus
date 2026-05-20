실행 순서

0. 프로젝트 폴더 생성

C:\Users\DS\Desktop\hallux_hva_project

1. 원본 데이터 배치

data/raw/xray_images
  - IMG000001.jpg ...
  - X-ray 원본 이미지 전체

data/raw/annotations
  - datasets.csv
  - datasets_v2.csv
  - annotations.xml

2. 패키지 설치

pip install -r requirements.txt

3. axis dataset 생성

python src/prepare/01_prepare_axis_dataset.py

4. train/val/test split 생성

python src/prepare/02_create_axis_splits.py

5. 이후 작성 예정

- src/train/train_axis_model.py
- src/evaluate/evaluate_axis_hva_ima.py
- src/prepare/prepare_xray_silhouette_dataset.py
- src/train/train_silhouette_hva_model.py
- src/prepare/prepare_real_foot_silhouette.py
- src/predict/predict_real_foot.py