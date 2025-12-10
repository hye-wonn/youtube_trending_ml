# youtube_channels_preprocess.py

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# -----------------------------------------------------
# 1. 경로 설정
# -----------------------------------------------------

# 노트북 기준 프로젝트 폴더
base_dir = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\YT_ChannelGrowth_Engagement"

# CSV 경로
csv_path = os.path.join(base_dir, "data", "raw", "youtube_2025_channels", "youtube_channel_info_v2.csv")

# -----------------------------------------------------
# 2. 데이터 불러오기
# -----------------------------------------------------

try:
    yt_channels_df = pd.read_csv(csv_path)

except Exception as e:
    raise ValueError(f"CSV 파일을 읽는 중 오류 발생 → {str(e)}")

# -----------------------------------------------------
# 3. 필요한 컬럼만 선택
# -----------------------------------------------------

cols_to_use = [
    "channel_id",
    "channel_name",
    "subscriber_count",
    "view_count",
    "video_count",
    "created_date",
    "category",
    "country",
    "videos_last_30_days",
    "views_last_30_days"
]

missing_cols = [col for col in cols_to_use if col not in yt_channels_df.columns]

if missing_cols:
    raise ValueError(f"필수 컬럼 누락: {missing_cols}")

yt_channels_df = yt_channels_df[cols_to_use].copy()

# -----------------------------------------------------
# 4. 숫자형 변환
# -----------------------------------------------------

num_cols = ["subscriber_count", "view_count", "video_count", "videos_last_30_days", "views_last_30_days"]

for col in num_cols:
    yt_channels_df[col] = pd.to_numeric(yt_channels_df[col], errors="coerce")

# -----------------------------------------------------
# 5. 날짜 변환
# -----------------------------------------------------

yt_channels_df["created_date"] = pd.to_datetime(yt_channels_df["created_date"], errors="coerce", utc=True)

# -----------------------------------------------------
# 6. 파생 컬럼 생성
# -----------------------------------------------------

ref_date = pd.Timestamp.utcnow().normalize()
yt_channels_df["channel_age_days"] = (ref_date - yt_channels_df["created_date"]).dt.days
yt_channels_df["upload_frequency"] = yt_channels_df["video_count"] / yt_channels_df["channel_age_days"].replace({0: np.nan})
yt_channels_df["subscriber_per_view"] = yt_channels_df["subscriber_count"] / yt_channels_df["view_count"].replace({0: np.nan})
yt_channels_df["views_per_video"] = yt_channels_df["view_count"] / yt_channels_df["video_count"].replace({0: np.nan})
yt_channels_df["uploads_per_subscriber"] = yt_channels_df["video_count"] / yt_channels_df["subscriber_count"].replace({0: np.nan})

# 범주형 변수 처리 + 인코딩
for col in ["category", "country"]:
    yt_channels_df[col] = yt_channels_df[col].astype(str).str.strip()
    yt_channels_df[col] = yt_channels_df[col].replace("", np.nan)
    yt_channels_df[f"{col}_encoded"] = yt_channels_df[col].astype("category").cat.codes

# -----------------------------------------------------
# 7. 파생 컬럼 검증
# -----------------------------------------------------

derived_cols = [
    "channel_age_days", "upload_frequency",
    "subscriber_per_view", "views_per_video", "uploads_per_subscriber",
    "category_encoded", "country_encoded"
]

missing_derived = [col for col in derived_cols if col not in yt_channels_df.columns]

if missing_derived:
    raise ValueError(f"파생 컬럼 생성 실패: {missing_derived}")

# -----------------------------------------------------
# 8. 저장 경로 자동 생성 함수
# -----------------------------------------------------

def get_next_version_file(base_dir, base_name):

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    existing_files = os.listdir(base_dir)
    pattern = re.compile(rf"{base_name}_v(\d+)\.csv")
    versions = []

    for f in existing_files:
        match = pattern.match(f)

        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1
    filename = f"{base_name}_v{next_version}.csv"
    return os.path.join(base_dir, filename)

# -----------------------------------------------------
# 9. 저장
# -----------------------------------------------------

save_dir = os.path.join(base_dir, "data", "processed")
base_name = "youtube_channels_clean"

save_path = get_next_version_file(save_dir, base_name)
yt_channels_df.to_csv(save_path, index=False)

print(f"채널 데이터 전처리 완료 및 저장: {save_path}")
print(f"최종 데이터 형태: {yt_channels_df.shape}")
