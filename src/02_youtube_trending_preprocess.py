# youtube_trending_preprocess.py

import pandas as pd
import os
import re
import json

# -----------------------------------------------------
# 1. 프로젝트 기준 경로 설정
# -----------------------------------------------------

base_dir = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\YT_ChannelGrowth_Engagement"

paths = {
    "BR": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "BR_youtube_trending_data.csv"),
    "CA": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "CA_youtube_trending_data.csv"),
    "DE": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "DE_youtube_trending_data.csv"),
    "FR": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "FR_youtube_trending_data.csv"),
    "GB": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "GB_youtube_trending_data.csv"),
    "IN": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "IN_youtube_trending_data.csv"),
    "JP": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "JP_youtube_trending_data.csv"),
    "KR": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "KR_youtube_trending_data.csv"),
    "MX": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "MX_youtube_trending_data.csv"),
    "RU": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "RU_youtube_trending_data.csv"),
    "US": os.path.join(base_dir, "data", "raw", "youtube_trending_video_dataset", "US_youtube_trending_data.csv"),
}

json_folder = os.path.join(base_dir, "data", "raw", "youtube_trending_video_categories")

# -----------------------------------------------------
# 2. CSV 파일 존재 여부 확인
# -----------------------------------------------------

missing = [country for country, path in paths.items() if not os.path.exists(path)]

if missing:
    raise FileNotFoundError(f"누락된 CSV 파일이 있습니다: {missing}")

# -----------------------------------------------------
# 3. 필수 컬럼 정의
# -----------------------------------------------------

use_cols = [
    "channelId", "video_id", "title",
    "publishedAt", "trending_date",
    "categoryId", "tags",
    "view_count", "likes", "comment_count"
]

# -----------------------------------------------------
# 4. CSV 읽기 및 전처리
# -----------------------------------------------------

yt_trending_list = []

for country, path in paths.items():
    df = pd.read_csv(path, low_memory=False, encoding="latin1")
    missing_cols = [col for col in use_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"{country} CSV에서 필수 컬럼 누락: {missing_cols}")
    
    df = df[use_cols]
    df["country"] = country
    yt_trending_list.append(df)

# -----------------------------------------------------
# 5. 데이터 병합
# -----------------------------------------------------

yt_trending_df = pd.concat(yt_trending_list, ignore_index=True)

# -----------------------------------------------------
# 6. 날짜 변환
# -----------------------------------------------------

yt_trending_df["publishedAt"] = pd.to_datetime(yt_trending_df["publishedAt"], errors="coerce")
yt_trending_df["trending_date"] = pd.to_datetime(yt_trending_df["trending_date"], errors="coerce")
yt_trending_df = yt_trending_df.dropna(subset=["publishedAt", "trending_date"])

# -----------------------------------------------------
# 7. tags 파싱
# -----------------------------------------------------

yt_trending_df["tags"] = yt_trending_df["tags"].fillna("")

def parse_tags(x):

    if x in ["[none]", "None", "none", ""]:
        return []
    
    return x.split("|")

yt_trending_df["tags_list"] = yt_trending_df["tags"].apply(parse_tags)
yt_trending_df["tags_count"] = yt_trending_df["tags_list"].apply(len)

# -----------------------------------------------------
# 8. trending_days 계산
# -----------------------------------------------------

yt_trending_df["trending_days"] = yt_trending_df.groupby("video_id")["video_id"].transform("count")

# -----------------------------------------------------
# 9. 날짜 기반 파생 컬럼
# -----------------------------------------------------

yt_trending_df["publish_month"] = yt_trending_df["publishedAt"].dt.month
yt_trending_df["publish_dayofweek"] = yt_trending_df["publishedAt"].dt.dayofweek
yt_trending_df["days_since_publish"] = (yt_trending_df["trending_date"] - yt_trending_df["publishedAt"]).dt.days

# -----------------------------------------------------
# 10. 비율 기반 파생 컬럼
# -----------------------------------------------------

yt_trending_df["like_ratio"] = yt_trending_df["likes"] / yt_trending_df["view_count"].replace(0, 1)
yt_trending_df["comment_ratio"] = yt_trending_df["comment_count"] / yt_trending_df["view_count"].replace(0, 1)
yt_trending_df["engagement_score"] = (yt_trending_df["likes"] + yt_trending_df["comment_count"]) / yt_trending_df["view_count"].replace(0, 1)

# -----------------------------------------------------
# 11. categoryId → category_name 매핑
# -----------------------------------------------------

def load_category_map(json_path):

    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {int(item["id"]): item["snippet"]["title"] for item in data.get("items", [])}
    return mapping

category_map = {}

for country in paths.keys():
    json_path = os.path.join(json_folder, f"{country}_category_id.json")
    category_map.update(load_category_map(json_path))

yt_trending_df["categoryId"] = yt_trending_df["categoryId"].astype(int)
yt_trending_df["category_name"] = yt_trending_df["categoryId"].map(category_map).fillna("Unknown")

# -----------------------------------------------------
# 12. 파생 컬럼 검증
# -----------------------------------------------------

derived_cols = [
    "trending_days", "tags_list", "tags_count", "category_name",
    "publish_month", "publish_dayofweek", "days_since_publish",
    "like_ratio", "comment_ratio", "engagement_score"
]

missing_derived = [col for col in derived_cols if col not in yt_trending_df.columns]

if missing_derived:
    raise ValueError(f"파생 컬럼이 생성되지 않음: {missing_derived}")

# -----------------------------------------------------
# 13. 저장 경로 자동 생성
# -----------------------------------------------------

def get_next_version_file(base_dir, base_name):
    existing_files = os.listdir(base_dir)
    pattern = re.compile(rf"{base_name}_v(\d+)\.csv")

    versions = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]
    next_version = max(versions) + 1 if versions else 1
    
    filename = f"{base_name}_v{next_version}.csv"
    return os.path.join(base_dir, filename)

# -----------------------------------------------------
# 14. 저장
# -----------------------------------------------------

save_dir = os.path.join(base_dir, "data", "processed")
os.makedirs(save_dir, exist_ok=True)
save_path = get_next_version_file(save_dir, "youtube_trending_video_clean")
yt_trending_df.to_csv(save_path, index=False)

print(f"저장 완료: {save_path}")
print(f"최종 데이터 형태: {yt_trending_df.shape}")
