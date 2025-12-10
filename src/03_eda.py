import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# ------------------------------------------------------
# 기본 설정
# ------------------------------------------------------

sns.set_style("whitegrid")

plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------
# 1. 데이터 로드 함수
# ------------------------------------------------------

def load_processed_data(base_dir):
    channels_path = os.path.join(base_dir, "youtube_channels_clean_v2.csv")
    trending_path = os.path.join(base_dir, "youtube_trending_video_clean_v2.csv")

    yt_channels_df = pd.read_csv(channels_path)
    yt_trending_df = pd.read_csv(trending_path)

    print(f"[INFO] 채널 데이터 로딩 완료: {yt_channels_df.shape}")
    print(f"[INFO] 트렌딩 데이터 로딩 완료: {yt_trending_df.shape}")

    return yt_channels_df, yt_trending_df

# ------------------------------------------------------
# 2. 병합 함수
# ------------------------------------------------------

def merge_data(yt_channels_df, yt_trending_df):
    merged_df = pd.merge(
        yt_trending_df,
        yt_channels_df,
        left_on="channelId",
        right_on="channel_id",
        how="left"
    )

    print(f"[INFO] 병합 완료: {merged_df.shape}")
    return merged_df

# ------------------------------------------------------
# 3. 결측치 출력
# ------------------------------------------------------

def show_missing_values(df, name=""):
    print(f"\n=== {name} 결측치 ===")
    print(df.isna().sum())

# ------------------------------------------------------
# 4. 채널별 트렌딩 영상 수 계산
# ------------------------------------------------------

def add_trending_video_count(yt_channels_df, yt_trending_df):

    yt_trending_df = yt_trending_df.rename(columns={"channelId": "channel_id"})

    channel_trending_counts = (
        yt_trending_df.groupby("channel_id")["video_id"]
        .nunique()
        .reset_index(name="channel_trending_video_count")
    )

    yt_channels_df = yt_channels_df.merge(
        channel_trending_counts,
        on="channel_id",
        how="left"
    )

    yt_channels_df["channel_trending_video_count"] = (
        yt_channels_df["channel_trending_video_count"].fillna(0)
    )

    return yt_channels_df

# ------------------------------------------------------
# 5. 시각화 함수들
# ------------------------------------------------------

def plot_channel_distributions(yt_channels_df):

    # 구독자 수
    plt.figure(figsize=(8, 5))
    sns.histplot(yt_channels_df["subscriber_count"], bins=50, log_scale=True)
    plt.title("구독자 수 분포 (로그 스케일)")
    plt.xlabel("subscriber_count")
    plt.show()

    # 영상 수
    plt.figure(figsize=(8, 5))
    sns.histplot(yt_channels_df["video_count"], bins=50, log_scale=True)
    plt.title("영상 수 분포 (로그 스케일)")
    plt.xlabel("video_count")
    plt.show()

    # 트렌딩 영상 수
    if "channel_trending_video_count" in yt_channels_df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(yt_channels_df["channel_trending_video_count"], bins=30)
        plt.title("채널별 트렌딩 영상 수")
        plt.show()

# ------------------------------------------------------
# 6. 카테고리/국가 요약
# ------------------------------------------------------

def summarize_by_group(yt_channels_df):
    if "channel_trending_ratio" in yt_channels_df.columns:
        ratio_col = "channel_trending_ratio"
    else:
        ratio_col = "subscriber_count"

    cat_summary = yt_channels_df.groupby("category").agg(
        avg_subscribers=("subscriber_count", "mean"),
        avg_views=("view_count", "mean"),
        avg_ratio=(ratio_col, "mean"),
        count_channels=("channel_id", "count")
    )
    print("\n[INFO] 카테고리별 요약:")
    print(cat_summary.head())

    country_summary = yt_channels_df.groupby("country").agg(
        avg_subscribers=("subscriber_count", "mean"),
        avg_views=("view_count", "mean"),
        avg_ratio=(ratio_col, "mean"),
        count_channels=("channel_id", "count")
    )
    print("\n[INFO] 국가별 요약:")
    print(country_summary.head())

# ------------------------------------------------------
# 7. 트렌딩 시계열 분석
# ------------------------------------------------------

def plot_trending_timeseries(merged_df):

    try:
        merged_df["trending_date_only"] = pd.to_datetime(
            merged_df["trending_date"]
        ).dt.date

        recent_trending = merged_df.groupby("trending_date_only").size()

        plt.figure(figsize=(12, 5))
        recent_trending.plot(kind="line")
        plt.title("날짜별 트렌딩 영상 수")
        plt.ylabel("트렌딩 영상 수")
        plt.xlabel("날짜")
        plt.show()

    except Exception as e:
        print("[ERROR] 시계열 그래프 실패:", e)

# ------------------------------------------------------
# 8. 영상 단위 반응률 계산
# ------------------------------------------------------

def compute_video_metrics(yt_trending_df):

    yt_trending_df["view_count_safe"] = yt_trending_df["view_count"].replace(0, pd.NA)

    yt_trending_df["engagement_rate"] = (
        yt_trending_df["likes"] + yt_trending_df["comment_count"]
    ) / yt_trending_df["view_count_safe"]

    yt_trending_df["like_rate"] = (
        yt_trending_df["likes"] / yt_trending_df["view_count_safe"]
    )

    yt_trending_df["comment_rate"] = (
        yt_trending_df["comment_count"] / yt_trending_df["view_count_safe"]
    )

    return yt_trending_df

# ------------------------------------------------------
# 9. 영상 단위 요약 생성
# ------------------------------------------------------

def create_video_summary(yt_trending_df, yt_channels_df):
    group_col = "channelId" if "channelId" in yt_trending_df.columns else "channel_id"

    video_metrics_df = (
        yt_trending_df.groupby("video_id")
        .agg(
            channel_id=(group_col, "first"),
            title=("title", "first"),
            country=("country", "first"),
            trending_days=("trending_days", "max"),
            max_views=("view_count", "max"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_like_rate=("like_rate", "mean"),
            avg_comment_rate=("comment_rate", "mean")
        )
        .reset_index()
    )

    # 채널 정보 병합
    video_metrics_with_channel = video_metrics_df.merge(
        yt_channels_df[["channel_id", "subscriber_count"]],
        on="channel_id",
        how="left"
    )

    return video_metrics_with_channel

# ------------------------------------------------------
# 10. 메인 실행 함수
# ------------------------------------------------------

def run_full_eda(base_dir):

    # 1. 데이터 로드
    yt_channels_df, yt_trending_df = load_processed_data(base_dir)

    # 2. 병합
    merged_df = merge_data(yt_channels_df, yt_trending_df)

    # 3. 결측치 확인
    show_missing_values(yt_channels_df, "채널 데이터")
    show_missing_values(yt_trending_df, "트렌딩 데이터")
    show_missing_values(merged_df, "병합 데이터")

    # 4. 채널 트렌딩 영상 수 추가
    yt_channels_df = add_trending_video_count(yt_channels_df, yt_trending_df)

    # 5. 기본 분포 시각화
    plot_channel_distributions(yt_channels_df)

    # 6. 카테고리/국가 요약
    summarize_by_group(yt_channels_df)

    # 7. 시계열 분석
    plot_trending_timeseries(merged_df)

    # 8~9. 영상 단위 반응률 및 요약 생성
    yt_trending_df = compute_video_metrics(yt_trending_df)
    video_summary = create_video_summary(yt_trending_df, yt_channels_df)

    print("\n[INFO] 영상 요약 생성 완료")
    print(video_summary.head())

    return {
        "channels": yt_channels_df,
        "trending": yt_trending_df,
        "merged": merged_df,
        "video_summary": video_summary
    }

# ------------------------------------------------------
# 단독 실행
# ------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\project\YT_ChannelGrowth_Engagement\data\processed"
    run_full_eda(BASE_DIR)
