# =====================================================
# 채널 성장 SHAP 분석 스크립트
# - RandomForest 회귀 모델 기반
# - SHAP bar plot / dot summary plot 저장
# =====================================================

import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_data(csv_path: str) -> pd.DataFrame:

    # 데이터 로드 함수
    df = pd.read_csv(csv_path, low_memory=False)
    print("채널 데이터 로드 완료:", df.shape)
    print("컬럼 목록:", list(df.columns))
    return df


def prepare_features(df: pd.DataFrame):

    # 독립변수/타깃 변수 구성 및 결측치 처리
    feature_cols = [
        'upload_frequency',
        'views_per_video',
        'subscriber_per_view',
        'video_count',
        'channel_age_days',
        'category_encoded',
        'country_encoded',
    ]

    target_col = 'views_last_30_days'

    # 방어 코드: 컬럼 체크
    missing_feats = [c for c in feature_cols if c not in df.columns]
    
    if missing_feats:
        raise ValueError(f"feature 컬럼 누락: {missing_feats}")

    if target_col not in df.columns:
        raise ValueError(f"타깃 컬럼 '{target_col}' 이 없습니다.")

    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    print("\n사용 독립변수:", feature_cols)
    print("타깃 변수:", target_col)

    return X, y, feature_cols, target_col


def train_model(X_train, y_train):

    # RandomForest Regressor 학습
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    print("RandomForest 회귀 모델 학습 완료")
    return rf


def compute_shap(rf_model, X_train):

    # SHAP 값 계산
    # 샘플링 (최대 3000개)
    if len(X_train) > 3000:
        X_sample = X_train.sample(3000, random_state=42)

    else:
        X_sample = X_train.copy()

    print("SHAP 계산용 샘플 크기:", X_sample.shape)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    print("SHAP 값 계산 완료")

    return X_sample, shap_values


def save_shap_plots(X_sample, shap_values, out_dir):

    # SHAP Summary / Bar Plot 저장
    os.makedirs(out_dir, exist_ok=True)

    # 1) bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    bar_path = os.path.join(out_dir, "shap_channel_growth_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("SHAP Bar Plot 저장 완료 →", bar_path)

    # 2) dot summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    dot_path = os.path.join(out_dir, "shap_channel_growth_summary.png")
    plt.tight_layout()
    plt.savefig(dot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("SHAP Summary Plot 저장 완료 →", dot_path)


def main():
    
    # CSV 경로
    csv_path = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\project\YT_ChannelGrowth_Engagement\data\processed\youtube_channels_clean_v2.csv"

    # 출력 폴더
    out_dir = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\project\YT_ChannelGrowth_Engagement\reports\figures\shap"

    # 1) 데이터 로드
    df = load_data(csv_path)

    # 2) feature / target 준비
    X, y, feature_cols, target_col = prepare_features(df)

    # 3) Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTrain:", X_train.shape, "/ Test:", X_test.shape)

    # 4) 모델 학습
    rf = train_model(X_train, y_train)

    # 5) SHAP 계산
    X_sample, shap_values = compute_shap(rf, X_train)

    # 6) SHAP 시각화 저장
    save_shap_plots(X_sample, shap_values, out_dir)

    print("\n채널 성장 SHAP 분석 종료")


if __name__ == "__main__":
    main()
