import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from xgboost import XGBRegressor, XGBClassifier

import shap
import matplotlib.pyplot as plt

# ======================================================
# 1. 데이터 불러오기
# ======================================================

csv_path = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\project\YT_ChannelGrowth_Engagement\data\processed\youtube_trending_video_clean_v2.csv"

df = pd.read_csv(csv_path, low_memory=False)
print("v2 데이터 로드 완료, shape:", df.shape)
print("컬럼 목록:", list(df.columns))

# 필요한 컬럼 존재 여부 확인
required_cols = [
    "view_count", "likes", "comment_count",
    "categoryId", "publish_dayofweek", "tags_count",
    "trending_days", "engagement_score"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"다음 컬럼이 없습니다: {missing}")

# 공통 입력 특징(Feature)
feature_cols = [
    "view_count", "likes", "comment_count",
    "categoryId", "publish_dayofweek", "tags_count"
]

X = df[feature_cols].copy().fillna(0)

# ======================================================
# [A] 트렌딩 유지기간 예측 회귀 모델 + SHAP 분석
# ======================================================

print("\n[A] 트렌딩 유지기간 회귀 + SHAP 분석 시작")

# 타깃(트렌딩 유지 일수)
y_trend = df["trending_days"].fillna(0)

# 학습/테스트 분리
X_train_tr, X_test_tr, y_train_tr, y_test_tr = train_test_split(
    X, y_trend,
    test_size=0.2,
    random_state=42
)

# XGBoost 회귀 모델
xgb_trend = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

xgb_trend.fit(X_train_tr, y_train_tr)

# 모델 성능 평가
y_pred_tr = xgb_trend.predict(X_test_tr)
rmse_tr = mean_squared_error(y_test_tr, y_pred_tr) ** 0.5
mae_tr = mean_absolute_error(y_test_tr, y_pred_tr)

print("\n[TrendingDays - XGBoost 회귀 결과]")
print("RMSE:", rmse_tr)
print("MAE :", mae_tr)

# -----------------------------
# SHAP 분석 (회귀 모델)
# -----------------------------

print("\n[SHAP] 트렌딩 유지기간 모델 SHAP 계산 중...")

# SHAP 계산용 샘플링
sample_size_tr = min(5000, len(X_test_tr))
X_tr_sample = X_test_tr.sample(sample_size_tr, random_state=42)

# TreeExplainer 생성 및 SHAP 값 계산
explainer_trend = shap.TreeExplainer(xgb_trend)
shap_values_trend = explainer_trend.shap_values(X_tr_sample)

# 변수 중요도 Bar Plot 저장
plt.figure()
shap.summary_plot(shap_values_trend, X_tr_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_trending_days_bar.png", dpi=200)
plt.close()

# 샘플별 영향도(Beeswarm) 저장
plt.figure()
shap.summary_plot(shap_values_trend, X_tr_sample, show=False)
plt.tight_layout()
plt.savefig("shap_trending_days_beeswarm.png", dpi=200)
plt.close()

print("트렌딩 유지기간 SHAP 그래프 저장 완료 →")
print("- shap_trending_days_bar.png")
print("- shap_trending_days_beeswarm.png")

# ======================================================
# [B] 고참여(high_engagement) 예측 분류 모델 + SHAP
# ======================================================

print("\n[B] 고참여(high_engagement) 분류 + SHAP 분석 시작")

# engagement_score 기반 HIGH(상위 20%) / LOW 라벨 생성
if df["engagement_score"].isna().all():
    raise ValueError("engagement_score가 모두 NaN 입니다. 먼저 engagement_score를 채워야 합니다.")

df_eng = df.dropna(subset=["engagement_score"]).copy()

threshold = df_eng["engagement_score"].quantile(0.8)
df_eng["high_engagement"] = (df_eng["engagement_score"] >= threshold).astype(int)

print("high_engagement 기준값(상위 20%):", threshold)
print("high_engagement 라벨 분포:")
print(df_eng["high_engagement"].value_counts())

# 특징 / 타깃 분리
X_cls = df_eng[feature_cols].fillna(0)
y_cls = df_eng["high_engagement"]

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(
    X_cls, y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

# XGBoost 분류 모델
xgb_cls = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

xgb_cls.fit(X_train_cl, y_train_cl)

# 성능 평가
y_pred_cl = xgb_cls.predict(X_test_cl)
print("\n[XGBoost Classifier 결과 (high_engagement)]")
print(classification_report(y_test_cl, y_pred_cl))

# -----------------------------
# SHAP 분석 (분류)
# -----------------------------

print("\n[SHAP] 고참여 분류 모델 SHAP 계산 중...")

sample_size_cl = min(5000, len(X_test_cl))
X_cl_sample = X_test_cl.sample(sample_size_cl, random_state=42)

explainer_cls = shap.TreeExplainer(xgb_cls)
shap_values_cls = explainer_cls.shap_values(X_cl_sample)

# shap_values가 클래스별로 리스트로 나오는 경우 처리
if isinstance(shap_values_cls, list):
    shap_values_cls_plot = shap_values_cls[1]   # positive class(=1) 기준

else:
    shap_values_cls_plot = shap_values_cls

# 변수 중요도 Bar Plot
plt.figure()
shap.summary_plot(shap_values_cls_plot, X_cl_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_high_engagement_bar.png", dpi=200)
plt.close()

# Beeswarm Plot
plt.figure()
shap.summary_plot(shap_values_cls_plot, X_cl_sample, show=False)
plt.tight_layout()
plt.savefig("shap_high_engagement_beeswarm.png", dpi=200)
plt.close()

print("고참여 SHAP 그래프 저장 완료 →")
print("- shap_high_engagement_bar.png")
print("- shap_high_engagement_beeswarm.png")

print("\n영상 단위 SHAP 분석 전체 완료!")
