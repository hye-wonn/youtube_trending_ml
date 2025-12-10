import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

# ============================================
# CSV 데이터 로드
# ============================================

csv_path = r"C:\Users\73bib\Desktop\유혜원\제주한라대학교\[2025] 1학년 2학기\빅데이터 기초 및 실습\project\YT_ChannelGrowth_Engagement\data\processed\youtube_trending_video_clean_v2.csv"

df = pd.read_csv(csv_path, low_memory=False)
print("v2 데이터 로드 완료, shape:", df.shape)
print("컬럼 목록:", list(df.columns))

# ============================================
# 공통: 모델 입력용 독립변수 리스트
# ============================================

base_feature_cols = [
    "view_count",          # 조회수
    "likes",               # 좋아요 수
    "comment_count",       # 댓글 수
    "categoryId",          # 카테고리 ID
    "publish_dayofweek",   # 요일 정보
    "tags_count",          # 태그 개수
]

# 실제 존재하는 컬럼만 필터링
feature_cols = [c for c in base_feature_cols if c in df.columns]
print("\n사용 독립변수(feature_cols):", feature_cols)

# ============================================
# 함수: 회귀모델 평가 함수
# ============================================

def eval_regression(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n[{name}]")
    print("RMSE:", rmse)
    print("MAE :", mae)
    return rmse, mae

# ============================================
# 1. 트렌딩 유지기간 예측 (회귀)
#    타깃 변수: trending_days
# ============================================

if "trending_days" not in df.columns:
    raise ValueError("trending_days 컬럼이 없습니다. v2 파일을 확인하세요.")

df_trend = df.dropna(subset=["trending_days"]).copy()

X1 = df_trend[feature_cols].fillna(0)     # 독립변수
Y1 = df_trend["trending_days"].fillna(0) # 종속변수

# 데이터 분할 (train/test)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, Y1, test_size=0.2, random_state=42
)

print("\n[1번] trending_days 모델링용 데이터 분할 완료")
print("Train:", X1_train.shape, "/ Test:", X1_test.shape)

# ---------------------- 1-1. 결정트리 회귀 ----------------------

tree = DecisionTreeRegressor(
    max_depth=8,            # 트리 깊이 제한
    min_samples_leaf=50,    # 최소 leaf 크기
    random_state=42,
)

tree.fit(X1_train, y1_train)
y1_pred_tree = tree.predict(X1_test)
eval_regression("Decision Tree Regressor (trending_days)", y1_test, y1_pred_tree)

# 변수 중요도 출력
tree_importance = pd.DataFrame(
    {"feature": X1.columns, "importance": tree.feature_importances_}
).sort_values(by="importance", ascending=False)

print("\n[Decision Tree - 변수 중요도]")
print(tree_importance)

# ---------------------- 1-2. 랜덤포레스트 회귀 ----------------------

rf1 = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
)

rf1.fit(X1_train, y1_train)
y1_pred_rf = rf1.predict(X1_test)
eval_regression("Random Forest Regressor (trending_days)", y1_test, y1_pred_rf)

# 변수 중요도 출력
rf1_importance = pd.DataFrame(
    {"feature": X1.columns, "importance": rf1.feature_importances_}
).sort_values(by="importance", ascending=False)

print("\n[Random Forest - 변수 중요도]")
print(rf1_importance)

# ---------------------- 1-3. XGBoost 회귀 ----------------------

xgb1 = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)

xgb1.fit(X1_train, y1_train)
y1_pred_xgb = xgb1.predict(X1_test)
eval_regression("XGBoost Regressor (trending_days)", y1_test, y1_pred_xgb)

# 변수 중요도 출력
xgb1_importance = pd.DataFrame(
    {"feature": X1.columns, "importance": xgb1.feature_importances_}
).sort_values(by="importance", ascending=False)

print("\n[XGBoost - 변수 중요도]")
print(xgb1_importance)

# ============================================
# 2. 참여도 모델링
#    (1) engagement_score 회귀
#    (2) high_engagement 분류
# ============================================

if "engagement_score" not in df.columns:
    raise ValueError("engagement_score 컬럼이 없습니다. v2 파일을 확인하세요.")

df_eng = df.dropna(subset=["engagement_score"]).copy()

# ---------------------- 2-1. 참여도 회귀 ----------------------

X2 = df_eng[feature_cols].fillna(0)
y2 = df_eng["engagement_score"].fillna(0)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

print("\n[2번-회귀] engagement_score 회귀용 데이터 분할 완료")
print("Train:", X2_train.shape, "/ Test:", X2_test.shape)

# 결정트리 회귀 모델 생성 및 평가
tree2 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=50, random_state=42)
tree2.fit(X2_train, y2_train)
y2_pred_tree = tree2.predict(X2_test)
eval_regression("Decision Tree Regressor (engagement_score)", y2_test, y2_pred_tree)

tree2_importance = pd.DataFrame(
    {"feature": X2.columns, "importance": tree2.feature_importances_}
).sort_values(by="importance", ascending=False)
print("\n[Decision Tree(engagement_score) - 변수 중요도]")
print(tree2_importance)

# 랜덤포레스트 회귀 모델 생성 및 평가
rf2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
)

rf2.fit(X2_train, y2_train)
y2_pred_rf = rf2.predict(X2_test)
eval_regression("Random Forest Regressor (engagement_score)", y2_test, y2_pred_rf)

rf2_importance = pd.DataFrame(
    {"feature": X2.columns, "importance": rf2.feature_importances_}
).sort_values(by="importance", ascending=False)
print("\n[Random Forest(engagement_score) - 변수 중요도]")
print(rf2_importance)

# XGBoost 회귀 모델 생성 및 평가
xgb2 = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)

xgb2.fit(X2_train, y2_train)
y2_pred_xgb = xgb2.predict(X2_test)
eval_regression("XGBoost Regressor (engagement_score)", y2_test, y2_pred_xgb)

xgb2_importance = pd.DataFrame(
    {"feature": X2.columns, "importance": xgb2.feature_importances_}
).sort_values(by="importance", ascending=False)
print("\n[XGBoost(engagement_score) - 변수 중요도]")
print(xgb2_importance)

# ---------------------- 2-2. 참여도 분류 ----------------------

# 상위 20% 기준 high_engagement 라벨 생성
threshold = df_eng["engagement_score"].quantile(0.8)
df_eng["high_engagement"] = (df_eng["engagement_score"] >= threshold).astype(int)

print("\nhigh_engagement threshold (상위 20%):", threshold)
print(df_eng["high_engagement"].value_counts())

X3 = df_eng[feature_cols].fillna(0)
y3 = df_eng["high_engagement"]

# 분류 모델용 데이터 분할
X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3
)

print("\n[2번-분류] high_engagement 분류용 데이터 분할 완료")
print("Train:", X3_train.shape, "/ Test:", X3_test.shape)

# 랜덤포레스트 분류 모델
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
)

rf_clf.fit(X3_train, y3_train)
y3_pred_rf = rf_clf.predict(X3_test)

print("\n[RandomForest Classifier 결과 (high_engagement)]")
print(classification_report(y3_test, y3_pred_rf))

print("\n[Confusion Matrix]")
print(confusion_matrix(y3_test, y3_pred_rf))

rf_clf_importance = pd.DataFrame(
    {"feature": X3.columns, "importance": rf_clf.feature_importances_}
).sort_values(by="importance", ascending=False)

print("\n[RandomForest Classifier - 변수 중요도]")
print(rf_clf_importance)

# XGBoost 분류 모델
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
)

xgb_clf.fit(X3_train, y3_train)
y3_pred_xgb = xgb_clf.predict(X3_test)

print("\n[XGBoost Classifier 결과 (high_engagement)]")
print(classification_report(y3_test, y3_pred_xgb))

xgb_clf_importance = pd.DataFrame(
    {"feature": X3.columns, "importance": xgb_clf.feature_importances_}
).sort_values(by="importance", ascending=False)

print("\n[XGBoost Classifier - 변수 중요도]")
print(xgb_clf_importance)

print("\n1번(트렌딩 유지기간) + 2번(참여도 회귀/분류) 핵심 모델링 완료!")
