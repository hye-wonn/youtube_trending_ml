import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix
)

# =============================================================================
# 1. 데이터 로드
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:

    # CSV 파일 불러오기
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[INFO] 데이터 로드 완료: {df.shape}")
    return df

# =============================================================================
# 2. 전처리
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # 타겟 컬럼 존재 여부 확인
    if "views_last_30_days" not in df.columns:
        raise ValueError("'views_last_30_days' 컬럼이 없습니다.")

    # 타겟 결측값 및 음수 제거
    df = df.copy()
    df = df.dropna(subset=["views_last_30_days"])
    df = df[df["views_last_30_days"] >= 0]

    print("[INFO] 전처리 완료")
    return df

# =============================================================================
# 3. 독립변수 설정
# =============================================================================

def get_feature_dataset(df: pd.DataFrame):

    # 사용할 피처 목록
    candidate_features = [
        "upload_frequency",
        "views_per_video",
        "subscriber_per_view",
        "video_count",
        "channel_age_days",
        "category_encoded",
        "country_encoded",
    ]

    # 실제 존재하는 컬럼만 사용
    feature_cols = [c for c in candidate_features if c in df.columns]
    print(f"[INFO] 사용 피처: {feature_cols}")

    # 독립변수와 타겟 데이터 구성
    X = df[feature_cols].fillna(0)
    y_reg = df["views_last_30_days"].fillna(0)

    return X, y_reg, feature_cols

# =============================================================================
# 4. 회귀 모델 학습
# =============================================================================

def train_regression_model(X, y):

    # Train/Test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # RandomForest 회귀 모델
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    metrics = {
        "RMSE": (mean_squared_error(y_test, y_pred) ** 0.5),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    # 피처 중요도
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values(by="importance", ascending=False)
    )

    print("[INFO] 회귀 모델 학습 완료")
    return model, metrics, importance

# =============================================================================
# 5. 분류용 타겟 생성 (성장 빠른 채널 vs 느린 채널)
# =============================================================================

def create_classification_target(df: pd.DataFrame):

    # 상위 20% 기준을 fast 성장 채널로 설정
    threshold = np.quantile(df["views_last_30_days"], 0.8)
    df["growth_fast"] = (df["views_last_30_days"] >= threshold).astype(int)

    print(f"[INFO] 성장 기준 threshold = {threshold}")
    return df

# =============================================================================
# 6. 분류 모델 학습
# =============================================================================

def train_classification_model(X, y):

    # Train/Test 분리 (class imbalance 대응 stratify 옵션 사용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # RandomForest 분류 모델
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 평가 결과
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # 피처 중요도
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values(by="importance", ascending=False)
    )

    print("[INFO] 분류 모델 학습 완료")
    return model, report, cm, importance

# =============================================================================
# 7. 전체 파이프라인 실행
# =============================================================================

def run_pipeline(csv_path: str):
    df = load_data(csv_path)
    df = preprocess(df)

    # ---- 회귀 ----
    X, y_reg, feature_cols = get_feature_dataset(df)
    reg_model, reg_metrics, reg_importance = train_regression_model(X, y_reg)

    # ---- 분류 ----
    df = create_classification_target(df)
    X_cls = df[feature_cols].fillna(0)
    y_cls = df["growth_fast"]

    cls_model, cls_report, cls_cm, cls_importance = train_classification_model(
        X_cls, y_cls
    )

    results = {
        "regression": {
            "model": reg_model,
            "metrics": reg_metrics,
            "importance": reg_importance,
        },
        "classification": {
            "model": cls_model,
            "report": cls_report,
            "confusion_matrix": cls_cm,
            "importance": cls_importance,
        },
    }

    print("[INFO] 전체 모델링 파이프라인 완료")
    return results

# =============================================================================
# 8. 단독 실행 시 (테스트용)
# =============================================================================

if __name__ == "__main__":
    # 프로젝트 구조에 맞게 상대경로로 변경
    CSV_PATH = "../data/processed/youtube_channels_clean_v2.csv"
    run_pipeline(CSV_PATH)
