"""
SmartPlate -- Machine Learning Model
Trains Random Forest and XGBoost models to predict next-day waste per item,
selects the best model, generates 7-day forecasts, and plots feature importance.
"""

import warnings
from pathlib import Path

import joblib
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import get_model_path, get_data_dir, COLORS

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def prepare_features(orders_df: pd.DataFrame, waste_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge orders and waste, engineer features for ML.

    Features created:
        - day_of_week_num (0-6)
        - month (1-12)
        - is_weekend, is_holiday
        - weather_encoded (0-3)
        - last_7day_avg_waste (rolling mean of waste_kg per item)
        - staff_count
        - item_encoded (label-encoded item name)

    Target: waste_kg
    """
    merged = pd.merge(
        orders_df[["date", "item_name", "category", "prep_qty", "orders_qty",
                    "is_weekend", "is_holiday", "weather", "staff_count", "event_flag"]],
        waste_df[["date", "item_name", "waste_kg"]],
        on=["date", "item_name"]
    )

    merged = merged.sort_values(["item_name", "date"]).reset_index(drop=True)

    # Numeric features
    merged["day_of_week_num"] = merged["date"].dt.dayofweek
    merged["month"] = merged["date"].dt.month

    # Weather encoding
    weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3}
    merged["weather_encoded"] = merged["weather"].map(weather_map)

    # Item encoding
    le = LabelEncoder()
    merged["item_encoded"] = le.fit_transform(merged["item_name"])

    # Rolling 7-day average waste per item
    merged["last_7day_avg_waste"] = (
        merged.groupby("item_name")["waste_kg"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )
    merged["last_7day_avg_waste"] = merged["last_7day_avg_waste"].fillna(
        merged.groupby("item_name")["waste_kg"].transform("mean")
    )

    return merged, le


# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "day_of_week_num", "month", "is_weekend", "is_holiday",
    "weather_encoded", "last_7day_avg_waste", "staff_count",
    "item_encoded", "event_flag"
]

TARGET_COL = "waste_kg"


def train_models(orders_df: pd.DataFrame, waste_df: pd.DataFrame) -> dict:
    """
    Train Random Forest and XGBoost models, compare performance,
    save the best model, and return results.

    Returns:
        dict with keys: best_model_name, rf_metrics, xgb_metrics,
                        best_model, label_encoder, feature_importance_df
    """
    merged, le = prepare_features(orders_df, waste_df)

    # Drop rows with NaN
    data = merged.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Time-based split
    )

    # ── Random Forest ──
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # ── XGBoost ──
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_available = True
    except ImportError:
        xgb_model = None
        xgb_mae = float("inf")
        xgb_rmse = float("inf")
        xgb_available = False

    # ── Select best model ──
    if xgb_available and xgb_mae < rf_mae:
        best_model = xgb_model
        best_name = "XGBoost"
    else:
        best_model = rf
        best_name = "Random Forest"

    # ── Save best model ──
    model_path = get_model_path()
    joblib.dump({
        "model": best_model,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
        "model_name": best_name,
    }, model_path)

    # ── Feature Importance ──
    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    results = {
        "best_model_name": best_name,
        "rf_metrics": {"MAE": round(rf_mae, 4), "RMSE": round(rf_rmse, 4)},
        "xgb_metrics": {
            "MAE": round(xgb_mae, 4) if xgb_available else "N/A",
            "RMSE": round(xgb_rmse, 4) if xgb_available else "N/A",
        },
        "best_model": best_model,
        "label_encoder": le,
        "feature_importance_df": fi_df,
    }
    return results


# ──────────────────────────────────────────────
# 7-Day Forecast
# ──────────────────────────────────────────────

def generate_forecast(
    orders_df: pd.DataFrame,
    waste_df: pd.DataFrame,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Generate waste_kg forecast for the next `days_ahead` days for each item.

    Returns a DataFrame with columns: date, item_name, forecast_waste_kg
    """
    # Load or train model
    model_path = get_model_path()
    if model_path.exists():
        saved = joblib.load(model_path)
        model = saved["model"]
        le = saved["label_encoder"]
    else:
        results = train_models(orders_df, waste_df)
        model = results["best_model"]
        le = results["label_encoder"]

    merged, _ = prepare_features(orders_df, waste_df)
    items = merged["item_name"].unique()
    last_date = merged["date"].max()

    forecast_records = []

    for item in items:
        item_data = merged[merged["item_name"] == item].sort_values("date")
        recent_waste = item_data["waste_kg"].tail(7).values

        for d in range(1, days_ahead + 1):
            future_date = last_date + pd.Timedelta(days=d)
            dow = future_date.dayofweek
            month = future_date.month
            is_weekend = 1 if dow >= 5 else 0
            is_holiday = 0  # Simplified — no holiday lookahead
            weather_enc = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            staff = int(item_data["staff_count"].mean())
            item_enc = le.transform([item])[0] if item in le.classes_ else 0
            avg_7 = float(np.mean(recent_waste[-7:])) if len(recent_waste) > 0 else 0
            event = 0

            features = pd.DataFrame([{
                "day_of_week_num": dow,
                "month": month,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "weather_encoded": weather_enc,
                "last_7day_avg_waste": avg_7,
                "staff_count": staff,
                "item_encoded": item_enc,
                "event_flag": event,
            }])

            pred = model.predict(features)[0]
            pred = max(pred, 0)  # No negative waste

            forecast_records.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "item_name": item,
                "forecast_waste_kg": round(pred, 2),
            })

            # Update rolling window
            recent_waste = np.append(recent_waste, pred)[-7:]

    forecast_df = pd.DataFrame(forecast_records)
    return forecast_df


# ──────────────────────────────────────────────
# Feature Importance Plot
# ──────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    """Create a horizontal bar chart of feature importances using Plotly."""
    df_sorted = fi_df.sort_values("importance", ascending=True)
    colors = COLORS["gradient"][:len(df_sorted)]

    fig = go.Figure(go.Bar(
        x=df_sorted["importance"],
        y=df_sorted["feature"],
        orientation="h",
        marker=dict(
            color=colors[:len(df_sorted)],
            line=dict(width=0),
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Feature Importance - Best Model",
            font=dict(size=15, color="#E8E0D8", family="DM Sans, sans-serif"),
        ),
        xaxis_title="Importance",
        yaxis_title="",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A89F96", family="DM Sans, sans-serif", size=12),
        xaxis=dict(gridcolor="rgba(196,154,108,0.06)", zeroline=False,
                   tickfont=dict(color="#8C8279")),
        yaxis=dict(gridcolor="rgba(196,154,108,0.06)", zeroline=False,
                   tickfont=dict(color="#8C8279")),
        margin=dict(l=120, r=30, t=50, b=40),
    )

    return fig


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from src.data_generator import load_data

    print("[SmartPlate] Training ML models...")
    orders_df, waste_df = load_data()
    results = train_models(orders_df, waste_df)

    print(f"\n[OK] Best model: {results['best_model_name']}")
    print(f"[RF] Random Forest  - MAE: {results['rf_metrics']['MAE']}, RMSE: {results['rf_metrics']['RMSE']}")
    print(f"[XGB] XGBoost       - MAE: {results['xgb_metrics']['MAE']}, RMSE: {results['xgb_metrics']['RMSE']}")
    print(f"\n[FEATURES] Feature Importance:")
    print(results['feature_importance_df'].to_string(index=False))

    print("\n[FORECAST] Generating 7-day forecast...")
    forecast = generate_forecast(orders_df, waste_df)
    print(f"[OK] Forecast generated for {forecast['item_name'].nunique()} items")
    print(forecast.head(15).to_string(index=False))
