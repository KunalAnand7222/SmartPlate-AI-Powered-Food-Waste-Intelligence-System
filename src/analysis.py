"""
SmartPlate — Analysis Module
Computes waste metrics, correlations, anomalies, and aggregations
for dashboard consumption.
"""

from datetime import timedelta

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Waste Percentage Analysis
# ──────────────────────────────────────────────

def waste_by_item(orders_df: pd.DataFrame, waste_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate waste % for each menu item.
    Waste % = (total leftover / total prep) * 100
    """
    merged = pd.merge(
        orders_df.groupby("item_name")["prep_qty"].sum(),
        waste_df.groupby("item_name")["leftover_qty"].sum(),
        left_index=True, right_index=True
    )
    merged["waste_pct"] = (merged["leftover_qty"] / merged["prep_qty"] * 100).round(2)
    merged["total_waste_kg"] = waste_df.groupby("item_name")["waste_kg"].sum()
    merged["total_waste_cost"] = waste_df.groupby("item_name")["waste_cost_inr"].sum()
    return merged.sort_values("waste_pct", ascending=False).reset_index()


def waste_by_category(orders_df: pd.DataFrame, waste_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate waste % grouped by category."""
    merged = pd.merge(
        orders_df.groupby("category")["prep_qty"].sum(),
        waste_df.groupby("category")["leftover_qty"].sum(),
        left_index=True, right_index=True
    )
    merged["waste_pct"] = (merged["leftover_qty"] / merged["prep_qty"] * 100).round(2)
    merged["total_waste_kg"] = waste_df.groupby("category")["waste_kg"].sum()
    merged["total_waste_cost"] = waste_df.groupby("category")["waste_cost_inr"].sum()
    return merged.sort_values("waste_pct", ascending=False).reset_index()


def waste_by_day_of_week(waste_df: pd.DataFrame) -> pd.DataFrame:
    """Average daily waste grouped by day of the week."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg = waste_df.groupby("day_of_week").agg(
        avg_waste_kg=("waste_kg", "mean"),
        total_waste_kg=("waste_kg", "sum"),
        total_waste_cost=("waste_cost_inr", "sum"),
    ).reindex(day_order).reset_index()
    agg["avg_waste_kg"] = agg["avg_waste_kg"].round(2)
    return agg


def waste_by_month(waste_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly waste totals and averages."""
    df = waste_df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    agg = df.groupby("month").agg(
        total_waste_kg=("waste_kg", "sum"),
        total_waste_cost=("waste_cost_inr", "sum"),
        avg_daily_waste_kg=("waste_kg", "mean"),
    ).reset_index()
    agg["total_waste_kg"] = agg["total_waste_kg"].round(2)
    agg["total_waste_cost"] = agg["total_waste_cost"].round(2)
    agg["avg_daily_waste_kg"] = agg["avg_daily_waste_kg"].round(2)
    return agg


# ──────────────────────────────────────────────
# Top Wasters
# ──────────────────────────────────────────────

def top_wasted_items(orders_df: pd.DataFrame, waste_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the top-n consistently wasted items based on average daily waste %.
    Consistency = high mean waste % with low coefficient of variation.
    """
    merged = pd.merge(
        orders_df[["date", "item_name", "prep_qty"]],
        waste_df[["date", "item_name", "leftover_qty", "waste_kg", "waste_cost_inr"]],
        on=["date", "item_name"]
    )
    merged["daily_waste_pct"] = (merged["leftover_qty"] / merged["prep_qty"] * 100)

    stats = merged.groupby("item_name")["daily_waste_pct"].agg(
        mean_waste_pct="mean",
        std_waste_pct="std",
    ).reset_index()
    stats["cv"] = stats["std_waste_pct"] / stats["mean_waste_pct"]  # Lower = more consistent
    # Score: high mean, low cv → consistently wasted
    stats["consistency_score"] = stats["mean_waste_pct"] * (1 / (1 + stats["cv"]))
    stats = stats.sort_values("consistency_score", ascending=False).head(n)
    stats["mean_waste_pct"] = stats["mean_waste_pct"].round(2)
    return stats.reset_index(drop=True)


# ──────────────────────────────────────────────
# Correlation Analysis
# ──────────────────────────────────────────────

def correlation_matrix(orders_df: pd.DataFrame, waste_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix between weather, events, staff, and waste metrics.
    """
    merged = pd.merge(
        orders_df[["date", "item_name", "prep_qty", "orders_qty",
                    "is_weekend", "is_holiday", "weather", "staff_count", "event_flag"]],
        waste_df[["date", "item_name", "waste_kg", "waste_cost_inr"]],
        on=["date", "item_name"]
    )

    # Encode weather as numeric
    weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3}
    merged["weather_encoded"] = merged["weather"].map(weather_map)

    numeric_cols = [
        "prep_qty", "orders_qty", "waste_kg", "waste_cost_inr",
        "is_weekend", "is_holiday", "weather_encoded", "staff_count", "event_flag"
    ]
    corr = merged[numeric_cols].corr().round(3)
    return corr


# ──────────────────────────────────────────────
# Weekly Waste Cost
# ──────────────────────────────────────────────

def weekly_waste_cost(waste_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate waste cost per week in INR."""
    df = waste_df.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    agg = df.groupby("week").agg(
        total_waste_kg=("waste_kg", "sum"),
        total_waste_cost=("waste_cost_inr", "sum"),
    ).reset_index()
    agg["total_waste_kg"] = agg["total_waste_kg"].round(2)
    agg["total_waste_cost"] = agg["total_waste_cost"].round(2)
    return agg


# ──────────────────────────────────────────────
# Anomaly Detection
# ──────────────────────────────────────────────

def detect_anomalies(waste_df: pd.DataFrame, window: int = 14, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect anomaly days where total daily waste exceeds
    rolling mean + (threshold × rolling std).
    
    Uses a 14-day rolling window by default.
    """
    daily = waste_df.groupby("date")["waste_kg"].sum().reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["rolling_mean"] = daily["waste_kg"].rolling(window=window, min_periods=1).mean()
    daily["rolling_std"] = daily["waste_kg"].rolling(window=window, min_periods=1).std().fillna(0)
    daily["upper_bound"] = daily["rolling_mean"] + threshold * daily["rolling_std"]
    daily["is_anomaly"] = daily["waste_kg"] > daily["upper_bound"]

    return daily


# ──────────────────────────────────────────────
# Waste Heatmap Data
# ──────────────────────────────────────────────

def waste_heatmap_data(waste_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table of average waste_kg: items (rows) vs day_of_week (columns).
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = waste_df.pivot_table(
        values="waste_kg",
        index="item_name",
        columns="day_of_week",
        aggfunc="mean"
    ).reindex(columns=day_order).round(2)
    return pivot


# ──────────────────────────────────────────────
# KPI Calculations
# ──────────────────────────────────────────────

def calculate_kpis(waste_df: pd.DataFrame) -> dict:
    """
    Calculate key performance indicators for the current and previous week.
    
    Returns a dict with:
        - total_waste_kg: this week's total waste
        - total_waste_cost: this week's total waste cost
        - pct_change: % change vs. last week
        - worst_item: highest waste item this week
    """
    max_date = waste_df["date"].max()
    week_start = max_date - timedelta(days=6)
    prev_week_start = week_start - timedelta(days=7)
    prev_week_end = week_start - timedelta(days=1)

    this_week = waste_df[(waste_df["date"] >= week_start) & (waste_df["date"] <= max_date)]
    prev_week = waste_df[(waste_df["date"] >= prev_week_start) & (waste_df["date"] <= prev_week_end)]

    tw_waste = this_week["waste_kg"].sum()
    tw_cost = this_week["waste_cost_inr"].sum()
    pw_waste = prev_week["waste_kg"].sum()

    pct_change = ((tw_waste - pw_waste) / pw_waste * 100) if pw_waste > 0 else 0.0

    worst = this_week.groupby("item_name")["waste_kg"].sum().idxmax() if len(this_week) > 0 else "N/A"

    return {
        "total_waste_kg": round(tw_waste, 2),
        "total_waste_cost": round(tw_cost, 2),
        "pct_change": round(pct_change, 2),
        "worst_item": worst,
        "week_range": f"{week_start.strftime('%d %b')} – {max_date.strftime('%d %b %Y')}",
    }


# ──────────────────────────────────────────────
# Item Drilldown
# ──────────────────────────────────────────────

def item_drilldown(orders_df: pd.DataFrame, waste_df: pd.DataFrame, item_name: str) -> pd.DataFrame:
    """
    Return monthly prep vs orders vs waste for a specific item.
    """
    merged = pd.merge(
        orders_df[orders_df["item_name"] == item_name][["date", "item_name", "prep_qty", "orders_qty"]],
        waste_df[waste_df["item_name"] == item_name][["date", "item_name", "leftover_qty", "waste_kg"]],
        on=["date", "item_name"]
    )
    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    agg = merged.groupby("month").agg(
        total_prep=("prep_qty", "sum"),
        total_orders=("orders_qty", "sum"),
        total_waste=("leftover_qty", "sum"),
        total_waste_kg=("waste_kg", "sum"),
    ).reset_index()
    return agg


# ──────────────────────────────────────────────
# Weekly Summary for AI Prompt
# ──────────────────────────────────────────────

def weekly_summary_text(orders_df: pd.DataFrame, waste_df: pd.DataFrame) -> str:
    """
    Generate a structured text summary of the latest week's data
    suitable for passing to an AI model.
    """
    kpis = calculate_kpis(waste_df)
    top_items = top_wasted_items(orders_df, waste_df, n=5)
    day_waste = waste_by_day_of_week(waste_df)

    lines = [
        f"=== WEEKLY WASTE REPORT ({kpis['week_range']}) ===",
        f"Total Waste: {kpis['total_waste_kg']} kg",
        f"Total Waste Cost: ₹{kpis['total_waste_cost']:,.0f}",
        f"Change vs Last Week: {kpis['pct_change']:+.1f}%",
        f"Worst Offender: {kpis['worst_item']}",
        "",
        "--- Top 5 Consistently Wasted Items ---",
    ]

    for _, row in top_items.iterrows():
        lines.append(f"  • {row['item_name']}: avg waste {row['mean_waste_pct']:.1f}%")

    lines.append("")
    lines.append("--- Average Daily Waste by Day ---")
    for _, row in day_waste.iterrows():
        lines.append(f"  • {row['day_of_week']}: {row['avg_waste_kg']:.2f} kg")

    cat_waste = waste_by_category(orders_df, waste_df)
    lines.append("")
    lines.append("--- Waste by Category ---")
    for _, row in cat_waste.iterrows():
        lines.append(f"  • {row['category']}: {row['waste_pct']:.1f}% waste rate, ₹{row['total_waste_cost']:,.0f} total cost")

    return "\n".join(lines)
