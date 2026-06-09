"""
SmartPlate — Synthetic Data Generator
Generates 2 years of realistic daily restaurant operations data
for 15 menu items across 4 categories with seasonal, weekend, 
holiday, and anomaly patterns baked in.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.utils import (
    MENU_ITEMS,
    get_all_items,
    get_item_category,
    get_data_dir,
    get_season,
    is_indian_holiday,
)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SEED = 42
DAYS = 730  # ~2 years
START_DATE = datetime(2024, 1, 1)

# Base prep quantities per category (daily per item)
BASE_PREP = {
    "Main Course": 45,
    "Starters": 55,
    "Breads": 80,
    "Desserts": 35,
}

# Cost per kg of waste by category (INR)
WASTE_COST_PER_KG = {
    "Main Course": 320,
    "Starters": 250,
    "Breads": 120,
    "Desserts": 280,
}

# Avg portion weight in kg
PORTION_WEIGHT_KG = {
    "Main Course": 0.35,
    "Starters": 0.20,
    "Breads": 0.08,
    "Desserts": 0.15,
}

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy", "Stormy"]


# ──────────────────────────────────────────────
# Generator Logic
# ──────────────────────────────────────────────

def generate_restaurant_data(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic restaurant order and waste data.

    Returns:
        (orders_df, waste_df) — Two DataFrames saved to data/generated/
    """
    np.random.seed(seed)
    random.seed(seed)

    all_items = get_all_items()
    records = []

    for day_offset in range(DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        day_of_week = current_date.strftime("%A")
        is_weekend = 1 if current_date.weekday() >= 5 else 0
        holiday = is_indian_holiday(current_date)
        is_holiday_flag = 1 if holiday else 0
        season = get_season(current_date)
        month = current_date.month

        # ── Weather generation (season-dependent) ──
        if season == "Monsoon":
            weather = np.random.choice(
                WEATHER_OPTIONS, p=[0.15, 0.25, 0.45, 0.15]
            )
        elif season == "Winter":
            weather = np.random.choice(
                WEATHER_OPTIONS, p=[0.40, 0.40, 0.15, 0.05]
            )
        elif season == "Summer":
            weather = np.random.choice(
                WEATHER_OPTIONS, p=[0.60, 0.25, 0.10, 0.05]
            )
        else:
            weather = np.random.choice(
                WEATHER_OPTIONS, p=[0.45, 0.35, 0.15, 0.05]
            )

        # ── Staff count (higher on weekends/holidays) ──
        base_staff = np.random.randint(8, 13)
        if is_weekend:
            base_staff += np.random.randint(2, 5)
        if holiday:
            base_staff += np.random.randint(1, 4)
        staff_count = min(base_staff, 20)

        # ── Event flag (festivals, weekends, random corporate) ──
        event_flag = 0
        if holiday:
            event_flag = 1
        elif is_weekend and np.random.random() < 0.25:
            event_flag = 1  # Weekend party / corporate event
        elif np.random.random() < 0.05:
            event_flag = 1  # Random midweek event

        # ── Generate data for each menu item ──
        for item_name in all_items:
            category = get_item_category(item_name)
            base = BASE_PREP[category]

            # Add item-level variance
            item_noise = hash(item_name) % 10 - 5
            prep_base = base + item_noise

            # ── Demand multipliers ──
            multiplier = 1.0

            # Weekend boost (+40%)
            if is_weekend:
                multiplier *= 1.40

            # Holiday/festival spike (+30-60%)
            if holiday:
                multiplier *= np.random.uniform(1.30, 1.60)

            # Event boost (+20%)
            if event_flag and not holiday:
                multiplier *= 1.20

            # Monsoon dip (-20% walk-ins)
            if season == "Monsoon":
                multiplier *= 0.80

            # Stormy weather further reduces (-15%)
            if weather == "Stormy":
                multiplier *= 0.85

            # Rainy weather slight dip (-8%)
            if weather == "Rainy":
                multiplier *= 0.92

            # Monthly seasonality (Dec-Jan festive boost, Jun-Jul dip)
            if month in (12, 1):
                multiplier *= 1.15
            elif month in (7, 8):
                multiplier *= 0.90

            # ── Prep quantity (restaurants prep based on expected demand) ──
            prep_qty = int(prep_base * multiplier * np.random.uniform(0.90, 1.15))
            prep_qty = max(prep_qty, 5)

            # ── Actual orders (slightly below prep, with noise) ──
            demand_ratio = np.random.uniform(0.65, 0.95)

            # Popular items sell better
            if item_name in ["Chicken Biryani", "Butter Naan", "Paneer Butter Masala"]:
                demand_ratio = np.random.uniform(0.80, 0.98)

            # Low-demand items
            if item_name in ["Mutton Rogan Josh", "Fish Amritsari", "Kulfi"]:
                demand_ratio = np.random.uniform(0.50, 0.80)

            orders_qty = int(prep_qty * demand_ratio)
            orders_qty = max(orders_qty, 1)

            # ── Anomaly injection (~3% of days) ──
            is_anomaly = np.random.random() < 0.03
            if is_anomaly:
                anomaly_type = np.random.choice(["oversupply", "undersupply"])
                if anomaly_type == "oversupply":
                    prep_qty = int(prep_qty * np.random.uniform(1.5, 2.0))
                else:
                    orders_qty = int(orders_qty * np.random.uniform(0.3, 0.5))

            # ── Leftover and waste ──
            leftover_qty = max(prep_qty - orders_qty, 0)
            portion_wt = PORTION_WEIGHT_KG[category]
            waste_kg = round(leftover_qty * portion_wt * np.random.uniform(0.7, 1.0), 2)
            waste_cost = round(waste_kg * WASTE_COST_PER_KG[category], 2)

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "item_name": item_name,
                "category": category,
                "prep_qty": prep_qty,
                "orders_qty": orders_qty,
                "leftover_qty": leftover_qty,
                "waste_kg": waste_kg,
                "waste_cost_inr": waste_cost,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday_flag,
                "weather": weather,
                "staff_count": staff_count,
                "event_flag": event_flag,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # ── Split into orders and waste DataFrames ──
    orders_df = df[[
        "date", "item_name", "category", "prep_qty", "orders_qty",
        "day_of_week", "is_weekend", "is_holiday", "weather",
        "staff_count", "event_flag"
    ]].copy()

    waste_df = df[[
        "date", "item_name", "category", "leftover_qty", "waste_kg",
        "waste_cost_inr", "day_of_week", "is_weekend", "is_holiday",
        "weather", "staff_count", "event_flag"
    ]].copy()

    # ── Save to CSV ──
    data_dir = get_data_dir()
    orders_df.to_csv(data_dir / "orders.csv", index=False)
    waste_df.to_csv(data_dir / "waste.csv", index=False)

    return orders_df, waste_df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load existing CSV data. If not found, generate fresh data first.

    Returns:
        (orders_df, waste_df)
    """
    data_dir = get_data_dir()
    orders_path = data_dir / "orders.csv"
    waste_path = data_dir / "waste.csv"

    if not orders_path.exists() or not waste_path.exists():
        return generate_restaurant_data()

    orders_df = pd.read_csv(orders_path, parse_dates=["date"])
    waste_df = pd.read_csv(waste_path, parse_dates=["date"])
    return orders_df, waste_df


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("[SmartPlate] Generating synthetic restaurant data...")
    orders, waste = generate_restaurant_data()
    print(f"[OK] Generated {len(orders):,} order records")
    print(f"[OK] Generated {len(waste):,} waste records")
    print(f"[DIR] Files saved to: {get_data_dir()}")
    print(f"\n[DATA] Date range: {orders['date'].min().date()} to {orders['date'].max().date()}")
    print(f"[DATA] Menu items: {orders['item_name'].nunique()}")
    print(f"[DATA] Categories: {orders['category'].nunique()}")
    print(f"\n[SAMPLE] Order record:\n{orders.iloc[0].to_dict()}")
    print(f"\n[SAMPLE] Waste record:\n{waste.iloc[0].to_dict()}")
