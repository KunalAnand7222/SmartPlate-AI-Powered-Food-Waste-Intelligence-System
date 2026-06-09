"""
SmartPlate -- Utility Functions
Shared helpers for formatting, date handling, and common operations.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────
# Indian Number System Formatter
# ──────────────────────────────────────────────

def format_inr(amount: float, prefix: str = "₹") -> str:
    """
    Format a number using the Indian numbering system.
    Examples:
        1234       → ₹1,234
        100000     → ₹1,00,000
        12345678   → ₹1,23,45,678
    """
    amount = round(amount, 2)
    is_negative = amount < 0
    amount = abs(amount)

    integer_part = int(amount)
    decimal_part = amount - integer_part

    s = str(integer_part)

    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        remaining = s[:-3]
        # Insert commas every 2 digits from the right in the remaining part
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last_three

    if decimal_part > 0:
        decimal_str = f"{decimal_part:.2f}"[1:]  # Remove leading 0
        formatted += decimal_str

    sign = "-" if is_negative else ""
    return f"{sign}{prefix}{formatted}"


# ──────────────────────────────────────────────
# Date Helpers
# ──────────────────────────────────────────────

INDIAN_HOLIDAYS = [
    # Major Indian holidays (month, day) — approximate fixed dates
    (1, 26),   # Republic Day
    (3, 29),   # Holi (approx)
    (4, 14),   # Ambedkar Jayanti / Tamil New Year
    (5, 1),    # May Day
    (8, 15),   # Independence Day
    (9, 7),    # Janmashtami (approx)
    (10, 2),   # Gandhi Jayanti
    (10, 12),  # Dussehra (approx)
    (10, 24),  # Diwali (approx)
    (10, 25),  # Diwali Day 2
    (11, 1),   # Kannada Rajyotsava / All Saints
    (11, 14),  # Children's Day
    (12, 25),  # Christmas
]


def is_indian_holiday(date: datetime) -> bool:
    """Check if a given date falls on an approximate Indian holiday."""
    return (date.month, date.day) in INDIAN_HOLIDAYS


def get_season(date: datetime) -> str:
    """Return the Indian season/weather label for a given date."""
    month = date.month
    if month in (6, 7, 8, 9):
        return "Monsoon"
    elif month in (11, 12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    else:
        return "Autumn"


# ──────────────────────────────────────────────
# Path Helpers
# ──────────────────────────────────────────────

def get_project_root() -> Path:
    """Return the project root directory (one level up from src/)."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the generated data directory path, creating it if needed."""
    data_dir = get_project_root() / "data" / "generated"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_model_path() -> Path:
    """Return the path for the saved ML model."""
    models_dir = get_project_root() / "data" / "generated"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "model.pkl"


# ──────────────────────────────────────────────
# Data Helpers
# ──────────────────────────────────────────────

MENU_ITEMS = {
    "Main Course": [
        "Paneer Butter Masala", "Dal Makhani", "Chicken Biryani",
        "Mutton Rogan Josh"
    ],
    "Starters": [
        "Paneer Tikka", "Chicken Tandoori", "Veg Spring Rolls",
        "Fish Amritsari"
    ],
    "Breads": [
        "Butter Naan", "Garlic Naan", "Tandoori Roti"
    ],
    "Desserts": [
        "Gulab Jamun", "Ras Malai", "Kheer", "Kulfi"
    ],
}


def get_all_items() -> list:
    """Return a flat list of all menu item names."""
    items = []
    for category, item_list in MENU_ITEMS.items():
        items.extend(item_list)
    return items


def get_item_category(item_name: str) -> str:
    """Return the category for a given menu item."""
    for category, items in MENU_ITEMS.items():
        if item_name in items:
            return category
    return "Unknown"


# ──────────────────────────────────────────────
# Chart Color Palette (Dark-theme friendly)
# ──────────────────────────────────────────────

COLORS = {
    "primary": "#C49A6C",
    "secondary": "#D4806A",
    "accent": "#7BA688",
    "warning": "#D4AA7C",
    "danger": "#C45D4D",
    "info": "#5B9EA6",
    "bg_dark": "#15130F",
    "bg_card": "#1C1A17",
    "text_light": "#E8E0D8",
    "gradient": ["#C49A6C", "#7BA688", "#D4806A", "#5B9EA6", "#D4AA7C",
                  "#8B7355", "#A8C5A0", "#B07D6A", "#6B9BA3", "#C9A882"],
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#A89F96", "family": "DM Sans, sans-serif"},
        "xaxis": {"gridcolor": "rgba(196,154,108,0.06)"},
        "yaxis": {"gridcolor": "rgba(196,154,108,0.06)"},
    }
}


def style_plotly_fig(fig):
    """Apply consistent warm-theme styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A89F96", family="DM Sans, sans-serif", size=12),
        xaxis=dict(gridcolor="rgba(196,154,108,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(196,154,108,0.06)", zeroline=False),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8C8279"),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
