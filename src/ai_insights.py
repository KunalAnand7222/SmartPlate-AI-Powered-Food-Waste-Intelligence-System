"""
SmartPlate -- AI Insights Module
Uses Grok API (xAI) to generate actionable restaurant operations
recommendations from weekly waste data summaries.

API keys can be configured via GROK_API_KEY environment variable.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")
GROK_API_KEY_ENV = os.getenv("GROK_API_KEY", "")


# ──────────────────────────────────────────────
# Fallback Static Recommendations
# ──────────────────────────────────────────────

FALLBACK_RECOMMENDATIONS = {
    "executive_summary": (
        "Based on historical patterns, your restaurant shows consistent overproduction "
        "in certain categories, particularly on weekdays when footfall is lower. "
        "Implementing demand-driven prep schedules could reduce food waste by an estimated "
        "15-25% and save approximately ₹40,000–₹60,000 per month."
    ),
    "recommendations": [
        {
            "title": "Implement Dynamic Prep Scheduling",
            "detail": (
                "Reduce prep quantities for low-demand items (Mutton Rogan Josh, Fish Amritsari, Kulfi) "
                "by 20-30% on weekdays. Use the previous week's order data as a baseline. "
                "Expected monthly savings: Rs 15,000-25,000."
            ),
        },
        {
            "title": "Introduce a Waste Tracking Dashboard for Kitchen Staff",
            "detail": (
                "Display daily waste metrics on a kitchen screen. Research shows that real-time "
                "visibility reduces waste by 10-15% through staff accountability. Assign a daily "
                "'waste champion' who reviews end-of-day numbers."
            ),
        },
        {
            "title": "Launch a 'Specials Menu' for Surplus Ingredients",
            "detail": (
                "Convert predicted surplus items into discounted daily specials or combo meals. "
                "For example, excess paneer from Paneer Butter Masala prep can be diverted into "
                "Paneer Wraps or Paneer Rice Bowls during lunch hours."
            ),
        },
    ],
}


# ──────────────────────────────────────────────
# Grok API Integration
# ──────────────────────────────────────────────

def get_ai_recommendations(weekly_summary: str, api_key: str = "") -> dict:
    """
    Send the weekly waste summary to Grok API and get structured recommendations.

    Args:
        weekly_summary: Structured text summary of the week's waste data.
        api_key: Optional Grok API key provided by the user via UI.

    Returns:
        dict with keys:
            - executive_summary (str): 1-paragraph overview
            - recommendations (list[dict]): 3 items, each with 'title' and 'detail'
    """
    effective_api_key = api_key if api_key else GROK_API_KEY_ENV

    if not effective_api_key:
        return FALLBACK_RECOMMENDATIONS

    try:
        system_prompt = (
            "You are a seasoned restaurant operations consultant specializing in food waste reduction "
            "and cost optimization for Indian restaurants. You have deep expertise in menu engineering, "
            "demand forecasting, kitchen workflow optimization, and sustainable practices. "
            "Your advice is always specific, actionable, and backed by data. "
            "You understand the Indian restaurant context: INR pricing, Indian cuisine items, "
            "monsoon/festival seasonality, and local supply chains."
        )

        user_prompt = f"""Analyze the following weekly food waste report from a restaurant and provide your expert recommendations.

{weekly_summary}

Please respond in EXACTLY this format (use these exact headers):

EXECUTIVE SUMMARY:
[Write exactly ONE paragraph summarizing the key findings and overall waste situation. Be specific with numbers.]

RECOMMENDATION 1:
Title: [concise title, no emojis]
Detail: [2-3 sentences with specific, actionable steps and expected impact in INR]

RECOMMENDATION 2:
Title: [concise title, no emojis]
Detail: [2-3 sentences with specific, actionable steps and expected impact in INR]

RECOMMENDATION 3:
Title: [concise title, no emojis]
Detail: [2-3 sentences with specific, actionable steps and expected impact in INR]"""

        headers = {
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{GROK_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": GROK_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30,
        )

        if response.status_code != 200:
            return FALLBACK_RECOMMENDATIONS

        data = response.json()
        reply_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not reply_text:
            return FALLBACK_RECOMMENDATIONS

        return _parse_llm_response(reply_text)

    except requests.ConnectionError:
        return FALLBACK_RECOMMENDATIONS
    except requests.Timeout:
        return FALLBACK_RECOMMENDATIONS
    except Exception:
        return FALLBACK_RECOMMENDATIONS


# ──────────────────────────────────────────────
# Response Parser
# ──────────────────────────────────────────────

def _parse_llm_response(text: str) -> dict:
    """
    Parse the LLM's structured text response into a dict.
    Falls back to static recommendations if parsing fails.
    """
    try:
        result = {"executive_summary": "", "recommendations": []}

        # Extract executive summary
        if "EXECUTIVE SUMMARY:" in text:
            parts = text.split("EXECUTIVE SUMMARY:")
            if len(parts) > 1:
                summary_block = parts[1]
                # Find the next RECOMMENDATION header
                if "RECOMMENDATION 1:" in summary_block:
                    result["executive_summary"] = summary_block.split("RECOMMENDATION 1:")[0].strip()
                else:
                    result["executive_summary"] = summary_block.strip()

        # Extract recommendations
        for i in range(1, 4):
            header = f"RECOMMENDATION {i}:"
            next_header = f"RECOMMENDATION {i + 1}:" if i < 3 else None

            if header in text:
                block = text.split(header)[1]
                if next_header and next_header in block:
                    block = block.split(next_header)[0]

                title = ""
                detail = ""

                if "Title:" in block and "Detail:" in block:
                    title = block.split("Title:")[1].split("Detail:")[0].strip()
                    detail = block.split("Detail:")[1].strip()

                    # Clean up any trailing recommendation headers
                    for cleanup_header in ["RECOMMENDATION", "EXECUTIVE"]:
                        if cleanup_header in detail:
                            detail = detail.split(cleanup_header)[0].strip()

                if title and detail:
                    result["recommendations"].append({
                        "title": title,
                        "detail": detail,
                    })

        # Validate: must have summary and 3 recommendations
        if not result["executive_summary"] or len(result["recommendations"]) < 3:
            return FALLBACK_RECOMMENDATIONS

        return result

    except Exception:
        return FALLBACK_RECOMMENDATIONS


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from src.data_generator import load_data
    from src.analysis import weekly_summary_text

    print("[SmartPlate] AI Insights Module (Grok API)")
    print("=" * 50)

    effective_key = GROK_API_KEY_ENV
    if effective_key:
        print(f"[OK] Grok API Key found. Model: {GROK_MODEL}")
    else:
        print(f"[WARN] No Grok API Key found in environment variables.")
        print("       Using fallback recommendations.")

    orders_df, waste_df = load_data()
    summary = weekly_summary_text(orders_df, waste_df)

    print(f"\n[SUMMARY] Weekly Summary (sent to {GROK_MODEL}):")
    print(summary)

    print("\n[AI] Getting AI recommendations...")
    recs = get_ai_recommendations(summary)

    print(f"\n[EXEC] Executive Summary:\n{recs['executive_summary']}")
    print("\n[RECS] Recommendations:")
    for i, rec in enumerate(recs["recommendations"], 1):
        print(f"\n  {i}. {rec['title']}")
        print(f"     {rec['detail']}")

