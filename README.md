# 🍽️ SmartPlate — AI-Powered Food Waste Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-006600?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🚨 The Problem

> **Restaurant food waste costs India ₹92,000 crore (~$11 billion) every year.**

Indian restaurants waste an estimated 40% of the food they prepare daily. This isn't just a financial problem — it's an environmental and ethical crisis. Most restaurants lack data-driven tools to predict demand, optimize prep quantities, and reduce waste systematically.

**SmartPlate** solves this by combining **data analytics**, **machine learning forecasting**, and **AI-powered recommendations** into a single, beautiful dashboard that restaurant managers can use every day.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **KPI Dashboard** | Real-time waste metrics — total waste (kg), cost (INR), weekly trends, worst offenders |
| 🗓️ **Waste Heatmap** | Visual matrix of items × days showing waste hotspots |
| 📉 **Anomaly Detection** | Automatically flags unusual waste days using rolling statistics |
| 🔍 **Item Drilldown** | Deep dive into any menu item's prep vs. orders vs. waste |
| 🔮 **7-Day Forecast** | ML-powered waste predictions using Random Forest & XGBoost |
| 🧠 **AI Recommendations** | Ollama-powered (local LLM) actionable insights from your waste data |
| 🎮 **What-If Simulator** | Simulate prep reductions and see projected INR savings |
| 🌙 **Dark Theme** | Beautiful dark-mode-compatible charts and UI |
| 🇮🇳 **INR Formatting** | All costs in Indian number system (₹1,00,000) |

---

## 🏗️ Architecture

```
smartplate/
├── data/generated/          # Generated CSV datasets
│   ├── orders.csv
│   └── waste.csv
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic data with realistic patterns
│   ├── analysis.py          # Waste analytics & anomaly detection
│   ├── ml_model.py          # RF + XGBoost training & forecasting
│   ├── ai_insights.py       # Ollama local LLM integration
│   └── utils.py             # Shared utilities & formatters
├── app.py                   # Streamlit dashboard (7 sections)
├── requirements.txt         # Python dependencies
├── .env                     # Ollama config (not committed)
└── README.md                # This file
```

---

## 🛠️ Tech Stack

- **Data & Analysis**: Pandas, NumPy, Seaborn, Matplotlib
- **Visualization**: Plotly, Streamlit
- **Machine Learning**: scikit-learn (Random Forest), XGBoost
- **AI Insights**: Ollama (local LLM — llama3.2, mistral, etc.)
- **Utilities**: joblib, python-dotenv

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smartplate.git
cd smartplate
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install & Run Ollama (Free, Local)

Download Ollama from [ollama.com](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2
```

The `.env` file is pre-configured:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

> 💡 **Ollama not running?** The dashboard still works — you'll get smart static recommendations as a fallback. You can also change the model to `mistral`, `gemma2`, etc.

### 5. Generate Data (Auto or Manual)

Data is auto-generated on first run. To generate manually:

```bash
python -m src.data_generator
```

### 6. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` 🎉

---

## 📸 Screenshots

> Screenshots will be added after first deployment.

| Section | Preview |
|---------|---------|
| KPI Cards | _Coming soon_ |
| Waste Heatmap |
| Anomaly Detection |
| 7-Day Forecast ![Uploading image.png…]()|
| AI Recommendations ![Uploading image.png…]()|
| What-If Simulator ![Uploading image.png…]()|

---

## 📊 Data Details

The synthetic data generator creates **2 years** of daily restaurant data with:

- **15 menu items** across 4 categories (Main Course, Starters, Breads, Desserts)
- **Realistic patterns**: Weekend +40% demand, monsoon dips, festival spikes
- **Anomaly injection**: ~3% of days have unusual overproduction or demand drops
- **Weather correlation**: Rainy/stormy days reduce walk-in customers
- **Staff variation**: Higher staff on weekends and holidays

---

## 🤖 ML Model Details

| Model | Features | Target |
|-------|----------|--------|
| Random Forest (200 trees) | day_of_week, month, is_weekend, is_holiday, weather, 7-day avg waste, staff, item, event | next-day waste_kg |
| XGBoost (200 rounds) | Same features | Same target |

The best model (by MAE) is automatically selected and saved as `model.pkl`.


---

<p align="center">
  🍽️ <strong>SmartPlate</strong> — Reducing restaurant food waste, one prediction at a time. 🌱
</p>
