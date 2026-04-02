# 📊 Price Sense AI

**AI-powered promotion analysis for mid-market retailers.**

> "Should I run this promotion?" — answered in seconds with projected lift, cannibalization impact, profit estimate, and risk analysis.

**Live demo:** https://gd1302-price-sense-ai.streamlit.app/

**Write-up:** [WRITEUP.md](WRITEUP.md) — design decisions, architecture, and case study write-up

## Quick Start (Local)

```bash
# 1. Clone or download this repo
git clone https://github.com/ZTK13/price-sense-ai.git
cd price-sense-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
price-sense-ai/
├── app.py                    # Main Streamlit UI
├── simulator.py              # Simulation engine (elasticity, cannibalization, waterfall, risk)
├── ml_model.py               # RandomForest cross-validation layer
├── data_utils.py             # Synthetic training data generator
├── requirements.txt
├── README.md
└── WRITEUP.md                # Design decisions, architecture, and case study write-up
```

## How It Works

The simulation engine uses established retail economics principles:

- **Price Elasticity**: Category-specific demand curves that model how price changes affect volume
- **Cannibalization Modeling**: Estimates how much "lift" is just shifted from adjacent products
- **Pantry Loading**: Accounts for customers stockpiling, which suppresses post-promo sales
- **Profit Waterfall**: Full financial breakdown from baseline → volume lift → discount cost → cannibalization → net impact
- **Sensitivity Analysis**: Sweeps discount levels to find the profit-optimal point
- **Risk Scoring**: Multi-factor risk assessment with actionable mitigations
- **ML Cross-Check**: A RandomForest model trained on 2,000 synthetic scenarios provides an independent estimate to validate the simulation output

## Supported Categories

Grocery, Specialty Food & Nuts, Beverages, Snacks & Confectionery, Dairy & Refrigerated, Health & Wellness, Household & Cleaning

## Tech Stack

- **Frontend**: Streamlit (Python)
- **Visualization**: Plotly
- **Analytics**: NumPy, Pandas
- **ML**: scikit-learn (RandomForestRegressor)
- **Deployment**: Streamlit Cloud

---

Built as a case study submission demonstrating AI-partnered product development. 
See the [problem statement](trinamix_case_study.pdf) and [writeup](WRITEUP.md) for more details.
