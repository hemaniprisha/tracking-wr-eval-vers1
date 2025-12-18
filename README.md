# 🎯 Tracking Prophet

**Predicting NFL Wide Receiver Success with In-Game Tracking Data**

*Because 40-times don't tell the whole story.*

---

## The Problem

NFL teams spend **$8M+** on rookie WR contracts with a **40-50% bust rate**. The NFL Combine has been the evaluation standard for decades, but combine metrics (40-yard dash, vertical jump, 3-cone drill) achieve an **R² of -0.155** when predicting NFL performance—worse than random guessing.

**Why?** Static tests in shorts and a t-shirt don't capture real football skills: route precision, separation against live defenders, or cutting ability in game situations.

---

## The Solution

This project uses **college in-game tracking data** to predict:
1. **Draft Position** — What do scouts actually value?
2. **NFL Rookie Performance** — What actually predicts success?

### Results

| Analysis | R² Score | Improvement vs Combine |
|----------|----------|------------------------|
| Draft Prediction | **0.689** | 544% better |
| NFL Rookie Performance | **0.120** | 177% better |

**Key Finding:** Scouts overweight raw speed, but **cutting ability** (speed through route breaks) is the strongest predictor of NFL success.

---

## Project Structure

```
tracking-prophet/
├── data/
│   └── tracking_with_outcomes.csv    # Raw tracking data
├── src/
│   ├── tracking_analysis_draft.py    # Draft prediction pipeline
│   ├── tracking_analysis_nfl.py      # NFL performance pipeline
│   ├── advanced_modeling.py          # ML model comparison & tuning
│   └── tracking_visuals.py           # Visualization generation
├── results/
│   ├── visualizations/               # Generated charts
│   ├── reports/                      # Executive summary, model comparison
│   └── data/                         # Processed data exports
├── run_analysis_draft.py             # Main runner for draft analysis
├── run_analysis_nfl.py               # Main runner for NFL analysis
├── streamlit_app.py                  # Interactive dashboard
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Analysis

**Draft Prediction:**
```bash
python run_analysis_draft.py --data data/tracking_with_outcomes.csv --output results/
```

**NFL Rookie Performance:**
```bash
python run_analysis_nfl.py --data data/tracking_with_outcomes.csv --output results/
```

### 3. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

---

## Features Engineered

15 football-meaningful features derived from raw tracking data:

| Category | Features |
|----------|----------|
| **Athleticism** | Speed Score, Burst Rate, First Step Quickness, Brake Rate |
| **Route Running** | Route Diversity, Separation Consistency, Sharp Cut Ability, Route Bend Ability |
| **Change of Direction** | Cut Separation, COD Separation Generated |
| **Workload** | Distance per Play, Explosive Rate, Total Plays |
| **Raw Metrics** | Max Speed 99, Change Direction Route Mean |

---

## Models Compared

| Model | Draft R² | NFL R² | Notes |
|-------|----------|--------|-------|
| XGBoost (Tuned) | **0.689** | -0.178 | Best for large samples |
| Random Forest | 0.677 | 0.084 | Solid baseline |
| Gradient Boosting | 0.663 | 0.025 | High overfit risk |
| LightGBM | 0.677 | -0.190 | Fast but overfit |
| ElasticNet | 0.366 | **0.120** | Best for small samples |
| Ridge | 0.375 | 0.101 | Simple baseline |

**Lesson:** Match model complexity to sample size. Tree models overfit with only 55 NFL samples.

---

## Key Insights

### What Scouts Value (Draft Model)
1. Speed Score (0.20)
2. Max Speed (0.12)
3. Separation Consistency (0.12)

### What Predicts NFL Success
1. **Sharp Cut Ability (0.51)** ← Dominates
2. Max Speed (0.39)
3. Route Bending (0.34)

**The Gap = Opportunity:** Players with elite cutting but average 40-times may be undervalued in the draft.

---

## Limitations

- **Small NFL sample** (55 players) limits confidence
- **Can't capture intangibles** (leadership, work ethic, football IQ)
- **NFL success factors unmeasured:** opportunity, QB quality, scheme fit, injuries
- **Single college season** (2022) → NFL rookies (2023-24)

---

## Tech Stack

- **Data Processing:** Pandas, NumPy
- **ML Models:** Scikit-learn, XGBoost, LightGBM
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **NFL Data:** nfl_data_py, rapidfuzz (name matching)

---

## Author

**Prisha Hemani**  
[GitHub](https://github.com/hemaniprisha) • [LinkedIn](https://www.linkedin.com/in/prisha-hemani-4194a8257/) • hemaniprisha1@gmail.com

