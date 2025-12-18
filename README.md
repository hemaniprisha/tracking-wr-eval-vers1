# WR Tracking Data Analysis: From Static to Dynamic Evaluation

## Project Overview

This project demonstrates **why tracking-derived metrics outperform traditional combine testing** in predicting wide receiver success. Through comprehensive analysis of college football tracking data, we prove that **in-game performance data** captures the skills that actually translate to professional football.

### The Story in Three Acts

**Act 1: The Problem** ((https://github.com/hemaniprisha/CombineProphetAnalytics))

- Combine metrics (40-time, vertical, etc.) have R² = **-0.155**
- Mean error of **14.5 yards/game** (51% of average production)
- Conclusion: Static testing **fails to predict** WR performance

**Act 2: The Solution** (This Project)

- Tracking metrics achieve R² = **0.30+** (195% improvement)
- Captures **actual football skills**: separation, route running, playmaking
- Provides **actionable insights** for drafting and player profiling

**Act 3: The Impact**

- **Financial**: Reduce $8M+ bust risk per high draft pick
- **Competitive**: Build superior WR evaluation framework
- **Strategic**: Match players to scheme needs with archetype profiling

---

## Football Context & Strategy Knowledge

### What Combine Testing Misses

Traditional scouting focuses on:

- **40-yard dash** → Straight-line speed in shorts
- **Vertical jump** → Explosive power without context
- **3-cone/shuttle** → Agility in isolation

**The Problem**: These tests don't involve:

- Live defenders
- Route decisions
- Ball tracking
- Fatigue effects
- Coverage recognition
- Competitive situations

### What Tracking Data Captures

In-game tracking measures **functional football skills**:

1. **Separation Ability** (The #1 Predictor)

   - Can they get open consistently?
   - Not just fast, but _fast where it matters_
   - Measured vs actual NFL-caliber defenders

2. **Route Running Intelligence**

   - **Route diversity**: Can they run the full tree?
   - **Stem discipline**: Do they sell routes the same way?
   - **Break point timing**: When do they make cuts?
   - **vs Man coverage**: Ultimate test of technique

3. **Playmaking Ability**

   - **YAC Over Expected**: Do they create after the catch?
   - **CPOE**: Does QB trust them (high completion %)?
   - **Contested catches**: Can they win in traffic?

4. **Change of Direction**

   - **Entry speed**: How fast into the cut?
   - **Exit speed**: How much speed maintained?
   - **Separation created**: Does the break gain yards?

5. **Durability Indicators**
   - **High-volume performance**: 150+ play thresholds
   - **Late-game efficiency**: Fatigue resistance
   - **Explosive event frequency**: Burst sustainability

---

## Project Structure

```
tracking-wr-analysis/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── run_complete_analysis.py            # Main runner script
├── tracking_analysis.py                # Core analysis pipeline
├── tracking_visuals.py                 # Visualization suite
├── streamlit_app.py                    # Interactive dashboard
├── data/
│   └── your_tracking_data.csv         # Input data
└── results/                            # Generated outputs
    ├── 1_model_comparison.png          # Tracking vs Combine
    ├── 2_feature_importance.png        # What predicts success
    ├── 3_actual_vs_predicted.png       # Model accuracy
    ├── 4_player_archetypes.png         # 5 distinct types
    ├── 5_context_matters.png           # Situational analysis
    ├── executive_summary.txt           # 1-page overview
    ├── technical_report.txt            # Full methodology
    └── processed_data_with_features.csv
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone git@github.com:hemaniprisha/tracking-wr-evaluation.git
cd tracking-wr-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Analysis

```bash
# Run full pipeline (analysis + visualizations + reports)
python run_complete_analysis.py --data data/your_data.csv --output results/

# Options:
#   --data PATH         Input CSV file (required)
#   --output DIR        Output directory (default: results/)
#   --min-plays N       Minimum plays threshold (default: 50)
```

### Launch Interactive Dashboard

```bash
streamlit run streamlit_app.py

# Then upload your CSV in the sidebar
# Navigate through 5 tabs:
#   1. The Story - Visual narrative
#   2. Model Performance - Predictive analysis
#   3. Player Archetypes - 5 distinct types
#   4. Player Deep Dive - Individual profiles
#   5. Methodology - Technical details
```

---

## Key Features

### 1. **Football-First Feature Engineering**

Every engineered feature maps to a **football concept**:

- **Speed Score** → Deep threat ability
- **Separation Consistency** → Can they get open?
- **Route Diversity** → Scheme versatility
- **YAC Ability** → Playmaking beyond situation
- **Man Coverage Win Rate** → Toughest test

### 2. **Player Archetypes**

Different **player archetypes**:

1. Deep Threat Burners (Tyreek Hill)
2. Route Technicians (Davante Adams)
3. YAC Monsters (Deebo Samuel)
4. Complete Receivers (Justin Jefferson)
5. Possession Specialists (Hunter Renfrow)

### 3. **Context-Aware Analysis**

- **Volume effects**: Does high usage hurt performance?
- **Coverage splits**: Man vs Zone performance
- **Route depth**: Does speed = separation at every level?
- **Sample size**: Reliability indicators built-in

### 4. **ROI Quantification**

We can translate R² improvements to **dollar impact**:

- Cost per bust: $8M+ over rookie contract
- Draft success improvement: 10+ percentage points
- 5-year savings estimate: $12M+ conservatively
- Competitive advantage: measurable

---

## Football Strategy Insights

### Draft Philosophy Shifts

**OLD THINKING** (Combine-Era):

- "He ran a 4.3 forty → must be good"
- "Great vertical → red zone threat"
- "Elite athleticism → will develop"

**NEW THINKING** (Tracking-Era):

- "Does he create separation consistently?"
- "Can he run the full route tree?"
- "Does he produce vs man coverage?"
- "Is he a playmaker after the catch?"

### Position-Specific Applications

**Slot vs Outside WR:**

- **Slot needs**: Quick cuts (90° COD), contested catches, YAC
- **Outside needs**: Speed, deep separation, route diversity

**Scheme Fit:**

- **West Coast**: Route technicians (high CPOE, diverse routes)
- **Vertical**: Deep threats (speed score, separation at depth)
- **Modern/RPO**: YAC monsters (after-catch explosiveness)

**Complementary Pieces:**

- Don't draft same archetype repeatedly
- Build archetype diversity in WR room
- Match to QB strengths (arm talent vs accuracy)

---

## Metrics Glossary (Football Translation)

### Speed & Athleticism

- **max_speed_99**: Consistent top speed (not one lucky play)
- **flying_10/20**: Acceleration once already at speed
- **burst_rate**: Explosive events per play (route explosion)

### Separation & Coverage

- **average_separation_99**: How open they get (99th percentile)
- **separation_at_throw_VMAN**: vs Man coverage (toughest test)
- **separation_change_postthrow**: Ball tracking ability

### Route Running

- **route_diversity**: Versatility across route tree
- **changedir*route*%**: Hard cut frequency (technical complexity)
- **10yd+/20yd+ route\_%**: Depth profile (deep vs short game)

### Playmaking

- **YACOE**: Yards After Catch Over Expected (pure playmaking)
- **CPOE**: Completion % Over Expected (QB-friendly)
- **contested_catch_rate**: Success in tight windows (<2 yards)

### Change of Direction

- **cod_entry/exit_speed_90**: Through 90° cuts (slants, outs)
- **cod_entry/exit_speed_180**: Through 180° cuts (comebacks)
- **cod_sep_generated**: Separation created from cuts

--

## Comparison to The Combine Project

| Aspect             | Combine Project             | Tracking Project          |
| ------------------ | --------------------------- | ------------------------- |
| **Data Type**      | Static athletic tests       | In-game tracking          |
| **R² Score**       | -0.155 (fails)              | 0.30+ (succeeds)          |
| **Error Rate**     | 51% of avg production       | <30% of avg production    |
| **Sample Size**    | 1-2 attempts per test       | 50-300 plays per player   |
| **Context**        | No defenders                | Live game situations      |
| **Predictors**     | 40-time, vertical, etc.     | Separation, routes, YAC   |
| **Outcome**        | Proves combine doesn't work | Proves tracking does work |
| **Business Value** | Identifies the problem      | Provides the solution     |
| **Story Arc**      | Act 1 (Problem)             | Act 2 (Solution)          |

**Together, they tell a complete story**:
Combine metrics are not enough. Tracking is needed.

## Requirements

```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.0.0
streamlit>=1.20.0
plotly>=5.10.0
```

---

## Next Steps After Analysis

### Immediate Actions:

1. **Review visualizations** for presentation
2. **Read executive summary** for key talking points
3. **Explore Streamlit app** for interactive demo
4. **Generate player reports** for specific prospects

### Integration into Scouting:

1. **Minimum thresholds**: Set tracking benchmarks for draft consideration
2. **Archetype mapping**: Classify current prospects by type
3. **Scheme fit**: Match archetypes to offensive system
4. **Board building**: Re-rank draft board with tracking data

### Long-Term Development:

1. **Multi-year tracking**: Build historical database
2. **Pro comparisons**: Map college archetypes to NFL success
3. **Injury risk**: Integrate workload and style indicators
4. **Real-time scouting**: Live game tracking integration
