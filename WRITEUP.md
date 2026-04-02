# Price Sense AI — Case Study Write-Up

## 1. How did I decide the features in the POC?

I started from the core user question in the brief: *"Should I run this promotion?"* — and worked backwards to identify what a retailer would need to feel confident acting on that answer.

**Feature prioritization framework:**

The brief specifies the output must include projected lift, cannibalization impact, profit estimate, and risk. I organized these into a decision hierarchy that mirrors how a category manager actually thinks:

- **First: Give me the answer.** The recommendation card (Run / Caution / Don't Run) with a confidence score is the hero element. Retailers don't want to wade through dashboards — they want a verdict.
- **Second: Show me why.** Four key metrics at the top: volume lift, promo profit impact, net 30-day impact, and risk score. These are the numbers that justify the recommendation.
- **Third: Let me explore.** Tabbed deep-dives — volume forecast chart, profit waterfall, sensitivity analysis, and full data tables — for the user who wants to understand the mechanics.
- **Fourth: Tell me what to do about it.** Risk factors with severity scores, and actionable mitigations. This is where the tool becomes genuinely useful beyond what a spreadsheet can do.

**What I deliberately excluded (and why):**

- Historical promo performance comparison (requires real data pipeline)
- Multi-product promo optimization (complexity explosion for a POC)
- Calendar/scheduling features (adjacent feature, not core to the decision)
- Real-time inventory integration (infrastructure-heavy, low demo value)

The goal was to build something that feels complete for the core decision, not a half-built version of a full platform.

---

## 2. How did I go about creating the product?

**Process (roughly in order):**

1. **Deconstructed the problem domain.** Read through retail promotion economics literature and identified the key analytical components: price elasticity, cannibalization, pantry loading, post-promotion dips, margin mechanics.

2. **Designed the simulation engine first.** Before any UI work, I built `simulator.py` with realistic retail math. The engine encodes category-specific profiles (elasticity ranges, margin benchmarks, cannibalization rates) derived from industry norms. This ensures the output looks credible to someone who knows retail.

3. **Made it horizontal by design.** The category profiles system (7 categories, each with distinct elasticity curves, margin structures, and consumer behaviors) means the tool works for a beverage brand or a grocery chain with zero code changes — just different default parameters.

4. **Built the UI for a demo audience.** Used Streamlit for rapid development, Plotly for interactive charts, and custom CSS for a polished look. The sidebar input form → main analysis output pattern is instantly familiar to anyone who's used a SaaS analytics tool.

5. **Validated with the reference scenario.** The "25% off Salted Pistachios 16oz" scenario produces a "Proceed with Caution" recommendation, which is the correct answer — a 25% discount on a 40% margin product with 4 competitors and high cannibalization should not be an easy yes.

**AI tools used:** Claude for architecture decisions, simulation logic, code generation, debugging, and this write-up. The entire project was built collaboratively with Claude as an AI pair-programmer.

---

## 3. Technical Architecture

### Current Architecture (POC)

```
┌─────────────────────────────────────────┐
│           Streamlit Frontend            │
│  (app.py — sidebar inputs, charts, UI)  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         Simulation Engine               │
│  (simulator.py)                         │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Category     │  │ Promotion       │  │
│  │ Profiles     │  │ Type Config     │  │
│  │ (dict-based) │  │ (dict-based)    │  │
│  └──────────────┘  └─────────────────┘  │
│                                         │
│  PromotionInput → PromotionSimulator    │
│                 → PromotionResult       │
│                                         │
│  Components:                            │
│  - Price elasticity calculator          │
│  - Cannibalization modeler              │
│  - Profit waterfall generator           │
│  - Post-promo dip estimator             │
│  - Risk scorer + mitigation generator   │
│  - Sensitivity sweep                    │
│  - Volume forecast builder              │
└─────────────────────────────────────────┘
```

**Key design decisions:**

- **Dataclass I/O**: `PromotionInput` and `PromotionResult` are well-defined dataclasses. This makes the engine testable, serializable, and easy to swap behind an API.
- **Category profiles as configuration**: All category-specific parameters are in a single dict. In production, this becomes a database table that merchandising teams can tune.
- **Simulation-first, ML as a cross-check**: The core recommendation logic uses interpretable economic heuristics (elasticity, cannibalization, pantry loading) — explainability matters more than accuracy in a POC. A RandomForest model (`ml_model.py`) runs alongside the simulation as an independent validator, trained on 2,000 synthetic scenarios generated by the simulator itself. When the two agree, confidence is higher; a large divergence flags edge cases worth investigating.

### Production Architecture (at scale)

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  React/Next  │────▶│  FastAPI     │────▶│  ML Pipeline     │
│  Frontend    │◀────│  Backend     │◀────│  (Airflow/       │
│              │     │               │     │   Prefect)       │
└──────────────┘     └───────┬───────┘     └────────┬─────────┘
                             │                      │
                     ┌───────▼───────┐      ┌───────▼─────────┐
                     │  PostgreSQL   │      │  Feature Store  │
                     │  (configs,    │      │  (Feast/Redis)  │
                     │   audit log)  │      │  - elasticity   │
                     └───────────────┘      │  - cross-price  │
                                            │  - seasonality  │
                     ┌───────────────┐      └─────────────────┘
                     │  Claude API   │
                     │  (narrative   │
                     │   generation) │
                     └───────────────┘
```

**What changes at scale:**

| Component | POC | Production |
|-----------|-----|------------|
| Elasticity | Category-level heuristic ranges | SKU-level models trained on POS data (log-log regression or Bayesian hierarchical) |
| Cannibalization | Fixed rates per category | Cross-price elasticity matrix from transaction-level basket data |
| Seasonality | Binary flag (peak/not) | Time-series decomposition (Prophet or similar) per SKU |
| Risk scoring | Rule-based weighted sum | Ensemble model incorporating historical promo outcomes |
| Frontend | Streamlit | React with real-time WebSocket updates |
| Backend | In-process Python | FastAPI with async workers, Redis cache |
| Narrative | Template-based text | Claude API for contextual, conversational insights |
| Data | Hardcoded category profiles | ETL pipeline ingesting POS, loyalty, inventory data |

---

## 4. Challenges Faced

**Challenge 1: Making the simulation feel real without real data.**
The biggest risk in a simulated system is outputs that feel obviously fake. I addressed this by:
- Using category-specific parameter ranges grounded in published retail research
- Adding slight randomness to elasticity calculations (simulating real-world noise)
- Implementing diminishing returns on deep discounts (a well-documented effect)
- Including post-promotion dips and pantry loading — effects that naive models ignore

**Challenge 2: Avoiding infinite recursion in sensitivity analysis.**
The sensitivity sweep needs to compute profit at different discount levels. My initial approach called `analyze()` recursively for each discount point, which triggered infinite recursion (each `analyze()` call runs its own sensitivity sweep). Fixed by implementing an inline profit calculator for the sweep that uses the same formulas without the recursive components.

**Challenge 3: Balancing depth vs. usability.**
Category managers are not data scientists. The UI needed to surface the right information at the right level. I solved this with progressive disclosure: banner recommendation → key metrics → tabbed deep-dives → full data tables. A user can get the answer in 2 seconds or spend 5 minutes exploring the analysis.

**Challenge 4: Making it horizontal.**
The brief requires the tool to work across categories. Rather than building category-specific logic, I parameterized everything through `CATEGORY_PROFILES`. Adding a new category (e.g., "Pet Food") is a single dictionary entry — no code changes.

---

## 5. What I Would Add with More Time

**Tier 1 — High-impact, achievable in days:**
- **Claude API integration** for generating natural-language executive summaries (the template-based version works, but an LLM-generated summary would be more contextual and compelling)
- **Promo comparison mode** — "Compare 20% off vs 25% off vs BOGO" side by side
- **Export to PDF/PPT** — category managers need to share recommendations with leadership
- **Historical promo log** — track past recommendations and actual outcomes to build trust

**Tier 2 — Significant value, needs weeks:**
- **Data ingestion pipeline** — connect to POS data (Snowflake, BigQuery) to replace simulated parameters with learned ones
- **Multi-product optimization** — "Given a $10K promo budget this month, which 5 promotions maximize total incremental profit?"
- **A/B test designer** — suggest test vs. control store groups for validating the model's predictions
- **Slack/Teams integration** — push recommendations to where buyers already work

**Tier 3 — Differentiating, needs months:**
- **Causal ML models** — replace heuristic elasticity with causal inference (diff-in-diff, synthetic control) trained on historical promo data
- **Competitor price monitoring** — factor in competitor pricing from web scraping or data vendors
- **Demand forecasting layer** — integrate time-series forecasting for better baseline estimation
- **Multi-tenant SaaS** — role-based access, org-level configurations, SSO

---

## 6. Additional Notes

**Assumptions made:**
- Category-level elasticity ranges are based on published retail research and industry benchmarks. In production, these would be learned from the retailer's own data.
- Cannibalization rates assume a typical assortment breadth. A retailer with 3 SKUs in a category will see different dynamics than one with 30.
- The simulation treats each promotion independently. In reality, promotions interact (a beverage promo during Super Bowl week behaves differently than the same promo in February).
- Margin figures use retail margin (price - cost / price). Some retailers use markup; the system would need a toggle.
- The ML model (`ml_model.py`) is trained on 2,000 synthetic scenarios generated by the simulator itself (`data_utils.py`). It learns a compressed approximation of the simulation function, which is why it tends to agree with the simulation. Its value is as a fast cross-validation signal and a demonstration of the ML layer's architecture — not as an independently trained model. In production, this would be replaced with a model trained on actual historical POS and promo outcome data.

**Why Streamlit over a React app:**
For a 24-hour build, Streamlit provides 80% of the polish at 20% of the development time. The resulting app is genuinely demo-ready — interactive charts, responsive layout, clean aesthetics. A React frontend would be warranted for production, but for a POC demonstration, Streamlit is the right tool.

**Why simulation over real ML:**
The brief explicitly says "simulated/mocked intelligence is fine." More importantly, the *product decisions* matter more than the model quality at this stage. The simulation engine demonstrates that I understand the analytical framework (elasticity, cannibalization, pantry loading, post-promo effects) that would underpin a real ML system. Slapping a random forest on synthetic data would look more "ML" but teach less about the problem domain.

---

## Tech Stack Summary

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend | Streamlit | Python-native, rapid prototyping, built-in widgets |
| Visualization | Plotly | Interactive charts, waterfall support, clean defaults |
| Analytics | NumPy | Efficient numerical computation for simulation |
| Data handling | Pandas | DataFrame display in Streamlit tables |
| ML layer | scikit-learn (RandomForestRegressor) | Lightweight, no-dependency ML for the cross-validation layer |
| Deployment | Streamlit Cloud | Free, zero-config, GitHub integration |
