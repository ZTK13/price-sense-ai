"""
Price Sense AI — Should You Run This Promotion?
An AI-powered promotion analysis tool for mid-market retailers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# SIMULATION ENGINE (inlined for single-file deployment)
# =============================================================================

CATEGORY_PROFILES = {
    "Grocery": {
        "elasticity_range": (-2.5, -1.2), "base_margin": 0.28,
        "cannibalization_rate": 0.18, "cross_category_rate": 0.05,
        "pantry_loading_factor": 0.25, "seasonality_amplitude": 0.08,
        "typical_weekly_units": 500, "avg_price": 5.99,
    },
    "Specialty Food & Nuts": {
        "elasticity_range": (-2.0, -0.9), "base_margin": 0.40,
        "cannibalization_rate": 0.22, "cross_category_rate": 0.03,
        "pantry_loading_factor": 0.30, "seasonality_amplitude": 0.12,
        "typical_weekly_units": 200, "avg_price": 9.99,
    },
    "Beverages": {
        "elasticity_range": (-3.0, -1.5), "base_margin": 0.35,
        "cannibalization_rate": 0.20, "cross_category_rate": 0.08,
        "pantry_loading_factor": 0.35, "seasonality_amplitude": 0.15,
        "typical_weekly_units": 800, "avg_price": 3.49,
    },
    "Snacks & Confectionery": {
        "elasticity_range": (-2.8, -1.3), "base_margin": 0.42,
        "cannibalization_rate": 0.25, "cross_category_rate": 0.10,
        "pantry_loading_factor": 0.20, "seasonality_amplitude": 0.10,
        "typical_weekly_units": 600, "avg_price": 4.49,
    },
    "Dairy & Refrigerated": {
        "elasticity_range": (-2.2, -1.0), "base_margin": 0.25,
        "cannibalization_rate": 0.15, "cross_category_rate": 0.04,
        "pantry_loading_factor": 0.10, "seasonality_amplitude": 0.06,
        "typical_weekly_units": 700, "avg_price": 4.99,
    },
    "Health & Wellness": {
        "elasticity_range": (-1.8, -0.6), "base_margin": 0.50,
        "cannibalization_rate": 0.12, "cross_category_rate": 0.03,
        "pantry_loading_factor": 0.15, "seasonality_amplitude": 0.05,
        "typical_weekly_units": 150, "avg_price": 14.99,
    },
    "Household & Cleaning": {
        "elasticity_range": (-2.0, -1.0), "base_margin": 0.30,
        "cannibalization_rate": 0.10, "cross_category_rate": 0.02,
        "pantry_loading_factor": 0.40, "seasonality_amplitude": 0.04,
        "typical_weekly_units": 300, "avg_price": 7.99,
    },
}

PROMO_TYPES = {
    "Percentage Off": {"execution_cost": 0.02, "awareness_multiplier": 1.0},
    "BOGO (Buy One Get One)": {"execution_cost": 0.05, "awareness_multiplier": 1.3},
    "Bundle Deal": {"execution_cost": 0.04, "awareness_multiplier": 1.1},
    "Loyalty Member Exclusive": {"execution_cost": 0.01, "awareness_multiplier": 0.7},
    "Flash Sale (24-48hr)": {"execution_cost": 0.03, "awareness_multiplier": 1.4},
}


@dataclass
class PromotionInput:
    product_name: str
    category: str
    regular_price: float
    discount_pct: float
    promo_type: str
    duration_days: int
    num_competing_products: int
    is_peak_season: bool
    has_display_support: bool
    weekly_baseline_units: Optional[int] = None
    product_margin_pct: Optional[float] = None


@dataclass
class PromotionResult:
    recommendation: str
    confidence_score: float
    recommendation_reason: str
    baseline_units_per_week: int
    projected_units_per_week: int
    volume_lift_pct: float
    incremental_units: int
    total_promo_units: int
    cannibalized_units: int
    cannibalization_pct: float
    true_incremental_units: int
    regular_revenue: float
    promo_revenue: float
    revenue_change: float
    regular_profit: float
    promo_profit: float
    profit_change: float
    promo_cost: float
    roi: float
    post_promo_dip_pct: float
    pantry_loading_pct: float
    net_30day_profit_impact: float
    risk_score: float
    risk_factors: list
    mitigations: list
    weekly_volume_forecast: list
    profit_waterfall: dict
    sensitivity_data: dict


class PromotionSimulator:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def analyze(self, inp):
        profile = CATEGORY_PROFILES[inp.category]
        baseline_units = inp.weekly_baseline_units or profile["typical_weekly_units"]
        margin_pct = inp.product_margin_pct or profile["base_margin"]
        elas_lo, elas_hi = profile["elasticity_range"]
        competition_factor = min(1.0, 0.5 + inp.num_competing_products * 0.1)
        elasticity = elas_lo + (elas_hi - elas_lo) * (1 - competition_factor)
        elasticity += self.rng.normal(0, 0.1)
        discount_frac = inp.discount_pct / 100.0
        raw_lift = -elasticity * discount_frac
        if inp.is_peak_season:
            raw_lift *= 1.15 + profile["seasonality_amplitude"]
        if inp.has_display_support:
            raw_lift *= 1.25
        promo_info = PROMO_TYPES[inp.promo_type]
        raw_lift *= promo_info["awareness_multiplier"]
        if discount_frac > 0.30:
            raw_lift *= (1 - 0.4 * (discount_frac - 0.30))
        volume_lift_pct = max(0.0, raw_lift) * 100
        projected_units = int(baseline_units * (1 + max(0, raw_lift)))
        incremental_units = projected_units - baseline_units
        promo_weeks = max(1, inp.duration_days / 7)
        total_promo_units = int(projected_units * promo_weeks)
        total_baseline_units = int(baseline_units * promo_weeks)
        cannibal_rate = profile["cannibalization_rate"]
        cannibal_rate *= min(1.5, 1.0 + inp.num_competing_products * 0.05)
        if inp.promo_type == "Bundle Deal":
            cannibal_rate *= 0.7
        cannibalized_units = int(incremental_units * promo_weeks * cannibal_rate)
        true_incremental_units = int(incremental_units * promo_weeks - cannibalized_units)
        promo_price = inp.regular_price * (1 - discount_frac)
        cost_per_unit = inp.regular_price * (1 - margin_pct)
        regular_revenue = total_baseline_units * inp.regular_price
        promo_revenue = total_promo_units * promo_price
        regular_profit = total_baseline_units * (inp.regular_price - cost_per_unit)
        promo_margin_per_unit = promo_price - cost_per_unit
        promo_profit_product = total_promo_units * promo_margin_per_unit
        execution_cost = promo_info["execution_cost"] * promo_revenue
        promo_profit = promo_profit_product - execution_cost
        profit_change = promo_profit - regular_profit
        promo_cost = regular_profit - promo_profit
        roi = (profit_change / max(abs(promo_cost), 1)) * 100 if promo_cost != 0 else 0
        pantry_loading = profile["pantry_loading_factor"] * discount_frac
        post_promo_dip = pantry_loading * 0.6 + discount_frac * 0.15
        post_promo_lost_units = int(baseline_units * post_promo_dip * 2)
        post_promo_lost_profit = post_promo_lost_units * (inp.regular_price - cost_per_unit)
        net_30day = profit_change - post_promo_lost_profit
        risk_factors = []
        risk_score = 0
        if discount_frac > 0.35:
            risk_factors.append(("Deep discount erodes margins", 25)); risk_score += 25
        elif discount_frac > 0.25:
            risk_factors.append(("Moderate-to-high discount level", 12)); risk_score += 12
        if cannibal_rate > 0.25:
            risk_factors.append(("High cannibalization from similar products", 20)); risk_score += 20
        elif cannibal_rate > 0.15:
            risk_factors.append(("Moderate cannibalization expected", 10)); risk_score += 10
        if pantry_loading > 0.15:
            risk_factors.append(("Pantry loading will suppress post-promo sales", 15)); risk_score += 15
        if inp.duration_days > 14:
            risk_factors.append(("Long promo trains customers to wait for deals", 15)); risk_score += 15
        if margin_pct < 0.25:
            risk_factors.append(("Low base margin leaves little room for discounting", 20)); risk_score += 20
        if not inp.has_display_support:
            risk_factors.append(("No display support limits awareness", 8)); risk_score += 8
        risk_score = min(100, risk_score)
        mitigations = self._generate_mitigations(inp, risk_factors, profile)
        rec, confidence, reason = self._decide(profit_change, net_30day, risk_score, volume_lift_pct, cannibalized_units, true_incremental_units, roi, inp)
        weekly_forecast = self._build_forecast(baseline_units, projected_units, promo_weeks, post_promo_dip)
        profit_waterfall = {
            "Baseline Profit": round(regular_profit, 2),
            "Volume Lift": round((incremental_units * promo_weeks) * promo_margin_per_unit, 2),
            "Discount Cost": round(-(total_promo_units * inp.regular_price * discount_frac), 2),
            "Cannibalization": round(-(cannibalized_units * (inp.regular_price - cost_per_unit)), 2),
            "Execution Cost": round(-execution_cost, 2),
            "Post-Promo Dip": round(-post_promo_lost_profit, 2),
        }
        sensitivity = self._sensitivity_sweep(inp, baseline_units, margin_pct, profile)
        return PromotionResult(
            recommendation=rec, confidence_score=round(confidence, 1), recommendation_reason=reason,
            baseline_units_per_week=baseline_units, projected_units_per_week=projected_units,
            volume_lift_pct=round(volume_lift_pct, 1), incremental_units=int(incremental_units * promo_weeks),
            total_promo_units=total_promo_units, cannibalized_units=cannibalized_units,
            cannibalization_pct=round(cannibal_rate * 100, 1), true_incremental_units=true_incremental_units,
            regular_revenue=round(regular_revenue, 2), promo_revenue=round(promo_revenue, 2),
            revenue_change=round(promo_revenue - regular_revenue, 2), regular_profit=round(regular_profit, 2),
            promo_profit=round(promo_profit, 2), profit_change=round(profit_change, 2),
            promo_cost=round(max(0, promo_cost), 2), roi=round(roi, 1),
            post_promo_dip_pct=round(post_promo_dip * 100, 1), pantry_loading_pct=round(pantry_loading * 100, 1),
            net_30day_profit_impact=round(net_30day, 2), risk_score=round(risk_score, 1),
            risk_factors=risk_factors, mitigations=mitigations,
            weekly_volume_forecast=weekly_forecast, profit_waterfall=profit_waterfall, sensitivity_data=sensitivity,
        )

    def _decide(self, profit_change, net_30day, risk_score, lift_pct, cannibalized, true_inc, roi, inp):
        score = 0; reasons = []
        if net_30day > 0:
            score += 35; reasons.append(f"Net positive 30-day profit impact (+${net_30day:,.0f})")
        elif profit_change > 0:
            score += 15; reasons.append("Promo period is profitable but post-promo dip erodes gains")
        else:
            score -= 20; reasons.append(f"Negative profit impact (${net_30day:,.0f})")
        if lift_pct > 40:
            score += 20; reasons.append(f"Strong volume lift of {lift_pct:.0f}%")
        elif lift_pct > 15:
            score += 10; reasons.append(f"Moderate volume lift of {lift_pct:.0f}%")
        if true_inc > 0: score += 15
        else: score -= 15; reasons.append("Cannibalization exceeds incremental volume")
        if risk_score > 60: score -= 25; reasons.append("Multiple high-risk factors present")
        elif risk_score > 35: score -= 10
        if roi > 20: score += 15
        elif roi < -20: score -= 15
        if score >= 30: rec = "✅ Run This Promotion"; confidence = min(95, 60 + score)
        elif score >= 0: rec = "⚠️ Proceed with Caution"; confidence = 40 + score
        else: rec = "🚫 Don't Run This Promotion"; confidence = min(90, 60 + abs(score))
        return rec, confidence, reasons[0] if reasons else "Balanced risk-reward profile"

    def _generate_mitigations(self, inp, risk_factors, profile):
        m = []; rn = [r[0] for r in risk_factors]
        if any("Deep discount" in r for r in rn): m.append(f"Test at {max(5, inp.discount_pct - 10):.0f}% off first")
        if any("cannibalization" in r.lower() for r in rn): m.append("Limit promo to a single SKU size to reduce portfolio cannibalization")
        if any("Pantry loading" in r for r in rn): m.append("Cap purchase quantity (e.g., limit 2 per customer) to reduce stockpiling")
        if any("Long promo" in r for r in rn): m.append("Shorten to 7 days — promotions beyond 2 weeks train deal-seeking behavior")
        if any("display" in r.lower() for r in rn): m.append("Add in-store display or digital feature to maximize awareness and lift")
        if any("Low base margin" in r for r in rn): m.append("Consider a bundle or BOGO instead")
        if not m: m.append("Monitor daily sell-through and pull the promo early if lift underperforms by >30%")
        return m

    def _build_forecast(self, baseline, projected, promo_weeks, post_dip):
        f = [{"week": "Pre-Promo", "units": baseline, "phase": "Baseline"}]
        pw = max(1, int(round(promo_weeks)))
        for i in range(pw):
            label = f"Promo Wk {i+1}" if pw > 1 else "Promo Week"
            factor = 0.90 if i == 0 and pw > 1 else 1.0
            f.append({"week": label, "units": int(projected * factor), "phase": "Promotion"})
        for i in range(3):
            dip = max(0, post_dip * (1 - i * 0.35))
            f.append({"week": f"Post Wk {i+1}", "units": int(baseline * (1 - dip)), "phase": "Post-Promo"})
        return f

    def _sensitivity_sweep(self, inp, baseline, margin_pct, profile):
        discounts = list(range(5, 55, 5)); results = {}; p = profile
        for d in discounts:
            elas_lo, elas_hi = p["elasticity_range"]
            comp_f = min(1.0, 0.5 + inp.num_competing_products * 0.1)
            elas = elas_lo + (elas_hi - elas_lo) * (1 - comp_f)
            df = d / 100.0; lift = -elas * df
            if inp.is_peak_season: lift *= 1.15 + p["seasonality_amplitude"]
            if inp.has_display_support: lift *= 1.25
            lift *= PROMO_TYPES[inp.promo_type]["awareness_multiplier"]
            if df > 0.30: lift *= (1 - 0.4 * (df - 0.30))
            lift = max(0, lift); proj = int(baseline * (1 + lift))
            pw = max(1, inp.duration_days / 7); total_u = int(proj * pw); total_base = int(baseline * pw)
            pp = inp.regular_price * (1 - df); cpu = inp.regular_price * (1 - margin_pct)
            reg_prof = total_base * (inp.regular_price - cpu)
            exec_cost = PROMO_TYPES[inp.promo_type]["execution_cost"] * total_u * pp
            promo_prof = total_u * (pp - cpu) - exec_cost
            pl = p["pantry_loading_factor"] * df; ppd = pl * 0.6 + df * 0.15
            post_lost = int(baseline * ppd * 2) * (inp.regular_price - cpu)
            results[d] = round((promo_prof - reg_prof) - post_lost, 2)
        return {"discounts": discounts, "net_30day_profit": [results[d] for d in discounts]}


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Price Sense AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* =========================
       MAIN HEADER
    ========================= */
    .main-header {
        background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(51,65,85,0.7))
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148,163,184,0.2);
    }
    .main-header h1 { color: #f8fafc;}
    .main-header p { color: rgba(226,232,240,0.9); }

    .main-header .badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    /* =========================
       RECOMMENDATION CARDS
    ========================= */
    .rec-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
    }

    .rec-run { background: rgba(34,197,94,0.08); border-left-color: #22c55e; }
    .rec-caution { background: rgba(245,158,11,0.08); border-left-color: #f59e0b; }
    .rec-stop { background: rgba(239,68,68,0.08); border-left-color: #ef4444; }

    .rec-card h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }

    .rec-card p {
        color: rgba(51, 65, 85, 0.9);
    }

    /* =========================
       SECTION HEADERS
    ========================= */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;   /* FIX: visible in dark */
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.4);
    }

    /* Light mode override */
    @media (prefers-color-scheme: light) {
        .section-header {
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
        }
    }

    /* =========================
       RISK ITEMS
    ========================= */
    .risk-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.2);
    }

    .risk-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 10px;
        flex-shrink: 0;
    }

    .risk-high { background: #ef4444; }
    .risk-med { background: #f59e0b; }
    .risk-low { background: #22c55e; }

    /* =========================
       CAPTION FIX (DARK MODE)
    ========================= */
    .stCaption {
        color: rgba(203, 213, 225, 0.9);
    }

    /* =========================
       TAB LABEL VISIBILITY
    ========================= */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* =========================
       CLEANUP
    ========================= */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 Price Sense AI <span class="badge">BETA</span></h1>
    <p>Should you run this promotion? Get AI-powered analysis of projected lift, cannibalization, profit impact, and risk — in seconds.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🧾 Inputs")
    st.markdown("## 🎯 Promotion Setup")
    st.markdown("Configure your proposed promotion below.")
    st.markdown("___")
    st.markdown("#### Product Details")
    category = st.selectbox("Product Category", options=list(CATEGORY_PROFILES.keys()), index=1, help="Select the category that best fits your product")
    product_name = st.text_input("Product Name", value="Salted Pistachios 16oz", help="e.g., 'Organic Kombucha 12pk', 'Cheddar Cheese 8oz'")
    regular_price = st.number_input("Regular Price ($)", min_value=0.50, max_value=500.0, value=12.99, step=0.50, help="Current shelf price")
    custom_margin = st.checkbox("Override category margin?", value=False)
    margin_pct = None
    if custom_margin:
        margin_pct = st.slider("Product Margin %", 5, 70, int(CATEGORY_PROFILES[category]["base_margin"] * 100), help="Your actual margin on this product") / 100.0
    st.markdown("___")
    st.markdown("#### Promotion Details")
    promo_type = st.selectbox("Promotion Type", options=list(PROMO_TYPES.keys()), help="Type of promotional mechanic")
    discount_pct = st.slider("Discount %", 5, 50, 25, step=5, help="Depth of discount off regular price")
    duration_days = st.slider("Duration (days)", 1, 28, 7, help="How long will the promotion run?")
    st.markdown("___")
    st.markdown("#### Market Context")
    num_competing = st.slider("Competing Products in Category", 0, 15, 4, help="How many similar products compete for the same shopper?")
    is_peak = st.checkbox("Peak Season?", value=False, help="Is this a high-demand period (holidays, summer, etc.)?")
    has_display = st.checkbox("In-Store Display / Feature Support?", value=True, help="End-cap, feature ad, or digital promotion?")
    custom_baseline = st.checkbox("Override baseline volume?", value=False)
    weekly_units = None
    if custom_baseline:
        weekly_units = st.number_input("Weekly Baseline Units", 10, 10000, CATEGORY_PROFILES[category]["typical_weekly_units"])
    st.markdown("___")
    analyze_btn = st.button("🚀 Analyze Promotion", use_container_width=True, type="primary")


def _generate_summary(result, inp):
    promo_price = inp.regular_price * (1 - inp.discount_pct / 100)
    if "Don't" in result.recommendation:
        opener = (
            f"**This promotion is not recommended.** At {inp.discount_pct}% off, "
            f"**{inp.product_name}** is projected to lose money on a net basis."
        )
    elif "Run" in result.recommendation:
        opener = (
            f"**This promotion is recommended.** Running {inp.discount_pct}% off on "
            f"**{inp.product_name}** (${inp.regular_price:.2f} → ${promo_price:.2f}) "
            f"for {inp.duration_days} days is projected to be profitable."
        )
    elif "Caution" in result.recommendation:
        opener = (
            f"**This promotion warrants caution.** While {inp.discount_pct}% off "
            f"on **{inp.product_name}** will drive traffic, several factors limit its effectiveness."
        )
    else:
        opener = (
            f"**This promotion is not recommended.** At {inp.discount_pct}% off, "
            f"**{inp.product_name}** is projected to lose money on a net basis."
        )
    vol = (
        f"We project a **{result.volume_lift_pct:.0f}% volume lift** "
        f"({result.baseline_units_per_week:,} → "
        f"{result.projected_units_per_week:,} units/week). "
        f"However, approximately **{result.cannibalization_pct:.0f}%** of incremental volume comes from cannibalization of adjacent products, "
        f"leaving **{result.true_incremental_units:,} true incremental units**."
    )
    if result.profit_change >= 0:
        prof = f"During the promotion window, profit increases by **${result.profit_change:,.0f}**. "
    else:
        prof = (
            f"The promotion reduces profit by **${abs(result.profit_change):,.0f}** "
            f"during the promo period. "
        )
    if result.net_30day_profit_impact >= 0:
        prof += (
            f"After factoring in the post-promotion sales dip ({result.post_promo_dip_pct:.0f}%), " 
            f"the net 30-day impact remains positive at **${result.net_30day_profit_impact:,.0f}**."
        )
    else:
        prof += (
            f"The post-promotion dip ({result.post_promo_dip_pct:.0f}%) further erodes returns, "
            f"bringing the 30-day net impact to **-${abs(result.net_30day_profit_impact):,.0f}**."
        )
    sens = result.sensitivity_data
    best_idx = max(range(len(sens["net_30day_profit"])), key=lambda i: sens["net_30day_profit"][i])
    best_d = sens["discounts"][best_idx]
    if best_d != int(inp.discount_pct):
        sn = f"💡 **Optimization insight:** Our sensitivity analysis suggests **{best_d}% off** would be the profit-maximizing discount for this product."
    else:
        sn = f"Your proposed {inp.discount_pct}% discount is well-calibrated — it's at or near the profit-optimal point for this product."
    return f"""
        {opener}
    
        <div class="section-header">📈 Volume Impact</div>
        {vol}
    
        <div class="section-header">💰 Profit Impact</div>
        {prof}
    
        <div class="section-header">💡 Optimization Insight</div>
        {sn}
        """


if analyze_btn or "result" in st.session_state:
    if analyze_btn:
        inp = PromotionInput(product_name=product_name, category=category, regular_price=regular_price, discount_pct=discount_pct, promo_type=promo_type, duration_days=duration_days, num_competing_products=num_competing, is_peak_season=is_peak, has_display_support=has_display, weekly_baseline_units=weekly_units, product_margin_pct=margin_pct)
        sim = PromotionSimulator()
        result = sim.analyze(inp)
        st.session_state["result"] = result
        st.session_state["inp"] = inp
    else:
        result = st.session_state["result"]
        inp = st.session_state["inp"]

    css_class = "rec-stop" if "Don't" in result.recommendation else "rec-run" if "Run" in result.recommendation else "rec-caution"
    st.markdown(f'<div class="rec-card {css_class}"><h2>{result.recommendation}</h2><p><strong>Confidence:</strong> {result.confidence_score}% · {result.recommendation_reason}</p></div>', unsafe_allow_html=True)

    promo_price = inp.regular_price * (1 - inp.discount_pct / 100)
    st.markdown("## 📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Volume Lift", f"+{result.volume_lift_pct}%", f"{result.true_incremental_units:,} true incremental units")
    with c2: st.metric("Promo Profit Impact", f"{'+'if result.profit_change>0 else ''}${result.profit_change:,.0f}", f"ROI: {result.roi:.0f}%")
    with c3: st.metric("Net 30-Day Impact", f"${result.net_30day_profit_impact:,.0f}", f"Post-promo dip: {result.post_promo_dip_pct:.0f}%", delta_color="normal")
    with c4:
        rl = "Low" if result.risk_score < 30 else "Medium" if result.risk_score < 60 else "High"
        st.metric("Risk Score", f"{result.risk_score:.0f}/100", rl, delta_color="inverse")

    st.markdown("___")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Volume Forecast", "💰 Profit Waterfall", "🔍 Sensitivity Analysis", "📊 Full Breakdown"])

    with tab1:
        st.markdown('<div class="section-header">6-Week Volume Forecast</div>', unsafe_allow_html=True)
        df_f = pd.DataFrame(result.weekly_volume_forecast)
        colors = {"Baseline": "#94a3b8", "Promotion": "#3b82f6", "Post-Promo": "#f59e0b"}
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_f["week"], y=df_f["units"], marker_color=[colors[p] for p in df_f["phase"]], text=df_f["units"], textposition="outside", textfont=dict(size=12)))
        fig.add_hline(y=result.baseline_units_per_week, line_dash="dash", line_color="#94a3b8", annotation_text="Baseline", annotation_position="top right")
        fig.update_layout(height=400,
                          plot_bgcolor="white",
                          xaxis=dict(title="", tickfont=dict(size=12)),
                          yaxis=dict(title="Units / Week", gridcolor="#f1f5f9"),
                          margin=dict(l=60, r=20, t=30, b=40),
                          font=dict(color=None),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📌 Baseline: {result.baseline_units_per_week:,} units/wk · During promo: {result.projected_units_per_week:,} units/wk · Cannibalized: {result.cannibalized_units:,} units ({result.cannibalization_pct:.0f}%)")

    with tab2:
        st.markdown('<div class="section-header">Profit Impact Waterfall</div>', unsafe_allow_html=True)
        wf = result.profit_waterfall
        labels = list(wf.keys()) + ["Net Impact"]; values = list(wf.values()); net = sum(values)
        measures = ["absolute"] + ["relative"] * (len(values) - 1) + ["total"]; values.append(net)
        fig2 = go.Figure(go.Waterfall(x=labels, y=values, measure=measures, connector={"line": {"color": "#e2e8f0"}}, increasing={"marker": {"color": "#22c55e"}}, decreasing={"marker": {"color": "#ef4444"}}, totals={"marker": {"color": "#3b82f6"}}, textposition="outside", text=[f"${v:,.0f}" for v in values], textfont=dict(size=11)))
        fig2.update_layout(height=420, plot_bgcolor="white", yaxis=dict(title="Profit ($)", gridcolor="#f1f5f9"), xaxis=dict(tickfont=dict(size=10)), margin=dict(l=60, r=20, t=30, b=60), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Discount Sensitivity — Net 30-Day Profit</div>', unsafe_allow_html=True)
        sens = result.sensitivity_data
        df_s = pd.DataFrame({"Discount %": [f"{d}%" for d in sens["discounts"]], "Net 30-Day Profit": sens["net_30day_profit"], "discount_num": sens["discounts"]})
        cd = int(inp.discount_pct)
        cs = ["#3b82f6" if d == cd else "#cbd5e1" for d in sens["discounts"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df_s["Discount %"], y=df_s["Net 30-Day Profit"], marker_color=cs, text=[f"${v:,.0f}" for v in df_s["Net 30-Day Profit"]], textposition="outside", textfont=dict(size=10)))
        fig3.add_hline(y=0, line_color="#94a3b8", line_width=1)
        fig3.update_layout(height=400, plot_bgcolor="white", yaxis=dict(title="Net 30-Day Profit ($)", gridcolor="#f1f5f9"), margin=dict(l=60, r=20, t=30, b=40), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        bi = max(range(len(sens["net_30day_profit"])), key=lambda i: sens["net_30day_profit"][i])
        bd = sens["discounts"][bi]; bp = sens["net_30day_profit"][bi]
        if bd != cd:
            st.info(f"💡 **Optimal discount: {bd}%** → Net 30-day profit of ${bp:,.0f}. Your proposed {cd}% discount yields ${sens['net_30day_profit'][sens['discounts'].index(cd)]:,.0f}.")
        else:
            st.success(f"✅ Your proposed {cd}% discount is at or near the profit-optimal point.")

    with tab4:
        st.markdown('<div class="section-header">Complete Analysis</div>', unsafe_allow_html=True)
        cl, cr = st.columns(2)
        with cl:
            st.markdown("##### 📦 Volume Analysis")
            st.dataframe(pd.DataFrame({"Metric": ["Baseline Weekly Units", "Projected Weekly Units", "Volume Lift %", "Total Promo Units", "Gross Incremental Units", "Cannibalized Units", "True Incremental Units", "Cannibalization Rate"], "Value": [f"{result.baseline_units_per_week:,}", f"{result.projected_units_per_week:,}", f"+{result.volume_lift_pct}%", f"{result.total_promo_units:,}", f"{result.incremental_units:,}", f"-{result.cannibalized_units:,}", f"{result.true_incremental_units:,}", f"{result.cannibalization_pct:.1f}%"]}), hide_index=True, use_container_width=True)
        with cr:
            st.markdown("##### 💵 Financial Analysis")
            st.dataframe(pd.DataFrame({"Metric": ["Regular Price", "Promo Price", "Baseline Revenue", "Promo Revenue", "Revenue Change", "Baseline Profit", "Promo Profit", "Profit Change", "Net 30-Day Impact", "Pantry Loading Effect", "Post-Promo Dip"], "Value": [f"${inp.regular_price:.2f}", f"${promo_price:.2f}", f"${result.regular_revenue:,.2f}", f"${result.promo_revenue:,.2f}", f"${result.revenue_change:,.2f}", f"${result.regular_profit:,.2f}", f"${result.promo_profit:,.2f}", f"${result.profit_change:,.2f}", f"${result.net_30day_profit_impact:,.2f}", f"{result.pantry_loading_pct:.1f}%", f"{result.post_promo_dip_pct:.1f}%"]}), hide_index=True, use_container_width=True)

    st.markdown("___")
    cr2, cm = st.columns(2)
    with cr2:
        st.markdown('<div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
        if result.risk_factors:
            for factor, severity in result.risk_factors:
                dc = "risk-high" if severity >= 20 else "risk-med" if severity >= 10 else "risk-low"
                st.markdown(f'<div class="risk-item"><div class="risk-dot {dc}"></div><span>{factor} <em style="color:rgba(148,163,184,0.9)">(+{severity} risk pts)</em></span></div>', unsafe_allow_html=True)
        else:
            st.success("No significant risk factors identified.")
    with cm:
        st.markdown('<div class="section-header">🛡️ Recommended Mitigations</div>', unsafe_allow_html=True)
        for i, m in enumerate(result.mitigations, 1):
            st.markdown(f"**{i}.** {m}")

    st.markdown("___")
    st.markdown('<div class="section-header">🤖 AI Executive Summary</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_generate_summary(result, inp), unsafe_allow_html=True)

else:
    st.info("👈 Configure a promotion in the sidebar and click **Analyze Promotion** to get started.")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("#### 📈 Volume Lift\nSee projected sales increase based on price elasticity and category dynamics.")
    with c2: st.markdown("#### 💰 Profit Impact\nUnderstand true profitability after accounting for margin erosion and execution costs.")
    with c3: st.markdown("#### ⚠️ Risk Analysis\nUncover hidden costs — cannibalization, pantry loading, and post-promo dips.")
    st.markdown("___")
    st.markdown("**Price Sense AI** helps mid-market retailers ($50M–$500M revenue) make data-driven promotion decisions. Stop relying on gut feel — know the true ROI before you run.\n\n*Try it now with the reference scenario: 25% off Salted Pistachios 16oz →*")
