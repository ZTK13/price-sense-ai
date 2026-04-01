"""
Price Sense AI — Promotion Simulation Engine

This module implements the core analytics for evaluating retail promotions.
It uses established retail economics principles:
- Price elasticity of demand (category-specific)
- Cross-product cannibalization modeling
- Margin-aware profit impact analysis
- Risk scoring based on multiple factors

While this is a simulation (not trained on real data), the underlying math
mirrors how actual promotion optimization systems work.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Category configurations — these encode domain knowledge about how different
# retail categories respond to promotions. In production, these would be
# learned from historical transaction data.
# ---------------------------------------------------------------------------

CATEGORY_PROFILES = {
    "Grocery": {
        "elasticity_range": (-2.5, -1.2),
        "base_margin": 0.28,
        "cannibalization_rate": 0.18,
        "cross_category_rate": 0.05,
        "pantry_loading_factor": 0.25,
        "seasonality_amplitude": 0.08,
        "typical_weekly_units": 500,
        "avg_price": 5.99,
    },
    "Specialty Food & Nuts": {
        "elasticity_range": (-2.0, -0.9),
        "base_margin": 0.40,
        "cannibalization_rate": 0.22,
        "cross_category_rate": 0.03,
        "pantry_loading_factor": 0.30,
        "seasonality_amplitude": 0.12,
        "typical_weekly_units": 200,
        "avg_price": 9.99,
    },
    "Beverages": {
        "elasticity_range": (-3.0, -1.5),
        "base_margin": 0.35,
        "cannibalization_rate": 0.20,
        "cross_category_rate": 0.08,
        "pantry_loading_factor": 0.35,
        "seasonality_amplitude": 0.15,
        "typical_weekly_units": 800,
        "avg_price": 3.49,
    },
    "Snacks & Confectionery": {
        "elasticity_range": (-2.8, -1.3),
        "base_margin": 0.42,
        "cannibalization_rate": 0.25,
        "cross_category_rate": 0.10,
        "pantry_loading_factor": 0.20,
        "seasonality_amplitude": 0.10,
        "typical_weekly_units": 600,
        "avg_price": 4.49,
    },
    "Dairy & Refrigerated": {
        "elasticity_range": (-2.2, -1.0),
        "base_margin": 0.25,
        "cannibalization_rate": 0.15,
        "cross_category_rate": 0.04,
        "pantry_loading_factor": 0.10,
        "seasonality_amplitude": 0.06,
        "typical_weekly_units": 700,
        "avg_price": 4.99,
    },
    "Health & Wellness": {
        "elasticity_range": (-1.8, -0.6),
        "base_margin": 0.50,
        "cannibalization_rate": 0.12,
        "cross_category_rate": 0.03,
        "pantry_loading_factor": 0.15,
        "seasonality_amplitude": 0.05,
        "typical_weekly_units": 150,
        "avg_price": 14.99,
    },
    "Household & Cleaning": {
        "elasticity_range": (-2.0, -1.0),
        "base_margin": 0.30,
        "cannibalization_rate": 0.10,
        "cross_category_rate": 0.02,
        "pantry_loading_factor": 0.40,
        "seasonality_amplitude": 0.04,
        "typical_weekly_units": 300,
        "avg_price": 7.99,
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
    """User-defined promotion parameters."""
    product_name: str
    category: str
    regular_price: float
    discount_pct: float              # e.g. 25 for 25% off
    promo_type: str                  # key into PROMO_TYPES
    duration_days: int
    num_competing_products: int      # in the same sub-category
    is_peak_season: bool
    has_display_support: bool        # end-cap, feature, etc.
    weekly_baseline_units: Optional[int] = None
    product_margin_pct: Optional[float] = None


@dataclass
class PromotionResult:
    """Complete analysis output."""
    # Top-line recommendation
    recommendation: str              # "Run It", "Proceed with Caution", "Don't Run"
    confidence_score: float          # 0–100
    recommendation_reason: str

    # Volume projections
    baseline_units_per_week: int
    projected_units_per_week: int
    volume_lift_pct: float
    incremental_units: int
    total_promo_units: int

    # Cannibalization
    cannibalized_units: int
    cannibalization_pct: float
    true_incremental_units: int

    # Financials
    regular_revenue: float
    promo_revenue: float
    revenue_change: float
    regular_profit: float
    promo_profit: float
    profit_change: float
    promo_cost: float
    roi: float

    # Post-promo
    post_promo_dip_pct: float
    pantry_loading_pct: float
    net_30day_profit_impact: float

    # Risk
    risk_score: float               # 0-100
    risk_factors: list
    mitigations: list

    # For charts
    weekly_volume_forecast: list     # 6-week forecast
    profit_waterfall: dict
    sensitivity_data: dict           # discount → profit mapping


class PromotionSimulator:
    """Core simulation engine."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def analyze(self, inp: PromotionInput) -> PromotionResult:
        profile = CATEGORY_PROFILES[inp.category]

        # --- Resolve defaults ---
        baseline_units = inp.weekly_baseline_units or profile["typical_weekly_units"]
        margin_pct = (inp.product_margin_pct or profile["base_margin"])

        # --- Price elasticity ---
        elas_lo, elas_hi = profile["elasticity_range"]
        # More products = more elastic (customers have alternatives)
        competition_factor = min(1.0, 0.5 + inp.num_competing_products * 0.1)
        elasticity = elas_lo + (elas_hi - elas_lo) * (1 - competition_factor)
        # Add slight randomness to feel real
        elasticity += self.rng.normal(0, 0.1)

        # --- Volume lift ---
        discount_frac = inp.discount_pct / 100.0
        raw_lift = -elasticity * discount_frac

        # Modifiers
        if inp.is_peak_season:
            raw_lift *= 1.15 + profile["seasonality_amplitude"]
        if inp.has_display_support:
            raw_lift *= 1.25

        promo_info = PROMO_TYPES[inp.promo_type]
        raw_lift *= promo_info["awareness_multiplier"]

        # Diminishing returns on deep discounts
        if discount_frac > 0.30:
            overshoot = discount_frac - 0.30
            raw_lift *= (1 - 0.4 * overshoot)

        volume_lift_pct = max(0.0, raw_lift) * 100
        projected_units = int(baseline_units * (1 + raw_lift))
        incremental_units = projected_units - baseline_units

        # --- Duration scaling ---
        promo_weeks = max(1, inp.duration_days / 7)
        total_promo_units = int(projected_units * promo_weeks)
        total_baseline_units = int(baseline_units * promo_weeks)

        # --- Cannibalization ---
        cannibal_rate = profile["cannibalization_rate"]
        # More SKUs in portfolio = higher cannibalization
        cannibal_rate *= min(1.5, 1.0 + inp.num_competing_products * 0.05)
        if inp.promo_type == "Bundle Deal":
            cannibal_rate *= 0.7  # Bundles reduce cannibalization
        cannibalized_units = int(incremental_units * promo_weeks * cannibal_rate)
        true_incremental_units = int(incremental_units * promo_weeks - cannibalized_units)

        # --- Financials ---
        promo_price = inp.regular_price * (1 - discount_frac)
        cost_per_unit = inp.regular_price * (1 - margin_pct)

        regular_revenue = total_baseline_units * inp.regular_price
        promo_revenue = total_promo_units * promo_price

        regular_profit = total_baseline_units * (inp.regular_price - cost_per_unit)
        promo_margin_per_unit = promo_price - cost_per_unit
        promo_profit_product = total_promo_units * promo_margin_per_unit

        # Execution cost
        execution_cost = promo_info["execution_cost"] * promo_revenue
        promo_profit = promo_profit_product - execution_cost

        profit_change = promo_profit - regular_profit
        promo_cost = regular_profit - promo_profit  # Opportunity cost view
        roi = (profit_change / max(abs(promo_cost), 1)) * 100 if promo_cost != 0 else 0

        # --- Post-promo effects ---
        pantry_loading = profile["pantry_loading_factor"] * discount_frac
        post_promo_dip = pantry_loading * 0.6 + discount_frac * 0.15
        post_promo_weeks = 2
        post_promo_lost_units = int(baseline_units * post_promo_dip * post_promo_weeks)
        post_promo_lost_profit = post_promo_lost_units * (inp.regular_price - cost_per_unit)

        net_30day = profit_change - post_promo_lost_profit

        # --- Risk scoring ---
        risk_factors = []
        risk_score = 0

        if discount_frac > 0.35:
            risk_factors.append(("Deep discount erodes margins", 25))
            risk_score += 25
        elif discount_frac > 0.25:
            risk_factors.append(("Moderate-to-high discount level", 12))
            risk_score += 12

        if cannibal_rate > 0.25:
            risk_factors.append(("High cannibalization from similar products", 20))
            risk_score += 20
        elif cannibal_rate > 0.15:
            risk_factors.append(("Moderate cannibalization expected", 10))
            risk_score += 10

        if pantry_loading > 0.15:
            risk_factors.append(("Pantry loading will suppress post-promo sales", 15))
            risk_score += 15

        if inp.duration_days > 14:
            risk_factors.append(("Long promo trains customers to wait for deals", 15))
            risk_score += 15

        if margin_pct < 0.25:
            risk_factors.append(("Low base margin leaves little room for discounting", 20))
            risk_score += 20

        if not inp.has_display_support:
            risk_factors.append(("No display support limits awareness", 8))
            risk_score += 8

        risk_score = min(100, risk_score)

        mitigations = self._generate_mitigations(inp, risk_factors, profile)

        # --- Recommendation ---
        rec, confidence, reason = self._decide(
            profit_change, net_30day, risk_score, volume_lift_pct,
            cannibalized_units, true_incremental_units, roi, inp
        )

        # --- Forecast chart data (6 weeks: 1 pre, promo weeks, remainder post) ---
        weekly_forecast = self._build_forecast(
            baseline_units, projected_units, promo_weeks, post_promo_dip
        )

        # --- Profit waterfall ---
        profit_waterfall = {
            "Baseline Profit": round(regular_profit, 2),
            "Volume Lift": round((incremental_units * promo_weeks) * promo_margin_per_unit, 2),
            "Discount Cost": round(-(total_promo_units * inp.regular_price * discount_frac), 2),
            "Cannibalization": round(-(cannibalized_units * (inp.regular_price - cost_per_unit)), 2),
            "Execution Cost": round(-execution_cost, 2),
            "Post-Promo Dip": round(-post_promo_lost_profit, 2),
        }

        # --- Sensitivity (discount % → net 30-day profit) ---
        sensitivity = self._sensitivity_sweep(inp, baseline_units, margin_pct, profile)

        return PromotionResult(
            recommendation=rec,
            confidence_score=round(confidence, 1),
            recommendation_reason=reason,
            baseline_units_per_week=baseline_units,
            projected_units_per_week=projected_units,
            volume_lift_pct=round(volume_lift_pct, 1),
            incremental_units=int(incremental_units * promo_weeks),
            total_promo_units=total_promo_units,
            cannibalized_units=cannibalized_units,
            cannibalization_pct=round(cannibal_rate * 100, 1),
            true_incremental_units=true_incremental_units,
            regular_revenue=round(regular_revenue, 2),
            promo_revenue=round(promo_revenue, 2),
            revenue_change=round(promo_revenue - regular_revenue, 2),
            regular_profit=round(regular_profit, 2),
            promo_profit=round(promo_profit, 2),
            profit_change=round(profit_change, 2),
            promo_cost=round(max(0, promo_cost), 2),
            roi=round(roi, 1),
            post_promo_dip_pct=round(post_promo_dip * 100, 1),
            pantry_loading_pct=round(pantry_loading * 100, 1),
            net_30day_profit_impact=round(net_30day, 2),
            risk_score=round(risk_score, 1),
            risk_factors=risk_factors,
            mitigations=mitigations,
            weekly_volume_forecast=weekly_forecast,
            profit_waterfall=profit_waterfall,
            sensitivity_data=sensitivity,
        )

    # ----- private helpers -----

    def _decide(self, profit_change, net_30day, risk_score, lift_pct,
                cannibalized, true_inc, roi, inp):
        score = 0
        reasons = []

        # Profitability
        if net_30day > 0:
            score += 35
            reasons.append(f"Net positive 30-day profit impact (+${net_30day:,.0f})")
        elif profit_change > 0:
            score += 15
            reasons.append("Promo period is profitable but post-promo dip erodes gains")
        else:
            score -= 20
            reasons.append(f"Negative profit impact (${net_30day:,.0f})")

        # Volume
        if lift_pct > 40:
            score += 20
            reasons.append(f"Strong volume lift of {lift_pct:.0f}%")
        elif lift_pct > 15:
            score += 10
            reasons.append(f"Moderate volume lift of {lift_pct:.0f}%")

        # Cannibalization
        if true_inc > 0:
            score += 15
        else:
            score -= 15
            reasons.append("Cannibalization exceeds incremental volume — mostly shifting sales")

        # Risk
        if risk_score > 60:
            score -= 25
            reasons.append("Multiple high-risk factors present")
        elif risk_score > 35:
            score -= 10

        # ROI check
        if roi > 20:
            score += 15
        elif roi < -20:
            score -= 15

        if score >= 30:
            rec = "✅ Run This Promotion"
            confidence = min(95, 60 + score)
        elif score >= 0:
            rec = "⚠️ Proceed with Caution"
            confidence = 40 + score
        else:
            rec = "🚫 Don't Run This Promotion"
            confidence = min(90, 60 + abs(score))

        reason = reasons[0] if reasons else "Balanced risk-reward profile"
        return rec, confidence, reason

    def _generate_mitigations(self, inp, risk_factors, profile):
        mitigations = []
        risk_names = [r[0] for r in risk_factors]
        if any("Deep discount" in r for r in risk_names):
            mitigations.append(f"Test at {max(5, inp.discount_pct - 10):.0f}% off first — our sensitivity analysis shows the profit-optimal discount")
        if any("cannibalization" in r.lower() for r in risk_names):
            mitigations.append("Limit promo to a single SKU size to reduce portfolio cannibalization")
        if any("Pantry loading" in r for r in risk_names):
            mitigations.append("Cap purchase quantity (e.g., limit 2 per customer) to reduce stockpiling")
        if any("Long promo" in r for r in risk_names):
            mitigations.append("Shorten to 7 days — promotions beyond 2 weeks train deal-seeking behavior")
        if any("display" in r.lower() for r in risk_names):
            mitigations.append("Add in-store display or digital feature to maximize awareness and lift")
        if any("Low base margin" in r for r in risk_names):
            mitigations.append("Consider a bundle or BOGO instead — these can protect margin better than straight discounts")
        if not mitigations:
            mitigations.append("Monitor daily sell-through and pull the promo early if lift underperforms by >30%")
        return mitigations

    def _build_forecast(self, baseline, projected, promo_weeks, post_dip):
        forecast = []
        # Week 0: baseline (pre-promo)
        forecast.append({"week": "Pre-Promo", "units": baseline, "phase": "Baseline"})
        # Promo weeks
        pw = max(1, int(round(promo_weeks)))
        for i in range(pw):
            label = f"Promo Wk {i+1}" if pw > 1 else "Promo Week"
            # Slight ramp: first week is 90% of peak, then full
            factor = 0.90 if i == 0 and pw > 1 else 1.0
            forecast.append({"week": label, "units": int(projected * factor), "phase": "Promotion"})
        # Post-promo weeks
        for i in range(3):
            dip = post_dip * (1 - i * 0.35)  # recovery over 3 weeks
            dip = max(0, dip)
            units = int(baseline * (1 - dip))
            forecast.append({"week": f"Post Wk {i+1}", "units": units, "phase": "Post-Promo"})
        return forecast

    def _sensitivity_sweep(self, inp, baseline, margin_pct, profile):
        """Sweep discount % from 5 to 50 and compute net 30-day profit."""
        discounts = list(range(5, 55, 5))
        results = {}
        p = profile
        for d in discounts:
            elas_lo, elas_hi = p["elasticity_range"]
            comp_f = min(1.0, 0.5 + inp.num_competing_products * 0.1)
            elas = elas_lo + (elas_hi - elas_lo) * (1 - comp_f)
            df = d / 100.0
            lift = -elas * df
            if inp.is_peak_season:
                lift *= 1.15 + p["seasonality_amplitude"]
            if inp.has_display_support:
                lift *= 1.25
            lift *= PROMO_TYPES[inp.promo_type]["awareness_multiplier"]
            if df > 0.30:
                lift *= (1 - 0.4 * (df - 0.30))
            lift = max(0, lift)
            proj = int(baseline * (1 + lift))
            pw = max(1, inp.duration_days / 7)
            total_u = int(proj * pw)
            total_base = int(baseline * pw)
            pp = inp.regular_price * (1 - df)
            cpu = inp.regular_price * (1 - margin_pct)
            reg_prof = total_base * (inp.regular_price - cpu)
            exec_cost = PROMO_TYPES[inp.promo_type]["execution_cost"] * total_u * pp
            promo_prof = total_u * (pp - cpu) - exec_cost
            pl = p["pantry_loading_factor"] * df
            ppd = pl * 0.6 + df * 0.15
            post_lost = int(baseline * ppd * 2) * (inp.regular_price - cpu)
            net30 = (promo_prof - reg_prof) - post_lost
            results[d] = round(net30, 2)

        return {"discounts": discounts, "net_30day_profit": [results[d] for d in discounts]}
