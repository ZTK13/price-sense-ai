import numpy as np
from dataclasses import dataclass
from typing import Optional

# Category-specific profiles grounded in retail economics
CATEGORY_PROFILES = {
    "Grocery": {
        "elasticity_range": (-2.5, -1.2), "base_margin": 0.28,
        "cannibalization_rate": 0.18, "pantry_loading_factor": 0.25,
        "seasonality_amplitude": 0.08, "typical_weekly_units": 500, "avg_price": 5.99,
    },
    "Specialty Food & Nuts": {
        "elasticity_range": (-2.0, -0.9), "base_margin": 0.40,
        "cannibalization_rate": 0.22, "pantry_loading_factor": 0.30,
        "seasonality_amplitude": 0.12, "typical_weekly_units": 200, "avg_price": 9.99,
    },
    "Beverages": {
        "elasticity_range": (-3.0, -1.5), "base_margin": 0.35,
        "cannibalization_rate": 0.20, "pantry_loading_factor": 0.35,
        "seasonality_amplitude": 0.15, "typical_weekly_units": 800, "avg_price": 3.49,
    },
    "Snacks & Confectionery": {
        "elasticity_range": (-2.8, -1.3), "base_margin": 0.42,
        "cannibalization_rate": 0.25, "pantry_loading_factor": 0.20,
        "seasonality_amplitude": 0.10, "typical_weekly_units": 600, "avg_price": 4.49,
    },
    "Dairy & Refrigerated": {
        "elasticity_range": (-2.2, -1.0), "base_margin": 0.25,
        "cannibalization_rate": 0.15, "pantry_loading_factor": 0.10,
        "seasonality_amplitude": 0.06, "typical_weekly_units": 700, "avg_price": 4.99,
    },
    "Health & Wellness": {
        "elasticity_range": (-1.8, -0.6), "base_margin": 0.50,
        "cannibalization_rate": 0.12, "pantry_loading_factor": 0.15,
        "seasonality_amplitude": 0.05, "typical_weekly_units": 150, "avg_price": 14.99,
    },
    "Household & Cleaning": {
        "elasticity_range": (-2.0, -1.0), "base_margin": 0.30,
        "cannibalization_rate": 0.10, "pantry_loading_factor": 0.40,
        "seasonality_amplitude": 0.04, "typical_weekly_units": 300, "avg_price": 7.99,
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
    unit_cost: float
    baseline_units_per_week: int
    discount_pct: float
    promo_type: str
    duration_days: int
    num_competing_products: int
    has_display_support: bool
    is_peak_season: bool


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

    def analyze(self, inp: PromotionInput) -> PromotionResult:
        profile = CATEGORY_PROFILES[inp.category]
        baseline = inp.baseline_units_per_week
        margin_pct = (inp.regular_price - inp.unit_cost) / inp.regular_price

        # --- Price elasticity (category-specific) ---
        elas_lo, elas_hi = profile["elasticity_range"]
        comp_factor = min(1.0, 0.5 + inp.num_competing_products * 0.1)
        elasticity = elas_lo + (elas_hi - elas_lo) * (1 - comp_factor)
        elasticity += self.rng.normal(0, 0.1)

        # --- Volume lift ---
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
        projected = int(baseline * (1 + max(0, raw_lift)))
        incremental = projected - baseline

        # --- Duration ---
        promo_weeks = max(1, inp.duration_days / 7)
        total_promo_units = int(projected * promo_weeks)
        total_baseline_units = int(baseline * promo_weeks)

        # --- Cannibalization ---
        cannibal_rate = profile["cannibalization_rate"]
        cannibal_rate *= min(1.5, 1.0 + inp.num_competing_products * 0.05)
        if inp.promo_type == "Bundle Deal":
            cannibal_rate *= 0.7
        cannibalized = int(incremental * promo_weeks * cannibal_rate)
        true_incremental = int(incremental * promo_weeks - cannibalized)

        # --- Financials ---
        promo_price = inp.regular_price * (1 - discount_frac)
        cost = inp.unit_cost
        regular_revenue = total_baseline_units * inp.regular_price
        promo_revenue = total_promo_units * promo_price
        regular_profit = total_baseline_units * (inp.regular_price - cost)
        promo_margin = promo_price - cost
        exec_cost = promo_info["execution_cost"] * promo_revenue
        promo_profit = total_promo_units * promo_margin - exec_cost
        profit_change = promo_profit - regular_profit
        promo_cost = regular_profit - promo_profit
        roi = (profit_change / max(abs(promo_cost), 1)) * 100 if promo_cost != 0 else 0

        # --- Post-promo effects ---
        pantry_loading = profile["pantry_loading_factor"] * discount_frac
        post_promo_dip = pantry_loading * 0.6 + discount_frac * 0.15
        post_lost_units = int(baseline * post_promo_dip * 2)
        post_lost_profit = post_lost_units * (inp.regular_price - cost)
        net_30day = profit_change - post_lost_profit

        # --- Risk scoring ---
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

        # --- Mitigations ---
        mitigations = []
        rn = [r[0] for r in risk_factors]
        if any("Deep discount" in r for r in rn):
            mitigations.append(f"Test at {max(5, inp.discount_pct - 10):.0f}% off first")
        if any("cannibalization" in r.lower() for r in rn):
            mitigations.append("Limit promo to a single SKU size to reduce portfolio cannibalization")
        if any("Pantry loading" in r for r in rn):
            mitigations.append("Cap purchase quantity (limit 2 per customer) to reduce stockpiling")
        if any("Long promo" in r for r in rn):
            mitigations.append("Shorten to 7 days — promotions beyond 2 weeks train deal-seeking behavior")
        if any("display" in r.lower() for r in rn):
            mitigations.append("Add in-store display or digital feature to maximize awareness")
        if any("Low base margin" in r for r in rn):
            mitigations.append("Consider a bundle or BOGO — can protect margin better than straight discounts")
        if not mitigations:
            mitigations.append("Monitor daily sell-through and pull promo early if lift underperforms by >30%")

        # --- Recommendation ---
        rec, confidence, reason = self._decide(profit_change, net_30day, risk_score, volume_lift_pct, true_incremental, roi)

        # --- Weekly forecast (pre → promo → post) ---
        forecast = [{"week": "Pre-Promo", "units": baseline, "phase": "Baseline"}]
        pw = max(1, int(round(promo_weeks)))
        for i in range(pw):
            label = f"Promo Wk {i+1}" if pw > 1 else "Promo Week"
            factor = 0.90 if i == 0 and pw > 1 else 1.0
            forecast.append({"week": label, "units": int(projected * factor), "phase": "Promotion"})
        for i in range(3):
            dip = max(0, post_promo_dip * (1 - i * 0.35))
            forecast.append({"week": f"Post Wk {i+1}", "units": int(baseline * (1 - dip)), "phase": "Post-Promo"})

        # --- Profit waterfall ---
        waterfall = {
            "Baseline Profit": round(regular_profit, 2),
            "Volume Lift": round((incremental * promo_weeks) * promo_margin, 2),
            "Discount Cost": round(-(total_promo_units * inp.regular_price * discount_frac), 2),
            "Cannibalization": round(-(cannibalized * (inp.regular_price - cost)), 2),
            "Execution Cost": round(-exec_cost, 2),
            "Post-Promo Dip": round(-post_lost_profit, 2),
        }

        # --- Sensitivity sweep ---
        sensitivity = self._sensitivity_sweep(inp, baseline, margin_pct, profile)

        return PromotionResult(
            recommendation=rec, confidence_score=round(confidence, 1),
            recommendation_reason=reason,
            baseline_units_per_week=baseline, projected_units_per_week=projected,
            volume_lift_pct=round(volume_lift_pct, 1),
            incremental_units=int(incremental * promo_weeks),
            total_promo_units=total_promo_units,
            cannibalized_units=cannibalized,
            cannibalization_pct=round(cannibal_rate * 100, 1),
            true_incremental_units=true_incremental,
            regular_revenue=round(regular_revenue, 2), promo_revenue=round(promo_revenue, 2),
            revenue_change=round(promo_revenue - regular_revenue, 2),
            regular_profit=round(regular_profit, 2), promo_profit=round(promo_profit, 2),
            profit_change=round(profit_change, 2), roi=round(roi, 1),
            post_promo_dip_pct=round(post_promo_dip * 100, 1),
            pantry_loading_pct=round(pantry_loading * 100, 1),
            net_30day_profit_impact=round(net_30day, 2),
            risk_score=round(risk_score, 1), risk_factors=risk_factors,
            mitigations=mitigations, weekly_volume_forecast=forecast,
            profit_waterfall=waterfall, sensitivity_data=sensitivity,
        )

    def _decide(self, profit_change, net_30day, risk_score, lift_pct, true_inc, roi):
        score = 0; reasons = []
        if net_30day > 0:
            score += 35; reasons.append(f"Net positive 30-day profit impact (+${net_30day:,.0f})")
        elif profit_change > 0:
            score += 15; reasons.append("Promo period profitable but post-promo dip erodes gains")
        else:
            score -= 20; reasons.append(f"Negative profit impact (${net_30day:,.0f})")
        if lift_pct > 40: score += 20; reasons.append(f"Strong volume lift of {lift_pct:.0f}%")
        elif lift_pct > 15: score += 10
        if true_inc > 0: score += 15
        else: score -= 15; reasons.append("Cannibalization exceeds incremental volume")
        if risk_score > 60: score -= 25; reasons.append("Multiple high-risk factors")
        elif risk_score > 35: score -= 10
        if roi > 20: score += 15
        elif roi < -20: score -= 15

        if score >= 30:
            return "✅ Run This Promotion", min(95, 60 + score), reasons[0] if reasons else "Strong overall profile"
        elif score >= 0:
            return "⚠️ Proceed with Caution", 40 + score, reasons[0] if reasons else "Balanced risk-reward"
        else:
            return "🚫 Don't Run This Promotion", min(90, 60 + abs(score)), reasons[0] if reasons else "Unfavorable economics"

    def _sensitivity_sweep(self, inp, baseline, margin_pct, profile):
        discounts = list(range(5, 55, 5)); results = {}
        for d in discounts:
            elas_lo, elas_hi = profile["elasticity_range"]
            comp_f = min(1.0, 0.5 + inp.num_competing_products * 0.1)
            elas = elas_lo + (elas_hi - elas_lo) * (1 - comp_f) + self.rng.normal(0, 0.1)
            df = d / 100.0; lift = -elas * df
            if inp.is_peak_season: lift *= 1.15 + profile["seasonality_amplitude"]
            if inp.has_display_support: lift *= 1.25
            lift *= PROMO_TYPES[inp.promo_type]["awareness_multiplier"]
            if df > 0.30: lift *= (1 - 0.4 * (df - 0.30))
            lift = max(0, lift); proj = int(baseline * (1 + lift))
            pw = max(1, inp.duration_days / 7)
            total_u = int(proj * pw); total_base = int(baseline * pw)
            pp = inp.regular_price * (1 - df)
            reg_prof = total_base * (inp.regular_price - inp.unit_cost)
            ec = PROMO_TYPES[inp.promo_type]["execution_cost"] * total_u * pp
            promo_prof = total_u * (pp - inp.unit_cost) - ec
            pl = profile["pantry_loading_factor"] * df; ppd = pl * 0.6 + df * 0.15
            post_lost = int(baseline * ppd * 2) * (inp.regular_price - inp.unit_cost)
            results[d] = round((promo_prof - reg_prof) - post_lost, 2)
        return {"discounts": discounts, "net_30day_profit": [results[d] for d in discounts]}
