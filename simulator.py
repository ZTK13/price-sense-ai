from dataclasses import dataclass


@dataclass
class PromotionInput:
    product_name: str
    category: str
    regular_price: float
    unit_cost: float
    baseline_units_per_week: int
    discount_pct: float
    duration_days: int
    num_competing_products: int
    has_display_support: bool
    is_peak_season: bool


@dataclass
class PromotionResult:
    volume_lift_pct: float
    projected_units_per_week: int
    incremental_units: int
    true_incremental_units: int
    cannibalization_pct: float
    profit_change: float
    net_30day_profit_impact: float
    post_promo_dip_pct: float
    recommendation: str
    sensitivity_data: dict


class PromotionSimulator:
    def analyze(self, inp: PromotionInput) -> PromotionResult:
        # Simple heuristic model
        lift = 0.02 * inp.discount_pct
        if inp.has_display_support:
            lift += 0.05
        if inp.is_peak_season:
            lift += 0.04

        volume_lift_pct = lift * 100
        projected_units = int(inp.baseline_units_per_week * (1 + lift))
        incremental_units = projected_units - inp.baseline_units_per_week

        cannibalization_pct = min(0.1 + 0.02 * inp.num_competing_products, 0.4)
        true_incremental_units = int(incremental_units * (1 - cannibalization_pct))

        margin = inp.regular_price - inp.unit_cost
        promo_price = inp.regular_price * (1 - inp.discount_pct / 100)
        promo_margin = promo_price - inp.unit_cost

        profit_change = true_incremental_units * promo_margin - incremental_units * margin * 0.3

        post_promo_dip_pct = 5 + inp.discount_pct * 0.1
        net_30day_profit = profit_change - abs(profit_change) * (post_promo_dip_pct / 100)

        if net_30day_profit > 0:
            rec = "Run Promotion"
        elif net_30day_profit < -100:
            rec = "Don't Run Promotion"
        else:
            rec = "Proceed with Caution"

        # Sensitivity (simple)
        discounts = [5, 10, 15, 20, 25, 30]
        profits = []
        for d in discounts:
            p = (d * 10) - (d ** 1.5)
            profits.append(p)

        sensitivity = {"discounts": discounts, "net_30day_profit": profits}

        return PromotionResult(
            volume_lift_pct=volume_lift_pct,
            projected_units_per_week=projected_units,
            incremental_units=incremental_units,
            true_incremental_units=true_incremental_units,
            cannibalization_pct=cannibalization_pct * 100,
            profit_change=profit_change,
            net_30day_profit_impact=net_30day_profit,
            post_promo_dip_pct=post_promo_dip_pct,
            recommendation=rec,
            sensitivity_data=sensitivity
        )
