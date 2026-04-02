import numpy as np
import pandas as pd
from simulator import PromotionSimulator, PromotionInput, CATEGORY_PROFILES, PROMO_TYPES


def randomize_inputs():
    cat = np.random.choice(list(CATEGORY_PROFILES.keys()))
    price = np.random.uniform(2, 15)
    cost = price * np.random.uniform(0.4, 0.75)
    return PromotionInput(
        product_name="Synthetic Product",
        category=cat,
        regular_price=round(price, 2),
        unit_cost=round(cost, 2),
        baseline_units_per_week=np.random.randint(100, 800),
        discount_pct=np.random.choice([5, 10, 15, 20, 25, 30]),
        promo_type=np.random.choice(list(PROMO_TYPES.keys())),
        duration_days=np.random.choice([3, 5, 7, 10, 14]),
        num_competing_products=np.random.randint(1, 12),
        has_display_support=np.random.choice([True, False]),
        is_peak_season=np.random.choice([True, False]),
    )


def generate_training_data(n=2000):
    np.random.seed(42)
    sim = PromotionSimulator()
    rows = []
    for _ in range(n):
        inp = randomize_inputs()
        result = sim.analyze(inp)
        rows.append({
            "discount": inp.discount_pct,
            "duration": inp.duration_days,
            "price": inp.regular_price,
            "cost": inp.unit_cost,
            "baseline": inp.baseline_units_per_week,
            "competition": inp.num_competing_products,
            "display": int(inp.has_display_support),
            "peak": int(inp.is_peak_season),
            "profit": result.net_30day_profit_impact,
        })
    return pd.DataFrame(rows)
