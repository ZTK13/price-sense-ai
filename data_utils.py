import numpy as np
import pandas as pd
from simulator import PromotionSimulator, PromotionInput


def randomize_inputs():
    return PromotionInput(
        product_name="Synthetic Product",
        category=np.random.choice(["Snacks", "Beverages", "Dairy"]),
        regular_price=np.random.uniform(2, 10),
        unit_cost=np.random.uniform(1, 5),
        baseline_units_per_week=np.random.randint(100, 500),
        discount_pct=np.random.choice([5, 10, 15, 20, 25, 30]),
        duration_days=np.random.choice([3, 5, 7, 10]),
        num_competing_products=np.random.randint(1, 10),
        has_display_support=np.random.choice([True, False]),
        is_peak_season=np.random.choice([True, False]),
    )


def generate_training_data(n=2000):
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
            "profit": result.net_30day_profit_impact
        })

    return pd.DataFrame(rows)
