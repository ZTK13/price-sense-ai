from sklearn.ensemble import RandomForestRegressor
from data_utils import generate_training_data
import pandas as pd


def train_model():
    df = generate_training_data()
    X = df.drop(columns=["profit"])
    y = df["profit"]
    model = RandomForestRegressor(n_estimators=50, max_depth=6)
    model.fit(X, y)
    return model, X.columns


def predict_with_model(model, feature_cols, inp):
    row = pd.DataFrame([{
        "discount": inp.discount_pct,
        "duration": inp.duration_days,
        "price": inp.regular_price,
        "cost": inp.unit_cost,
        "baseline": inp.baseline_units_per_week,
        "competition": inp.num_competing_products,
        "display": int(inp.has_display_support),
        "peak": int(inp.is_peak_season),
    }])
    return model.predict(row[feature_cols])[0]


def get_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    return sorted(zip(feature_cols, importances), key=lambda x: -x[1])
