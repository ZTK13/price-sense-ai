import streamlit as st
from simulator import PromotionSimulator, PromotionInput
from ml_model import train_model, predict_with_model, get_feature_importance

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Price Sense AI", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return train_model()

model, feature_cols = load_model()

# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.ml_profit = None
    st.session_state.input = None

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Price Sense AI")

st.markdown(
    """
    <p>
    <strong>Price Sense AI</strong> helps mid-market retailers 
    ($50M–$500M revenue) make data-driven promotion decisions.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
with st.sidebar:
    st.header("Promotion Inputs")

    product_name = st.text_input("Product Name", "Salted Pistachios")
    category = st.selectbox("Category", ["Snacks", "Beverages", "Dairy"])

    regular_price = st.number_input("Regular Price ($)", 1.0, 20.0, 5.0)
    unit_cost = st.number_input("Unit Cost ($)", 0.5, 10.0, 2.5)

    baseline_units = st.number_input("Baseline Units/Week", 50, 1000, 200)

    discount = st.slider("Discount %", 5, 30, 20)
    duration = st.slider("Duration (days)", 3, 14, 7)

    competition = st.slider("Competing Products", 1, 10, 4)

    display = st.checkbox("Display Support", True)
    peak = st.checkbox("Peak Season", False)

    analyze = st.button("🚀 Analyze Promotion")

# -------------------------------
# ANALYSIS LOGIC
# -------------------------------
if analyze:
    inp = PromotionInput(
        product_name=product_name,
        category=category,
        regular_price=regular_price,
        unit_cost=unit_cost,
        baseline_units_per_week=baseline_units,
        discount_pct=discount,
        duration_days=duration,
        num_competing_products=competition,
        has_display_support=display,
        is_peak_season=peak,
    )

    sim = PromotionSimulator()
    result = sim.analyze(inp)

    ml_profit = predict_with_model(model, feature_cols, inp)

    st.session_state.result = result
    st.session_state.ml_profit = ml_profit
    st.session_state.input = inp

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
if st.session_state.result:
    result = st.session_state.result
    inp = st.session_state.input
    ml_profit = st.session_state.ml_profit

    # ---------------------------
    # RECOMMENDATION
    # ---------------------------
    st.subheader("📌 Recommendation")
    st.write(f"**{result.recommendation}**")

    # ---------------------------
    # METRICS
    # ---------------------------
    st.subheader("📊 Key Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Volume Lift", f"{result.volume_lift_pct:.1f}%")
    c2.metric("True Incremental Units", f"{result.true_incremental_units}")
    c3.metric("Net Profit (Simulation)", f"${result.net_30day_profit_impact:,.0f}")

    # ---------------------------
    # ML PREDICTION
    # ---------------------------
    st.subheader("🤖 ML Prediction")

    st.write(
        f"Simulation Estimate: **${result.net_30day_profit_impact:,.0f}**  \n"
        f"ML Prediction: **${ml_profit:,.0f}**"
    )

    # ---------------------------
    # AGREEMENT / DISAGREEMENT SIGNAL
    # ---------------------------
    diff = abs(result.net_30day_profit_impact - ml_profit)

    if diff > 200:
        st.warning("⚠️ Model and simulation strongly disagree — investigate scenario")
    elif diff > 100:
        st.info("ℹ️ Moderate difference between model and simulation")
    else:
        st.success("✅ Model and simulation are aligned")

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    st.subheader("📊 Key Drivers")

    feat_imp = get_feature_importance(model, feature_cols)[:5]

    for f, imp in feat_imp:
        st.write(f"{f.capitalize()}: {imp:.2f}")

    # ---------------------------
    # REASONING
    # ---------------------------
    st.subheader("🧠 Why this recommendation?")

    if result.net_30day_profit_impact < 0:
        st.write("❌ Negative net profit impact")
    else:
        st.write("✅ Positive net profit impact")

    if result.cannibalization_pct > 20:
        st.write("⚠️ High cannibalization risk")

    if inp.discount_pct > 25:
        st.write("⚠️ Deep discount reduces margins")

    # ---------------------------
    # RESET BUTTON
    # ---------------------------
    st.markdown("---")
    if st.button("← Reset Analysis"):
        st.session_state.clear()
        st.rerun()

# -------------------------------
# EMPTY STATE
# -------------------------------
else:
    st.info("👈 Configure inputs in the sidebar and click **Analyze Promotion**")
