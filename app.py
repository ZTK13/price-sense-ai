"""
Price Sense AI — Should You Run This Promotion?
AI-powered promotion analysis for mid-market retailers.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from simulator import PromotionSimulator, PromotionInput, PromotionResult, CATEGORY_PROFILES, PROMO_TYPES
from ml_model import train_model, predict_with_model, get_feature_importance

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Price Sense AI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
     /* =========================
       MAIN HEADER
    ========================= */
    .main-header {
        background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(51,65,85,0.7));
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148,163,184,0.2);
    }
    .main-header h1 { color: #f8fafc; font-size: 2rem; margin: 0; font-weight: 700; }
    .main-header p { color: rgba(226,232,240,0.9); font-size: 1.05rem; margin: 0.5rem 0 0 0; }
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
    .rec-card { padding: 1.5rem 2rem;
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
        font-size: 0.95rem;
    }
    
    /* =========================
       SECTION HEADERS
    ========================= */
    .section-header {
        font-size: 1.15rem;
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

# ── Load ML Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return train_model()

model, feature_cols = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Price Sense AI <span class="badge">BETA</span></h1>
    <p>Should you run this promotion? Get AI-powered analysis of projected lift,
    cannibalization, profit impact, and risk — in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Inputs ───────────────────────────────────────────────────────────
def collapse_sidebar():
    st.markdown("""
        <script>
        setTimeout(() => {
            const btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (btn) btn.click();
        }, 100);
        </script>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎯 Promotion Setup")
    st.markdown("___")
    st.markdown("#### Product Details")

    category = st.selectbox("Product Category", options=list(CATEGORY_PROFILES.keys()), index=1)
    product_name = st.text_input("Product Name", value="Salted Pistachios 16oz")
    regular_price = st.number_input("Regular Price ($)", min_value=0.50, max_value=100.0, value=12.99, step=0.50)
    unit_cost = st.number_input("Unit Cost ($)", min_value=0.25, max_value=50.0, value=7.79, step=0.25)

    st.markdown("___")
    st.markdown("#### Promotion Details")

    promo_type = st.selectbox("Promotion Type", options=list(PROMO_TYPES.keys()))
    discount_pct = st.slider("Discount %", 5, 50, 25, step=5)
    duration_days = st.slider("Duration (days)", 1, 28, 7)

    st.markdown("___")
    st.markdown("#### Market Context")

    baseline_units = st.number_input("Baseline Units/Week", 50, 2000, 200)
    num_competing = st.slider("Competing Products", 0, 15, 4)
    is_peak = st.checkbox("Peak Season?", value=False)
    has_display = st.checkbox("In-Store Display Support?", value=True)

    st.markdown("___")
    analyze_btn = st.button("🚀 Analyze Promotion", use_container_width=True, type="primary")

# ── Summary Generator ────────────────────────────────────────────────────────
def _generate_summary(result, inp, ml_profit):
    promo_price = inp.regular_price * (1 - inp.discount_pct / 100)

    # Opener
    if "Don't" in result.recommendation:
        opener = (
            f"<strong>This promotion is not recommended.</strong> At {inp.discount_pct}% off, "
            f"<strong>{inp.product_name}</strong> is projected to lose money on a net basis."
        )
    elif "Run" in result.recommendation:
        opener = (
            f"<strong>This promotion is recommended.</strong> Running {inp.discount_pct}% off on "
            f"<strong>{inp.product_name}</strong> (${inp.regular_price:.2f} → ${promo_price:.2f}) "
            f"for {inp.duration_days} days is projected to be profitable."
        )
    else:
        opener = (
            f"<strong>This promotion warrants caution.</strong> While {inp.discount_pct}% off "
            f"on <strong>{inp.product_name}</strong> will drive traffic, several factors limit its effectiveness."
        )

    # Volume
    vol = (
        f"We project a <strong>{result.volume_lift_pct:.0f}% volume lift</strong> "
        f"({result.baseline_units_per_week:,} → {result.projected_units_per_week:,} units/week). "
        f"However, approximately <strong>{result.cannibalization_pct:.0f}%</strong> of incremental volume comes from "
        f"cannibalization of adjacent products, leaving <strong>{result.true_incremental_units:,} true incremental units</strong>."
    )

    # Net profit
    if result.net_30day_profit_impact >= 0:
        prof = (
            f"The net 30-day profit impact is <strong>+${result.net_30day_profit_impact:,.0f}</strong>, "
            f"including the post-promo dip effect."
        )
    else:
        prof = (
            f"The net 30-day impact is <strong>-${abs(result.net_30day_profit_impact):,.0f}</strong> "
            f"after accounting for the {result.post_promo_dip_pct:.0f}% post-promo dip."
        )

    ml_note = f"Our ML model independently predicts <strong>${ml_profit:,.0f}</strong> net profit for this scenario."

    # Sensitivity
    sens = result.sensitivity_data
    best_idx = max(range(len(sens["net_30day_profit"])), key=lambda i: sens["net_30day_profit"][i])
    best_d = sens["discounts"][best_idx]
    if best_d != int(inp.discount_pct):
        opt = (
            f"💡 <strong>Optimization insight:</strong> Sensitivity analysis suggests "
            f"<strong>{best_d}% off</strong> may be the profit-maximizing discount."
        )
    else:
        opt = f"Your proposed {inp.discount_pct}% discount is at or near the profit-optimal point."

    return f"""
            {opener}

            <div class="section-header">📈 Volume Impact</div>
            <p>{vol}</p>

            <div class="section-header">💰 Profit Impact</div>
            <p>{prof}</p>
            
            <div class="section-header">🤖 ML Prediction</div>
            <p>{ml_note}</p>

            <div class="section-header">💡 Optimization Insight</div>
            <p>{opt}</p>
            """

# ── Analysis ─────────────────────────────────────────────────────────────────
if analyze_btn or "result" in st.session_state:
    if analyze_btn:
        inp = PromotionInput(
            product_name=product_name, category=category,
            regular_price=regular_price, unit_cost=unit_cost,
            baseline_units_per_week=baseline_units,
            discount_pct=discount_pct, promo_type=promo_type,
            duration_days=duration_days,
            num_competing_products=num_competing,
            is_peak_season=is_peak, has_display_support=has_display,
        )
        sim = PromotionSimulator()
        result = sim.analyze(inp)
        ml_profit = predict_with_model(model, feature_cols, inp)
        st.session_state["result"] = result
        st.session_state["inp"] = inp
        st.session_state["ml_profit"] = ml_profit
    else:
        result = st.session_state["result"]
        inp = st.session_state["inp"]
        ml_profit = st.session_state["ml_profit"]

    # ── Recommendation Card ──────────────────────────────────────────────
    css_class = "rec-stop" if "Don't" in result.recommendation else "rec-run" if "Run" in result.recommendation else "rec-caution"
    st.markdown(f"""
    <div class="rec-card {css_class}">
        <h2>{result.recommendation}</h2>
        <p><strong>Confidence:</strong> {result.confidence_score}% · {result.recommendation_reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──────────────────────────────────────────────────────
    promo_price = inp.regular_price * (1 - inp.discount_pct / 100)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Volume Lift", f"+{result.volume_lift_pct}%",
                  f"{result.true_incremental_units:,} true incremental")
    with c2:
        sign = "+" if result.profit_change > 0 else ""
        st.metric("Promo Profit Impact", f"{sign}${result.profit_change:,.0f}",
                  f"ROI: {result.roi:.0f}%")
    with c3:
        st.metric("Net 30-Day Impact", f"${result.net_30day_profit_impact:,.0f}",
                  f"Post-promo dip: {result.post_promo_dip_pct:.0f}%", delta_color="normal")
    with c4:
        risk_label = "Low" if result.risk_score < 30 else "Medium" if result.risk_score < 60 else "High"
        st.metric("Risk Score", f"{result.risk_score:.0f}/100", risk_label, delta_color="inverse")

    st.markdown("___")

    # ── Charts (Tabs) ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Volume Forecast", "💰 Profit Waterfall",
        "🔍 Sensitivity Analysis", "📊 Full Breakdown"
    ])

    with tab1:
        st.markdown('<div class="section-header">6-Week Volume Forecast</div>', unsafe_allow_html=True)
        df_f = pd.DataFrame(result.weekly_volume_forecast)
        colors = {"Baseline": "#94a3b8", "Promotion": "#3b82f6", "Post-Promo": "#f59e0b"}
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_f["week"], y=df_f["units"],
            marker_color=[colors[p] for p in df_f["phase"]],
            text=df_f["units"], textposition="outside",
            textfont=dict(size=12, color="#334155"),
        ))
        fig.add_hline(y=result.baseline_units_per_week, line_dash="dash",
                      line_color="#94a3b8", annotation_text="Baseline",
                      annotation_position="top right")
        fig.update_layout(height=400, plot_bgcolor="white",
                          xaxis=dict(title=""), yaxis=dict(title="Units / Week", gridcolor="#f1f5f9"),
                          margin=dict(l=60, r=20, t=30, b=40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📌 Baseline: {result.baseline_units_per_week:,} · "
                   f"During promo: {result.projected_units_per_week:,} · "
                   f"Cannibalized: {result.cannibalized_units:,} ({result.cannibalization_pct:.0f}%)")

    with tab2:
        st.markdown('<div class="section-header">Profit Impact Waterfall</div>', unsafe_allow_html=True)
        wf = result.profit_waterfall
        labels = list(wf.keys()) + ["Net Impact"]
        values = list(wf.values()); net = sum(values)
        measures = ["absolute"] + ["relative"] * (len(values) - 1) + ["total"]
        values.append(net)
        fig2 = go.Figure(go.Waterfall(
            x=labels, y=values, measure=measures,
            connector={"line": {"color": "#e2e8f0"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#3b82f6"}},
            textposition="outside",
            text=[f"${v:,.0f}" for v in values], textfont=dict(size=11),
        ))
        fig2.update_layout(height=420, plot_bgcolor="white",
                           yaxis=dict(title="Profit ($)", gridcolor="#f1f5f9"),
                           xaxis=dict(tickfont=dict(size=10)),
                           margin=dict(l=60, r=20, t=30, b=60), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Discount Sensitivity — Net 30-Day Profit</div>', unsafe_allow_html=True)
        sens = result.sensitivity_data
        cd = int(inp.discount_pct)
        cs = ["#3b82f6" if d == cd else "#cbd5e1" for d in sens["discounts"]]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=[f"{d}%" for d in sens["discounts"]], y=sens["net_30day_profit"],
            marker_color=cs,
            text=[f"${v:,.0f}" for v in sens["net_30day_profit"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig3.add_hline(y=0, line_color="#94a3b8", line_width=1)
        fig3.update_layout(height=400, plot_bgcolor="white",
                           yaxis=dict(title="Net 30-Day Profit ($)", gridcolor="#f1f5f9"),
                           margin=dict(l=60, r=20, t=30, b=40), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        best_idx = max(range(len(sens["net_30day_profit"])), key=lambda i: sens["net_30day_profit"][i])
        best_d = sens["discounts"][best_idx]; best_p = sens["net_30day_profit"][best_idx]
        if best_d != cd:
            st.info(f"💡 **Optimal discount: {best_d}%** → ${best_p:,.0f} net profit. "
                    f"Your {cd}% yields ${sens['net_30day_profit'][sens['discounts'].index(cd)]:,.0f}.")
        else:
            st.success(f"✅ Your {cd}% discount is at or near the profit-optimal point.")

    with tab4:
        st.markdown('<div class="section-header">Complete Analysis</div>', unsafe_allow_html=True)
        cl, cr = st.columns(2)
        with cl:
            st.markdown("##### 📦 Volume Analysis")
            st.dataframe(pd.DataFrame({
                "Metric": ["Baseline Weekly Units", "Projected Weekly Units", "Volume Lift %",
                           "Total Promo Units", "Gross Incremental", "Cannibalized Units",
                           "True Incremental", "Cannibalization Rate"],
                "Value": [f"{result.baseline_units_per_week:,}", f"{result.projected_units_per_week:,}",
                          f"+{result.volume_lift_pct}%", f"{result.total_promo_units:,}",
                          f"{result.incremental_units:,}", f"-{result.cannibalized_units:,}",
                          f"{result.true_incremental_units:,}", f"{result.cannibalization_pct:.1f}%"],
            }), hide_index=True, use_container_width=True)
        with cr:
            st.markdown("##### 💵 Financial Analysis")
            st.dataframe(pd.DataFrame({
                "Metric": ["Regular Price", "Promo Price", "Baseline Revenue", "Promo Revenue",
                           "Revenue Change", "Baseline Profit", "Promo Profit", "Profit Change",
                           "Net 30-Day Impact", "Pantry Loading", "Post-Promo Dip"],
                "Value": [f"${inp.regular_price:.2f}", f"${promo_price:.2f}",
                          f"${result.regular_revenue:,.2f}", f"${result.promo_revenue:,.2f}",
                          f"${result.revenue_change:,.2f}", f"${result.regular_profit:,.2f}",
                          f"${result.promo_profit:,.2f}", f"${result.profit_change:,.2f}",
                          f"${result.net_30day_profit_impact:,.2f}",
                          f"{result.pantry_loading_pct:.1f}%", f"{result.post_promo_dip_pct:.1f}%"],
            }), hide_index=True, use_container_width=True)

    # ── Risk & Mitigations ───────────────────────────────────────────────
    st.markdown("___")
    col_risk, col_mit = st.columns(2)
    with col_risk:
        st.markdown('<div class="section-header">⚠️ Risk Factors</div>', unsafe_allow_html=True)
        if result.risk_factors:
            for factor, severity in result.risk_factors:
                dc = "risk-high" if severity >= 20 else "risk-med" if severity >= 10 else "risk-low"
                st.markdown(f'<div class="risk-item"><div class="risk-dot {dc}"></div>'
                            f'<span>{factor} <em style="color:#94a3b8">(+{severity} pts)</em></span></div>',
                            unsafe_allow_html=True)
        else:
            st.success("No significant risk factors identified.")
    with col_mit:
        st.markdown('<div class="section-header">🛡️ Recommended Mitigations</div>', unsafe_allow_html=True)
        for i, m in enumerate(result.mitigations, 1):
            st.markdown(
                f"<strong>{i}.</strong> {m}",
                unsafe_allow_html=True
            )

    # ── ML Prediction & Agreement ────────────────────────────────────────
    st.markdown("___")
    ml_col, feat_col = st.columns(2)
    with ml_col:
        st.markdown('<div class="section-header">🤖 ML Model Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f"<strong>Simulation estimate:</strong> ${result.net_30day_profit_impact:,.0f}",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<strong>ML prediction:</strong> ${ml_profit:,.0f}",
            unsafe_allow_html=True
        )
        diff = abs(result.net_30day_profit_impact - ml_profit)
        if diff > 200:
            st.warning("⚠️ Model and simulation strongly disagree — investigate scenario")
        elif diff > 100:
            st.info("ℹ️ Moderate difference between model and simulation")
        else:
            st.success("✅ Model and simulation are aligned")

    with feat_col:
        st.markdown('<div class="section-header">📊 Top Feature Drivers (ML)</div>', unsafe_allow_html=True)
        feat_imp = get_feature_importance(model, feature_cols)[:5]
        feat_df = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
        feat_df["Feature"] = feat_df["Feature"].str.capitalize()
        fig_feat = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"],
            orientation="h", marker_color="#3b82f6",
            text=[f"{v:.2f}" for v in feat_df["Importance"]],
            textposition="outside",
        ))
        fig_feat.update_layout(height=250, plot_bgcolor="white",
                               yaxis=dict(autorange="reversed"),
                               margin=dict(l=100, r=40, t=10, b=20), showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)

    # ── AI Executive Summary ─────────────────────────────────────────────
    st.markdown("___")
    st.markdown('<div class="section-header">🧠 AI Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(_generate_summary(result, inp, ml_profit), unsafe_allow_html=True)

    # ── Reset ────────────────────────────────────────────────────────────
    st.markdown("___")
    if st.button("← Reset Analysis"):
        st.session_state.clear()
        st.rerun()

else:
    # ── Landing State ────────────────────────────────────────────────────
    st.markdown("### 👈 Configure a promotion in the sidebar and click <strong>Analyze</strong>")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📈 Volume Lift\nProjected sales increase using price elasticity and category dynamics.")
    with c2:
        st.markdown("#### 💰 Profit Impact\nTrue profitability after margin erosion, cannibalization, and execution costs.")
    with c3:
        st.markdown("#### ⚠️ Risk Analysis\nHidden costs — cannibalization, pantry loading, and post-promo dips.")
    st.markdown("___")
    st.markdown(
        """
        <p>
            <strong>Price Sense AI</strong> helps mid-market retailers 
            ($50M&nbsp;&ndash;&nbsp;$500M revenue) 
            make data-driven promotion decisions. 
            Stop relying on gut feel — know the true ROI before you run.
        </p>
        """,
        unsafe_allow_html=True
    )
