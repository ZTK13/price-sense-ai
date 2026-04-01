"""
Price Sense AI — Should You Run This Promotion?
An AI-powered promotion analysis tool for mid-market retailers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from engine import (
    PromotionSimulator,
    PromotionInput,
    PromotionResult,
    CATEGORY_PROFILES,
    PROMO_TYPES,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Price Sense AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }
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

    /* Recommendation cards */
    .rec-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .rec-run {
        background: #f0fdf4;
        border-left-color: #22c55e;
    }
    .rec-caution {
        background: #fffbeb;
        border-left-color: #f59e0b;
    }
    .rec-stop {
        background: #fef2f2;
        border-left-color: #ef4444;
    }
    .rec-card h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }
    .rec-card p {
        margin: 0;
        color: #475569;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.3rem 0;
    }
    .metric-card .subtext {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .positive { color: #16a34a !important; }
    .negative { color: #dc2626 !important; }

    /* Risk factors */
    .risk-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #f1f5f9;
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

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    [data-testid="stSidebar"] h2 {
        color: #0f172a;
        font-size: 1.1rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Price Sense AI <span class="badge">BETA</span></h1>
    <p>Should you run this promotion? Get AI-powered analysis of projected lift, 
    cannibalization, profit impact, and risk — in seconds.</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Promotion Inputs ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Promotion Setup")
    st.markdown("Configure your proposed promotion below.")

    st.markdown("---")
    st.markdown("#### Product Details")

    category = st.selectbox(
        "Product Category",
        options=list(CATEGORY_PROFILES.keys()),
        index=1,  # Default to Specialty Food & Nuts
        help="Select the category that best fits your product"
    )

    product_name = st.text_input(
        "Product Name",
        value="Salted Pistachios 16oz",
        help="e.g., 'Organic Kombucha 12pk', 'Cheddar Cheese 8oz'"
    )

    regular_price = st.number_input(
        "Regular Price ($)",
        min_value=0.50, max_value=500.0, value=12.99, step=0.50,
        help="Current shelf price"
    )

    custom_margin = st.checkbox("Override category margin?", value=False)
    margin_pct = None
    if custom_margin:
        margin_pct = st.slider(
            "Product Margin %", 5, 70,
            int(CATEGORY_PROFILES[category]["base_margin"] * 100),
            help="Your actual margin on this product"
        ) / 100.0

    st.markdown("---")
    st.markdown("#### Promotion Details")

    promo_type = st.selectbox(
        "Promotion Type",
        options=list(PROMO_TYPES.keys()),
        help="Type of promotional mechanic"
    )

    discount_pct = st.slider(
        "Discount %", 5, 50, 25, step=5,
        help="Depth of discount off regular price"
    )

    duration_days = st.slider(
        "Duration (days)", 1, 28, 7,
        help="How long will the promotion run?"
    )

    st.markdown("---")
    st.markdown("#### Market Context")

    num_competing = st.slider(
        "Competing Products in Category", 0, 15, 4,
        help="How many similar products compete for the same shopper?"
    )

    is_peak = st.checkbox(
        "Peak Season?", value=False,
        help="Is this a high-demand period (holidays, summer, etc.)?"
    )

    has_display = st.checkbox(
        "In-Store Display / Feature Support?", value=True,
        help="End-cap, feature ad, or digital promotion?"
    )

    custom_baseline = st.checkbox("Override baseline volume?", value=False)
    weekly_units = None
    if custom_baseline:
        weekly_units = st.number_input(
            "Weekly Baseline Units", 10, 10000,
            CATEGORY_PROFILES[category]["typical_weekly_units"]
        )

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Promotion", use_container_width=True, type="primary")


# ── Summary generator (template-based, no API needed) ────────────────────────
def _generate_summary(result: PromotionResult, inp: PromotionInput) -> str:
    """Generate a natural-language executive summary."""
    promo_price = inp.regular_price * (1 - inp.discount_pct / 100)

    if "Run" in result.recommendation:
        opener = (
            f"**This promotion is recommended.** Running {inp.discount_pct}% off on "
            f"**{inp.product_name}** (${inp.regular_price:.2f} → ${promo_price:.2f}) "
            f"for {inp.duration_days} days is projected to be profitable."
        )
    elif "Caution" in result.recommendation:
        opener = (
            f"**This promotion warrants caution.** While {inp.discount_pct}% off on "
            f"**{inp.product_name}** will drive traffic, several factors limit its effectiveness."
        )
    else:
        opener = (
            f"**This promotion is not recommended.** At {inp.discount_pct}% off, "
            f"**{inp.product_name}** is projected to lose money on a net basis."
        )

    vol_insight = (
        f"We project a **{result.volume_lift_pct:.0f}% volume lift** "
        f"({result.baseline_units_per_week:,} → {result.projected_units_per_week:,} units/week). "
        f"However, approximately **{result.cannibalization_pct:.0f}%** of incremental volume "
        f"comes from cannibalization of adjacent products, leaving "
        f"**{result.true_incremental_units:,} true incremental units**."
    )

    if result.profit_change >= 0:
        profit_insight = (
            f"During the promotion window, profit increases by **${result.profit_change:,.0f}**. "
        )
    else:
        profit_insight = (
            f"The promotion reduces profit by **${abs(result.profit_change):,.0f}** during the promo period. "
        )

    if result.net_30day_profit_impact >= 0:
        profit_insight += (
            f"After factoring in the post-promotion sales dip ({result.post_promo_dip_pct:.0f}%), "
            f"the net 30-day impact remains positive at **${result.net_30day_profit_impact:,.0f}**."
        )
    else:
        profit_insight += (
            f"The post-promotion dip ({result.post_promo_dip_pct:.0f}%) further erodes returns, "
            f"bringing the 30-day net impact to **-${abs(result.net_30day_profit_impact):,.0f}**."
        )

    sens = result.sensitivity_data
    best_idx = max(range(len(sens["net_30day_profit"])),
                   key=lambda i: sens["net_30day_profit"][i])
    best_d = sens["discounts"][best_idx]
    if best_d != int(inp.discount_pct):
        sens_note = (
            f"💡 **Optimization insight:** Our sensitivity analysis suggests "
            f"**{best_d}% off** would be the profit-maximizing discount for this product "
            f"and market context."
        )
    else:
        sens_note = (
            f"Your proposed {inp.discount_pct}% discount is well-calibrated — it's at or near "
            f"the profit-optimal point for this product."
        )

    return f"{opener}\n\n{vol_insight}\n\n{profit_insight}\n\n{sens_note}"


# ── Main Analysis ────────────────────────────────────────────────────────────
if analyze_btn or "result" in st.session_state:
    if analyze_btn:
        inp = PromotionInput(
            product_name=product_name,
            category=category,
            regular_price=regular_price,
            discount_pct=discount_pct,
            promo_type=promo_type,
            duration_days=duration_days,
            num_competing_products=num_competing,
            is_peak_season=is_peak,
            has_display_support=has_display,
            weekly_baseline_units=weekly_units,
            product_margin_pct=margin_pct,
        )
        sim = PromotionSimulator()
        result = sim.analyze(inp)
        st.session_state["result"] = result
        st.session_state["inp"] = inp
    else:
        result = st.session_state["result"]
        inp = st.session_state["inp"]

    # ── Recommendation banner ────────────────────────────────────────────
    if "Run" in result.recommendation:
        css_class = "rec-run"
    elif "Caution" in result.recommendation:
        css_class = "rec-caution"
    else:
        css_class = "rec-stop"

    st.markdown(f"""
    <div class="rec-card {css_class}">
        <h2>{result.recommendation}</h2>
        <p><strong>Confidence:</strong> {result.confidence_score}% · {result.recommendation_reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key metrics row ──────────────────────────────────────────────────
    promo_price = inp.regular_price * (1 - inp.discount_pct / 100)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sign = "+" if result.volume_lift_pct > 0 else ""
        st.metric("Volume Lift", f"{sign}{result.volume_lift_pct}%",
                  f"{result.true_incremental_units:,} true incremental units")
    with col2:
        sign = "+" if result.profit_change > 0 else ""
        st.metric("Promo Profit Impact",
                  f"{sign}${result.profit_change:,.0f}",
                  f"ROI: {result.roi:.0f}%")
    with col3:
        st.metric("Net 30-Day Impact",
                  f"${result.net_30day_profit_impact:,.0f}",
                  f"Post-promo dip: {result.post_promo_dip_pct:.0f}%",
                  delta_color="normal")
    with col4:
        risk_label = "Low" if result.risk_score < 30 else "Medium" if result.risk_score < 60 else "High"
        st.metric("Risk Score", f"{result.risk_score:.0f}/100", risk_label,
                  delta_color="inverse")

    st.markdown("---")

    # ── Charts ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Volume Forecast", "💰 Profit Waterfall",
        "🔍 Sensitivity Analysis", "📊 Full Breakdown"
    ])

    with tab1:
        st.markdown('<div class="section-header">6-Week Volume Forecast</div>',
                    unsafe_allow_html=True)
        df_forecast = pd.DataFrame(result.weekly_volume_forecast)
        colors = {"Baseline": "#94a3b8", "Promotion": "#3b82f6", "Post-Promo": "#f59e0b"}
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_forecast["week"], y=df_forecast["units"],
            marker_color=[colors[p] for p in df_forecast["phase"]],
            text=df_forecast["units"],
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
        ))
        fig.add_hline(y=result.baseline_units_per_week, line_dash="dash",
                      line_color="#94a3b8", annotation_text="Baseline",
                      annotation_position="top right")
        fig.update_layout(
            height=400,
            plot_bgcolor="white",
            xaxis=dict(title="", tickfont=dict(size=12)),
            yaxis=dict(title="Units / Week", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"📌 Baseline: {result.baseline_units_per_week:,} units/wk · "
            f"During promo: {result.projected_units_per_week:,} units/wk · "
            f"Cannibalized: {result.cannibalized_units:,} units ({result.cannibalization_pct:.0f}%)"
        )

    with tab2:
        st.markdown('<div class="section-header">Profit Impact Waterfall</div>',
                    unsafe_allow_html=True)
        waterfall = result.profit_waterfall
        labels = list(waterfall.keys()) + ["Net Impact"]
        values = list(waterfall.values())
        net = sum(values)
        measures = ["absolute"] + ["relative"] * (len(values) - 1) + ["total"]
        values.append(net)

        fig2 = go.Figure(go.Waterfall(
            x=labels, y=values, measure=measures,
            connector={"line": {"color": "#e2e8f0"}},
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#3b82f6"}},
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            textfont=dict(size=11),
        ))
        fig2.update_layout(
            height=420,
            plot_bgcolor="white",
            yaxis=dict(title="Profit ($)", gridcolor="#f1f5f9"),
            xaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=60, r=20, t=30, b=60),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Discount Sensitivity — Net 30-Day Profit</div>',
                    unsafe_allow_html=True)
        sens = result.sensitivity_data
        df_sens = pd.DataFrame({
            "Discount %": [f"{d}%" for d in sens["discounts"]],
            "Net 30-Day Profit": sens["net_30day_profit"],
            "discount_num": sens["discounts"],
        })
        current_d = int(inp.discount_pct)
        colors_sens = ["#3b82f6" if d == current_d else "#cbd5e1" for d in sens["discounts"]]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_sens["Discount %"], y=df_sens["Net 30-Day Profit"],
            marker_color=colors_sens,
            text=[f"${v:,.0f}" for v in df_sens["Net 30-Day Profit"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig3.add_hline(y=0, line_color="#94a3b8", line_width=1)
        fig3.update_layout(
            height=400,
            plot_bgcolor="white",
            yaxis=dict(title="Net 30-Day Profit ($)", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=30, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Find optimal
        best_idx = max(range(len(sens["net_30day_profit"])),
                       key=lambda i: sens["net_30day_profit"][i])
        best_d = sens["discounts"][best_idx]
        best_p = sens["net_30day_profit"][best_idx]
        if best_d != current_d:
            st.info(f"💡 **Optimal discount: {best_d}%** → Net 30-day profit of ${best_p:,.0f}. "
                    f"Your proposed {current_d}% discount yields ${sens['net_30day_profit'][sens['discounts'].index(current_d)]:,.0f}.")
        else:
            st.success(f"✅ Your proposed {current_d}% discount is at or near the profit-optimal point.")

    with tab4:
        st.markdown('<div class="section-header">Complete Analysis</div>',
                    unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("##### 📦 Volume Analysis")
            vol_data = {
                "Metric": [
                    "Baseline Weekly Units",
                    "Projected Weekly Units",
                    "Volume Lift %",
                    "Total Promo Units",
                    "Gross Incremental Units",
                    "Cannibalized Units",
                    "True Incremental Units",
                    "Cannibalization Rate",
                ],
                "Value": [
                    f"{result.baseline_units_per_week:,}",
                    f"{result.projected_units_per_week:,}",
                    f"+{result.volume_lift_pct}%",
                    f"{result.total_promo_units:,}",
                    f"{result.incremental_units:,}",
                    f"-{result.cannibalized_units:,}",
                    f"{result.true_incremental_units:,}",
                    f"{result.cannibalization_pct:.1f}%",
                ],
            }
            st.dataframe(pd.DataFrame(vol_data), hide_index=True, use_container_width=True)

        with col_r:
            st.markdown("##### 💵 Financial Analysis")
            fin_data = {
                "Metric": [
                    "Regular Price",
                    "Promo Price",
                    "Baseline Revenue",
                    "Promo Revenue",
                    "Revenue Change",
                    "Baseline Profit",
                    "Promo Profit",
                    "Profit Change",
                    "Net 30-Day Impact",
                    "Pantry Loading Effect",
                    "Post-Promo Dip",
                ],
                "Value": [
                    f"${inp.regular_price:.2f}",
                    f"${promo_price:.2f}",
                    f"${result.regular_revenue:,.2f}",
                    f"${result.promo_revenue:,.2f}",
                    f"${result.revenue_change:,.2f}",
                    f"${result.regular_profit:,.2f}",
                    f"${result.promo_profit:,.2f}",
                    f"${result.profit_change:,.2f}",
                    f"${result.net_30day_profit_impact:,.2f}",
                    f"{result.pantry_loading_pct:.1f}%",
                    f"{result.post_promo_dip_pct:.1f}%",
                ],
            }
            st.dataframe(pd.DataFrame(fin_data), hide_index=True, use_container_width=True)

    # ── Risk & Mitigations ───────────────────────────────────────────────
    st.markdown("---")
    col_risk, col_mit = st.columns(2)

    with col_risk:
        st.markdown('<div class="section-header">⚠️ Risk Factors</div>',
                    unsafe_allow_html=True)
        if result.risk_factors:
            for factor, severity in result.risk_factors:
                dot_class = "risk-high" if severity >= 20 else "risk-med" if severity >= 10 else "risk-low"
                st.markdown(f"""
                <div class="risk-item">
                    <div class="risk-dot {dot_class}"></div>
                    <span>{factor} <em style="color:#94a3b8">(+{severity} risk pts)</em></span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant risk factors identified.")

    with col_mit:
        st.markdown('<div class="section-header">🛡️ Recommended Mitigations</div>',
                    unsafe_allow_html=True)
        for i, m in enumerate(result.mitigations, 1):
            st.markdown(f"**{i}.** {m}")

    # ── AI Summary (optional) ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Executive Summary</div>',
                unsafe_allow_html=True)

    summary = _generate_summary(result, inp)
    st.markdown(summary)


else:
    # ── Landing state ────────────────────────────────────────────────────
    st.markdown("### 👈 Configure a promotion in the sidebar and click **Analyze**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 📈 Volume Lift
        See projected sales increase based on 
        price elasticity and category dynamics.
        """)
    with col2:
        st.markdown("""
        #### 💰 Profit Impact
        Understand true profitability after 
        accounting for margin erosion and execution costs.
        """)
    with col3:
        st.markdown("""
        #### ⚠️ Risk Analysis
        Uncover hidden costs — cannibalization, 
        pantry loading, and post-promo dips.
        """)

    st.markdown("---")
    st.markdown("""
    **Price Sense AI** helps mid-market retailers ($50M–$500M revenue) make data-driven 
    promotion decisions. Stop relying on gut feel — know the true ROI before you run.
    
    *Try it now with the reference scenario: 25% off Salted Pistachios 16oz →*
    """)