import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Smart F&B Analytics Simulator", layout="wide")

st.title("🍽️ Smart F&B Inventory & Profit Simulator (臺灣餐飲智慧分析)")
st.markdown("Interactive Machine Learning tool to optimize daily prep stock, reduce food waste, and forecast profitability across restaurant tiers.")

# 1. Fungsi Machine Learning Dinamis Sesuai Segmen Restoran (100% Gratis)
@st.cache_resource
def train_demand_model(min_p, max_p, base_demand):
    np.random.seed(42)
    n_days = 350
    price = np.random.uniform(min_p, max_p, n_days)
    is_weekend = np.random.choice([0, 1], n_days, p=[0.7, 0.3])
    weather = np.random.choice([0, 1, 2], n_days, p=[0.25, 0.45, 0.30])
    promo_active = np.random.choice([0, 1], n_days, p=[0.6, 0.4])

    mid_p = (min_p + max_p) / 2
    elasticity = (base_demand * 0.45) / (max_p - min_p)

    demand = (
        base_demand
        - elasticity * (price - mid_p)
        + (base_demand * 0.25) * is_weekend
        + (base_demand * 0.12) * weather
        + (base_demand * 0.18) * promo_active
        + np.random.normal(0, base_demand * 0.05, n_days)
    )
    demand = np.maximum(demand, 8).astype(int)

    X = pd.DataFrame({
        "price": price,
        "is_weekend": is_weekend,
        "weather": weather,
        "promo_active": promo_active
    })
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, demand)
    return model

tab1, tab2 = st.tabs([
    "📊 ML Demand & Stock Simulator",
    "🏆 Case Study: Rijsttafel Java (103 Orders)"
])

with tab1:
    st.subheader("Interactive Scenario Planner (Multi-Tier Restaurant Model)")
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.markdown("**1. Select Restaurant Concept**")
        concept = st.selectbox(
            "Market Segment",
            [
                "Casual Dining / Pop-Up Set (NT$ 150 – 800)",
                "Upscale Bistro / Family Dining (NT$ 400 – 1,800)",
                "Fine Dining / Tasting Menu (NT$ 1,200 – 5,000)"
            ]
        )

        # Pengaturan otomatis batas harga & porsi berdasarkan kategori restoran
        if "Casual" in concept:
            min_p, max_p, def_p, step_p = 150, 800, 500, 25
            min_c, max_c, def_c, step_c = 50, 450, 200, 15
            min_s, max_s, def_s = 30, 220, 110
            base_demand, promo_fee = 115, 500
        elif "Upscale" in concept:
            min_p, max_p, def_p, step_p = 400, 1800, 750, 50
            min_c, max_c, def_c, step_c = 150, 900, 300, 25
            min_s, max_s, def_s = 25, 160, 85
            base_demand, promo_fee = 85, 1200
        else:
            min_p, max_p, def_p, step_p = 1200, 5000, 2500, 100
            min_c, max_c, def_c, step_c = 400, 2500, 950, 50
            min_s, max_s, def_s = 10, 90, 45
            base_demand, promo_fee = 45, 3000

        st.markdown("**2. Operational Parameters**")
        menu_price = st.slider("Menu Selling Price (NT$)", min_p, max_p, def_p, step=step_p)
        food_cost = st.slider("Ingredient Cost per Portion (NT$)", min_c, max_c, def_c, step=step_c)
        prep_stock = st.slider("Planned Daily Prep Stock (Portions)", min_s, max_s, def_s, step=5)
        day_type = st.selectbox("Day Type", ["Weekday (平日)", "Weekend (假日)"])
        weather_cond = st.selectbox("Local Weather Forecast", ["Heavy Rain (大雨)", "Cloudy (多雲)", "Sunny (晴天)"])
        promo = st.checkbox(f"Run Targeted Marketing Promo (+NT$ {promo_fee:,} Daily Cost)")

    model = train_demand_model(min_p, max_p, base_demand)

    weekend_val = 1 if "Weekend" in day_type else 0
    weather_map = {"Heavy Rain (大雨)": 0, "Cloudy (多雲)": 1, "Sunny (晴天)": 2}
    weather_val = weather_map[weather_cond]
    promo_val = 1 if promo else 0

    input_df = pd.DataFrame([{
        "price": menu_price,
        "is_weekend": weekend_val,
        "weather": weather_val,
        "promo_active": promo_val
    }])

    predicted_demand = int(round(model.predict(input_df)[0]))
    recommended_stock = int(round(predicted_demand * 1.05))
    actual_sold = min(prep_stock, predicted_demand)
    unmet_demand = max(0, predicted_demand - prep_stock)
    unsold_waste = max(0, prep_stock - predicted_demand)

    revenue = actual_sold * menu_price
    total_cost = (prep_stock * food_cost) + (promo_fee if promo else 0)
    net_profit = revenue - total_cost

    with col_output:
        st.markdown("**AI Forecast & Financial Impact**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Demand", f"{predicted_demand} portions")
        m2.metric("Optimal Prep Stock", f"{recommended_stock} portions")
        m3.metric("Estimated Waste", f"{unsold_waste} portions")
        m4.metric("Projected Net Profit", f"NT$ {net_profit:,}")

        st.markdown("**Smart Operational Recommendation:**")
        if unsold_waste > 10:
            st.warning(
                f"⚠️ Overstock Alert: Reducing prep stock from {prep_stock} to {recommended_stock} portions "
                f"saves NT$ {(prep_stock - recommended_stock) * food_cost:,} in potential ingredient waste."
            )
        elif unmet_demand > 8:
            st.info(
                f"📈 Opportunity Alert: High demand expected. Increasing prep stock by {unmet_demand} portions "
                f"could capture an extra NT$ {unmet_demand * (menu_price - food_cost):,} in margin."
            )
        else:
            st.success("✅ Optimal Balance: Your planned stock closely matches AI-predicted customer demand.")

        prices_range = np.linspace(min_p, max_p, 15).astype(int)
        sim_rows = []
        for p in prices_range:
            d = int(round(model.predict(pd.DataFrame([{
                "price": p, "is_weekend": weekend_val, "weather": weather_val, "promo_active": promo_val
            }]))[0]))
            profit_est = (min(prep_stock, d) * p) - (prep_stock * food_cost) - (promo_fee if promo else 0)
            sim_rows.append({"Price (NT$)": p, "Estimated Net Profit (NT$)": profit_est})
        st.line_chart(pd.DataFrame(sim_rows).set_index("Price (NT$)"))

with tab2:
    st.subheader("Capstone Pop-Up Restaurant Analysis: Rijsttafel Java")
    st.markdown(
        "Empirical operational data from managing a **33-member cross-functional team** "
        "(Kitchen, Service, Secretariat, Design) serving **103 pre-order customers**."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Pre-Orders Fulfilled", "103 Sets", "100% On-Time")
    c2.metric("Menu Price per Set", "NT$ 500", "Indonesian Rijsttafel")
    c3.metric("Total Gross Revenue", "NT$ 51,500", "Pre-Order Model")
    c4.metric("Team Members Managed", "33 Staff", "4 Divisions")

    div_data = pd.DataFrame({
        "Division": ["Kitchen Production", "Service & Front-of-House", "Secretariat & Finance", "Visual & Menu Design"],
        "Allocated Staff": [14, 10, 5, 4]
    }).set_index("Division")
    st.bar_chart(div_data)
