import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Smart F&B Analytics Simulator", layout="wide")

st.title("🍽️ Smart F&B Inventory & Profit Simulator (臺灣餐飲智慧分析)")
st.markdown("Interactive Machine Learning tool to optimize daily prep stock, reduce food waste, and forecast profitability for SME restaurants.")

# 1. Buat Data Simulasi Historis & Latih Model ML di Memori (100% Gratis)
@st.cache_resource
def train_demand_model():
    np.random.seed(42)
    n_days = 300
    price = np.random.uniform(120, 300, n_days)
    is_weekend = np.random.choice([0, 1], n_days, p=[0.7, 0.3])
    # Cuaca: 0 = Hujan Deras/Taifun, 1 = Berawan, 2 = Cerah
    weather = np.random.choice([0, 1, 2], n_days, p=[0.25, 0.45, 0.30])
    promo_active = np.random.choice([0, 1], n_days, p=[0.6, 0.4])

    # Rumus dasar permintaan harian + variasi acak
    demand = (
        160
        - 0.32 * price
        + 38 * is_weekend
        + 22 * weather
        + 28 * promo_active
        + np.random.normal(0, 8, n_days)
    )
    demand = np.maximum(demand, 20).astype(int)

    X = pd.DataFrame({
        "price": price,
        "is_weekend": is_weekend,
        "weather": weather,
        "promo_active": promo_active
    })
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, demand)
    return model

model = train_demand_model()

tab1, tab2 = st.tabs([
    "📊 ML Demand & Stock Simulator",
    "🏆 Case Study: Rijsttafel Java (103 Orders)"
])

with tab1:
    st.subheader("Interactive Scenario Planner (Zero Data Upload Required)")
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.markdown("**Operational Parameters**")
        menu_price = st.slider("Menu Selling Price (NT$)", 120, 300, 180, step=10)
        food_cost = st.slider("Ingredient Cost per Portion (NT$)", 40, 150, 75, step=5)
        prep_stock = st.slider("Planned Daily Prep Stock (Portions)", 50, 220, 130, step=5)
        day_type = st.selectbox("Day Type", ["Weekday (平日)", "Weekend (假日)"])
        weather_cond = st.selectbox("Local Weather Forecast", ["Heavy Rain (大雨)", "Cloudy (多雲)", "Sunny (晴天)"])
        promo = st.checkbox("Run Social Media Promo (+NT$ 500 Fixed Cost)")

    # Konversi input ke format model
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
    recommended_stock = int(round(predicted_demand * 1.05))  # 5% safety buffer
    actual_sold = min(prep_stock, predicted_demand)
    unmet_demand = max(0, predicted_demand - prep_stock)
    unsold_waste = max(0, prep_stock - predicted_demand)

    revenue = actual_sold * menu_price
    total_cost = (prep_stock * food_cost) + (500 if promo else 0)
    net_profit = revenue - total_cost

    with col_output:
        st.markdown("**AI Forecast & Financial Impact**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Demand", f"{predicted_demand} portions")
        m2.metric("Optimal Prep Stock", f"{recommended_stock} portions")
        m3.metric("Estimated Waste", f"{unsold_waste} portions")
        m4.metric("Projected Net Profit", f"NT$ {net_profit:,}")

        st.markdown("**Smart Operational Recommendation:**")
        if unsold_waste > 15:
            st.warning(
                f"⚠️ Overstock Alert: Reducing prep stock from {prep_stock} to {recommended_stock} portions "
                f"saves NT$ {(prep_stock - recommended_stock) * food_cost:,} in potential food waste."
            )
        elif unmet_demand > 10:
            st.info(
                f"📈 Opportunity Alert: High traffic expected. Increasing prep stock by {unmet_demand} portions "
                f"could capture an extra NT$ {unmet_demand * (menu_price - food_cost):,} in margin."
            )
        else:
            st.success("✅ Optimal Balance: Your planned stock closely matches AI-predicted customer traffic.")

        # Simulasi grafik perbandingan harga vs profit
        prices_range = np.arange(120, 305, 15)
        sim_rows = []
        for p in prices_range:
            d = int(round(model.predict(pd.DataFrame([{
                "price": p, "is_weekend": weekend_val, "weather": weather_val, "promo_active": promo_val
            }]))[0]))
            profit_est = (d * (p - food_cost)) - (500 if promo else 0)
            sim_rows.append({"Price (NT$)": p, "Estimated Profit (NT$)": profit_est})
        st.line_chart(pd.DataFrame(sim_rows).set_index("Price (NT$)"))

with tab2:
    st.subheader("Capstone Pop-Up Restaurant Analysis: Rijsttafel Java")
    st.markdown(
        "Empirical operational data from managing a **33-member cross-functional team** "
        "(Kitchen, Service, Secretariat, Design) serving **103 pre-order customers**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pre-Orders Fulfilled", "103 Sets", "100% On-Time")
    c2.metric("Team Members Managed", "33 Staff", "4 Divisions")
    c3.metric("Inventory Waste Rate", "< 3.2%", "Pre-Order Model")

    div_data = pd.DataFrame({
        "Division": ["Kitchen Production", "Service & Front-of-House", "Secretariat & Finance", "Visual & Menu Design"],
        "Allocated Staff": [14, 10, 5, 4]
    }).set_index("Division")
    st.bar_chart(div_data)
