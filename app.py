import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ❌ ลบการ Import XGBoost, LightGBM, CatBoost ออกไปได้เลย!
# ✔️ ระบบจะอ่าน Gradient Boosting ได้อัตโนมัติจากไฟล์ .pkl ผ่าน joblib

@st.cache_resource
def load_model():
    try:
        data = joblib.load('profit_prediction_model.pkl')
        if isinstance(data, dict):
            return data['pipeline']
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        return None

model = load_model()

# --- ส่วน UI สำหรับหน้าเว็บ (เหมือนเดิม) ---
st.set_page_config(page_title="Profit Predictor", page_icon="💰")
st.title("💰 ระบบพยากรณ์กำไรอัจฉริยะ")
st.write("โมเดลที่ดีที่สุดในโปรเจกต์นี้คือ Gradient Boosting ที่ผ่านการปรับแต่งไฮเปอร์พารามิเตอร์ด้วย Bayesian Optimization (Optuna)")

st.subheader("📊 กรอกรายละเอียดคำสั่งซื้อ")
col1, col2 = st.columns(2)

with col1:
    sales = st.number_input("ยอดขาย (Sales)", min_value=0.0, value=100.0)
    quantity = st.number_input("จำนวนสินค้า (Quantity)", min_value=1, value=1)
    discount = st.slider("ส่วนลด (Discount)", 0.0, 1.0, 0.0)

with col2:
    shipping_cost = st.number_input("ค่าขนส่ง (Shipping Cost)", min_value=0.0, value=10.0)
    year = st.selectbox("ปี (Year)", [2024, 2025, 2026])
    market = st.selectbox("ตลาด (Market)", ['APAC', 'EU', 'US', 'LATAM', 'Africa'])

if model is not None:
    input_df = pd.DataFrame([{
        'sales': sales, 'quantity': quantity, 'discount': discount, 'shipping_cost': shipping_cost,
        'year': year, 'market': market, 'ship_mode': 'Standard Class', 'segment': 'Consumer',
        'region': 'Southeast Asia', 'category': 'Technology', 'sub_category': 'Phones', 'order_priority': 'Medium'
    }])

    if st.button("🚀 วิเคราะห์กำไร"):
        with st.spinner('กำลังคำนวณ...'):
            prediction = model.predict(input_df)[0]
            st.markdown("---")
            if prediction >= 0:
                st.balloons()
                st.success(f"### 📈 คาดการณ์กำไร: {prediction:,.2f} ดอลลาร์")
            else:
                st.error(f"### 📉 คาดการณ์ขาดทุน: {prediction:,.2f} ดอลลาร์")
