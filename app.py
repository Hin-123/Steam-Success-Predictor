import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. IMPORT โมเดลทุกตัวที่อาจถูกเรียกใช้ตอน unpickle ---
# (ตามที่คุณถามว่าต้องนำเข้าตัวอื่นไหม: จำเป็นต้องใส่ไว้เพื่อป้องกัน AttributeError 
# หากในไฟล์ .pkl มีการอ้างอิงถึงโมเดลเหล่านี้จากขั้นตอนการเปรียบเทียบ)
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- 2. การโหลดโมเดลอย่างปลอดภัย ---
@st.cache_resource
def load_model():
    try:
        # โหลดไฟล์ pkl ที่สร้างจาก joblib.dump(model_data, ...)
        data = joblib.load('profit_prediction_model.pkl')
        # ตรวจสอบว่าเป็น Dictionary ตามที่เซฟไว้หรือไม่
        if isinstance(data, dict):
            return data['pipeline']
        return data
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        return None

model = load_model()

# --- 3. ส่วนการแสดงผล UI ---
st.set_page_config(page_title="Profit Predictor", page_icon="💰")
st.title("💰 ระบบพยากรณ์กำไร (Profit Predictor)")
st.write("เครื่องมือนี้ใช้โมเดล Gradient Boosting ที่ผ่านการจูนด้วย Optuna เพื่อคาดการณ์กำไร")

# --- 4. ส่วนรับข้อมูล Input (ให้ตรงกับ Columns ที่โมเดลต้องการ) ---
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

# --- 5. การเตรียมข้อมูลและทำนายผล ---
if model is not None:
    # สร้าง DataFrame ให้มีคอลัมน์เหมือนตอน Train เป๊ะๆ ตาม cat_cols และ num_cols
    input_df = pd.DataFrame([{
        'sales': sales,
        'quantity': quantity,
        'discount': discount,
        'shipping_cost': shipping_cost,
        'year': year,
        'market': market,
        # ค่า Default สำหรับฟีเจอร์อื่นๆ ที่โมเดลต้องการแต่ไม่ได้ให้กรอก
        'ship_mode': 'Standard Class',
        'segment': 'Consumer',
        'region': 'Southeast Asia',
        'category': 'Technology',
        'sub_category': 'Phones',
        'order_priority': 'Medium'
    }])

    if st.button("🚀 วิเคราะห์กำไร"):
        with st.spinner('กำลังคำนวณ...'):
            # ทำนายผลโดยใช้ Pipeline (ซึ่งจะทำ Scale และ OneHot ให้อัตโนมัติ)
            prediction = model.predict(input_df)[0]
            
            st.markdown("---")
            if prediction >= 0:
                st.balloons()
                st.success(f"### 📈 คาดการณ์กำไร: {prediction:,.2f} หน่วย")
            else:
                st.error(f"### 📉 คาดการณ์ขาดทุน: {prediction:,.2f} หน่วย")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** ผลการวิเคราะห์นี้เป็นเพียงการประมาณการทางสถิติ")
