import streamlit as st
import pandas as pd
import joblib

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Steam Game Success Predictor", layout="centered")

# --- โหลดโมเดลที่บันทึกไว้ ---
@st.cache_resource
def load_model():
    # ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับที่คุณบันทึกไว้
    return joblib.load('best_steam_model_gradientboosting.pkl')

model = load_model()

# --- ส่วนหัวของแอป ---
st.title("🎮 Steam Game Owners Predictor")
st.markdown("กรอกข้อมูลรายละเอียดเกมเพื่อพยากรณ์ยอดผู้ซื้อ (Owners)")
st.divider()

# --- ส่วนรับข้อมูลจากผู้ใช้ (Input Section) ---
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("ราคาเกม (USD)", min_value=0.0, value=19.99, step=0.99)
    ccu = st.number_input("ยอดผู้เล่นพร้อมกันสูงสุด (Peak CCU)", min_value=0, value=500)

with col2:
    positive = st.number_input("จำนวนรีวิวบวก (Positive)", min_value=0, value=1000)
    negative = st.number_input("จำนวนรีวิวลบ (Negative)", min_value=0, value=100)

developer = st.text_input("ชื่อผู้พัฒนา (Developer)", value="Unknown")

# --- ส่วนการพยากรณ์ (Prediction) ---
if st.button("Predict Owners", type="primary"):
    # สร้าง DataFrame สำหรับ Input ให้ตรงกับที่โมเดลต้องการ
    input_data = pd.DataFrame({
        'price': [price],
        'ccu': [ccu],
        'positive': [positive],
        'negative': [negative],
        'developer': [developer]
    })
    
    # พยากรณ์ผล
    prediction = model.predict(input_data)[0]
    
    # แสดงผลลัพธ์
    st.divider()
    st.subheader("ผลการวิเคราะห์:")
    
    # ปรับแต่งการแสดงผลตัวเลข (ให้ไม่มีทศนิยมและมีคอมม่า)
    predicted_owners = max(0, int(prediction)) # ป้องกันค่าติดลบถ้าโมเดลเพี้ยน
    
    st.metric(label="ประมาณการยอดผู้ซื้อ (Estimated Owners)", value=f"{predicted_owners:,} คน")
    
    if predicted_owners > 100000:
        st.success("🌟 เกมนี้มีแนวโน้มที่จะเป็นเกมยอดฮิต (Hit Game)!")
    elif predicted_owners > 10000:
        st.info("📈 เกมนี้อยู่ในระดับมาตรฐานที่น่าพอใจ")
    else:
        st.warning("⚠️ แนวโน้มยอดขายอาจจะไม่สูงนัก ลองปรับกลยุทธ์การตลาดดูนะ")

# --- ส่วนท้าย ---
st.caption("หมายเหตุ: ข้อมูลนี้เป็นการพยากรณ์จากโมเดล Machine Learning โดยอ้างอิงจากข้อมูลในอดีตเท่านั้น")
