import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("lasso_model.pkl")

# Load dataset
df = pd.read_csv("Laptop_price.csv")

# Judul Dashboard
st.title("💻 Laptop Price Prediction")

# Form Input Spesifikasi Laptop
st.header("🛠️ Input Laptop Specifications")
processor_speed = st.slider("Processor Speed (GHz)", 1.0, 4.0, 2.5)
ram_size = st.slider("RAM Size (GB)", 4, 64, 8)
storage_capacity = st.slider("Storage Capacity (GB)", 128, 2000, 512)
screen_size = st.slider("Screen Size (inch)", 10.0, 17.0, 15.6)
weight = st.slider("Weight (kg)", 1.0, 5.0, 2.3)
brand = st.selectbox("Brand", df["Brand"].unique())

# Prediksi Harga
input_data = pd.DataFrame([{
    "Brand": brand,
    "Processor_Speed": processor_speed,
    "RAM_Size": ram_size,
    "Storage_Capacity": storage_capacity,
    "Screen_Size": screen_size,
    "Weight": weight
}])

predicted_price = model.predict(input_data)[0]
st.subheader(f"💰 Predicted Price: Rp {int(predicted_price):,}")

# Segmentasi Berdasarkan Harga
if predicted_price <= 12000:
    segment = "Entry-level"
elif predicted_price <= 19000:
    segment = "Mid-range"
else:
    segment = "High-end"

st.markdown(f"📊 Market Segment : **{segment}**")

# Histogram Distribusi Harga
st.header("📈 Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Price"], bins=30, kde=True, color="salmon", edgecolor="black", ax=ax)
ax.axvline(predicted_price, color="blue", linestyle="--", label="Predicted Price")
ax.legend()
st.pyplot(fig)

# Sidebar Profil
st.sidebar.header("Profile")
st.sidebar.markdown("**Name :** Elizabeth Meliani")
st.sidebar.markdown("**Email :** melzyunho@gmail.com")
st.sidebar.markdown("**Bio :** Data Scientist Learner")
