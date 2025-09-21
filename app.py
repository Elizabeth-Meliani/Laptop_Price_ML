import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("lasso_model.pkl")

# Load dataset
df = pd.read_csv("Laptop_price.csv")

# Fungsi untuk memfilter data berdasarkan segmen
def filter_segment(dataframe, segment_option):
    if segment_option == "Entry-level":
        return dataframe[(dataframe["Price"] >= 8000) & (dataframe["Price"] <= 12000)]
    elif segment_option == "Mid-range":
        return dataframe[(dataframe["Price"] >= 16000) & (dataframe["Price"] <= 19000)]
    elif segment_option == "High-end":
        return dataframe[(dataframe["Price"] >= 30000) & (dataframe["Price"] <= 34000)]
    else:
        return dataframe # Semua data

# Judul Dashboard
st.title("💻 Laptop Price Prediction App")

with st.form("Laptop Specs"):
    st.header("🛠️ Input Laptop Specifications")

    processor_speed = st.slider("Processor Speed (GHz)", 1.0, 4.0, 2.5)
    ram_size = st.slider("RAM Size (GB)", 4, 64, 8)
    storage_capacity = st.slider("Storage Capacity (GB)", 128, 2000, 512)
    screen_size = st.slider("Screen Size (inch)", 10.0, 17.0, 15.6)
    weight = st.slider("Weight (kg)", 1.0, 5.0, 2.3)
    brand = st.selectbox("Brand", df["Brand"].unique())

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Brand": brand,
            "Processor_Speed": processor_speed,
            "RAM_Size": ram_size,
            "Storage_Capacity": storage_capacity,
            "Screen_Size": screen_size,
            "Weight": weight
        }])


# Prediksi Harga
st.success(f"Predicted Price: Rp {predicted_price:,.0f}")
st.caption("Prediksi dilakukan dengan model Lasso Regression.")

# Segmentasi Berdasarkan Prediksi Harga
if 8000 <= predicted_price <= 12000:
    segment = "Entry-level"
elif 16000 <= predicted_price <= 19000:
    segment = "Mid-range"
elif 30000 <= predicted_price <= 34000:
    segment = "High-end"
else:
    segment = "Out of defined range"

# Eksplorasi Segmen Pasar (Menggunakan fungsi filter_segment)
st.header("🔍 Explore Market Segments")
segment_option_data = st.selectbox("Pilih Segmen Harga untuk Dataframe", ["Entry-level", "Mid-range", "High-end"])
filtered_df = filter_segment(df, segment_option_data)

# Tampilkan Data Segmen
st.subheader(f"📊 Segment: {segment_option_data}")
st.dataframe(filtered_df)

st.header("📈 Price Distribution by Segment")

# Pilihan Segmen untuk Histogram (Menggunakan fungsi filter_segment)
segment_option_plot = st.selectbox("Pilih Segmen Harga untuk Visualisasi", ["All", "Entry-level", "Mid-range", "High-end"])
plot_df = filter_segment(df, segment_option_plot)

# Plot Histogram
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(plot_df["Price"], bins=30, kde=False, color="salmon", edgecolor="black", ax=ax)

ax.axvline(predicted_price, color="blue", linestyle="--", linewidth=2)
ax.set_title(f"Distribusi Harga - {segment_option_plot}", fontsize=14)
ax.set_xlabel("Harga (dalam ribu rupiah)", fontsize=12)
ax.set_ylabel("Jumlah Unit", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()

st.pyplot(fig)

# Sidebar Profil
st.sidebar.header("Profile")
st.sidebar.markdown("**Name :** Elizabeth Meliani")
st.sidebar.markdown("**Email :** melzyunho@gmail.com")
st.sidebar.markdown("**Bio :** Data Scientist Learner")

# Link GitHub
st.sidebar.markdown("📂 **GitHub :** [Laptop_Price_ML](https://github.com/Elizabeth-Meliani/Laptop_Price_ML)")
