import streamlit as st
import pandas as pd
import plotly.express as px
import os

def run():
    st.title("Delivery Time Prediction — Exploratory Data Analysis (EDA)")

    st.image(
        "https://www.apto.digital/au/wp-content/uploads/2023/09/Swiggy-Banner-1.webp",
        caption="Source: apto.digital (Swiggy Delivery Banner)"
    )

    st.markdown("""
    ## 🧭 Project Background  
    Efisiensi pengiriman merupakan faktor kunci dalam **operasional logistik dan kepuasan pelanggan**.
    """)

    # --- FIX PATH CSV ---
    file_path = os.path.join(os.path.dirname(__file__), "Food_Delivery_Times_Clean.csv")
    df = pd.read_csv(file_path)
    st.dataframe(df.head())
    st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ===============================
    # 1️⃣ Distribusi waktu pengantaran
    # ===============================
    st.subheader("1️⃣ Distribusi Waktu Pengantaran (`delivery_time_min`)")
    fig = px.histogram(df, x="delivery_time_min", nbins=30, color_discrete_sequence=["#FF7F50"])
    st.plotly_chart(fig, use_container_width=True)
    st.info("Mayoritas pengiriman selesai dalam 20–40 menit.")

    # ===============================
    # 2️⃣ Distribusi jarak & waktu persiapan
    # ===============================
    st.subheader("2️⃣ Distribusi `distance_km` dan `preparation_time_min`")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="distance_km", nbins=25, color_discrete_sequence=["#00BFFF"])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="preparation_time_min", nbins=25, color_discrete_sequence=["#8A2BE2"])
        st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # 3️⃣ Pengaruh cuaca
    # ===============================
    st.subheader("3️⃣ Pengaruh Cuaca terhadap Waktu Pengantaran")
    fig = px.box(df, x="weather", y="delivery_time_min", color="weather")
    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # 4️⃣ Pengaruh tingkat lalu lintas
    # ===============================
    st.subheader("4️⃣ Pengaruh Tingkat Lalu Lintas (`traffic_level`)")
    fig = px.box(df, x="traffic_level", y="delivery_time_min", color="traffic_level")
    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # 5️⃣ Perbedaan waktu dalam sehari
    # ===============================
    st.subheader("5️⃣ Waktu Pengantaran berdasarkan `time_of_day`")
    avg_tod = df.groupby("time_of_day")["delivery_time_min"].mean().reset_index()
    fig = px.bar(avg_tod, x="time_of_day", y="delivery_time_min", color="time_of_day")
    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # 6️⃣ Hubungan pengalaman kurir
    # ===============================
    st.subheader("6️⃣ Pengalaman Kurir terhadap Efisiensi Pengiriman")
    fig = px.scatter(df, x="courier_experience_yrs", y="delivery_time_min", trendline="ols",
                     color_discrete_sequence=["#32CD32"])
    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # 7️⃣ Korelasi antar fitur numerik
    # ===============================
    st.subheader("7️⃣ Korelasi Antar Fitur Numerik")
    num_cols = ["distance_km", "preparation_time_min", "courier_experience_yrs", "delivery_time_min"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # Ringkasan
    # ===============================
    st.header("📋 Ringkasan EDA")
    st.markdown("""  
    - Cuaca & traffic berpengaruh besar  
    - Pengalaman kurir mempercepat pengantaran  
    - Kombinasi variabel lingkungan + operasional penting untuk analisis prediksi
    """)

if __name__ == "__main__":
    run()


