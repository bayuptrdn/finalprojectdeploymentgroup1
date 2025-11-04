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

    # --- FIX FILE PATH ---
    file_path = os.path.join(os.path.dirname(__file__), "Food_Delivery_Times_Clean.csv")
    df = pd.read_csv(file_path)
    st.dataframe(df.head())
    st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # --- VISUALISASI ---
    st.subheader("Distribusi Waktu Pengantaran (`delivery_time_min`)")
    fig = px.histogram(df, x="delivery_time_min", nbins=30, color_discrete_sequence=["#FF7F50"])
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run()

