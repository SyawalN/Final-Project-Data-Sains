import streamlit as st
from functions.ui.menu_klasifikasi import render_ui_klasifikasi
from functions.ui.menu_prediksi import render_ui_prediksi

def main():
    studi_kasus = st.selectbox(
        "Studi Kasus",
        ["=== Pilih Studi Kasus ===", "Prediksi", "Klasifikasi"]
    )

    if studi_kasus == "Prediksi":
        render_ui_prediksi()
    
    elif studi_kasus == "Klasifikasi":
        render_ui_klasifikasi()

if __name__ == "__main__":
    main()