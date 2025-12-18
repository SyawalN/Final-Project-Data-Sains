import zipfile
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import joblib

from functions.models.backpropagation import backpropagation
from functions.models.lstm import lstm
from functions.models.svm import svr

def render_ui_prediksi():
    st.title("Prediksi Penjualan")

    file = st.file_uploader("Unggah Dataset Dalam Format CSV")
    jumlah_hari = st.number_input("Masukkan Jumlah Hari", min_value=0, step=1, format="%d")

    if file is not None and jumlah_hari > 0:
        dataset = pd.read_csv(file)
        data = dataset["Total"].values
        tanggal = dataset["Tanggal"].values

        input, target = [], []

        for i in range(len(data) - jumlah_hari):
            input.append(np.array(data[i:i+jumlah_hari]))
            target.append(data[i+jumlah_hari])

        if "show_preview" not in st.session_state:
            st.session_state.show_preview = False

        def preview():
            st.session_state.show_preview = not st.session_state.show_preview

        label_preview = "Hide" if st.session_state.show_preview else "Preview"

        if st.button(label_preview, on_click=preview):
            pass

        if st.session_state.show_preview:
            baris = []

            for i in range(len(input)):
                row = {}
                row["Input"] = str(input[i])
                row["Target"] = target[i]
                baris.append(row)

            preview = pd.DataFrame(baris)
            st.write("Preview Data")
            st.dataframe(preview)

    methode = st.selectbox("Metode", ["=== Pilih Metode ===", "Backpropagation", "LSTM", "SVR (RBF)"]) 

    if methode != "=== Pilih Metode ===":
        option = st.selectbox("Mode", ["=== Pilih Mode ===", "Training", "Validasi/Testing"])

        if methode == "Backpropagation":
            if option == "Training":
                learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.1)
                steps = st.number_input("Steps", min_value=0, step=100, format="%d")

                if learning_rate > 0 and steps > 0:
                    if st.button("Mulai"):
                        model = backpropagation(learning_rate, jumlah_hari)
                        error = model.train(input, target, steps)

                        if error is not None:
                            fig, ax = plt.subplots()
                            ax.plot(error)
                            ax.set_title("Training Loss Setiap 100 Steps")
                            ax.set_xlabel("Steps")
                            ax.set_ylabel("Loss")

                            st.pyplot(fig)

                            st.write("Error Pertama: ", error[0])
                            st.write("Error Terakhir: ", error[-1])

                            buffer = BytesIO()
                            model.save_model(buffer)
                            buffer.seek(0)

                            st.download_button(
                                label="Download Model",
                                data=buffer,
                                file_name="model.npz",
                                mime="application/octet-stream"
                            )

            elif option == "Validasi/Testing":
                model = backpropagation()
                file = st.file_uploader("Unggah Model")
                scaler = st.file_uploader("Unggah Scaler")

                if jumlah_hari > 0:
                    tanggal = tanggal[jumlah_hari:] 

                if file is not None and scaler is not None:
                    model.load_model(file)
                    pred, scaler, rata_rata = model.validasi(input, target, scaler)

                    baris = []
                    array_target = np.array(target)
                    target_data = array_target.reshape(-1, 1)
                    actual_target = scaler.inverse_transform(target_data)
                    target_value = np.rint(actual_target).astype(int)

                    for i in range(len(input)):
                        row = {}
                        row["Tanggal"] = tanggal[i]
                        row["Prediksi"] = pred[i]
                        row["Target"] = target_value[i]
                        baris.append(row)

                    output = pd.DataFrame(baris)
                    st.write("Hasil Pengujian")
                    st.dataframe(output)

                    fig, ax = plt.subplots(figsize=(12,5))

                    ax.plot(pred, label="Prediksi", linewidth=1, color="blue")
                    ax.plot(target_value, label="Target", linewidth=1, color="red")

                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Nilai")
                    ax.set_title("Perbandingan Prediksi dan Target")
                    ax.legend(loc="upper left")

                    st.pyplot(fig)

                    st.write("Error Rata-Rata: ", rata_rata)
        elif methode == "LSTM":
            if option == "Training":
                learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.01)
                steps = st.number_input("Steps", min_value=0, step=100, format="%d")

                if learning_rate > 0 and steps > 0:
                    if st.button("Mulai"):
                        input_train = [np.array(data[i:i+jumlah_hari]).reshape(-1, 1) for i in range(len(data) - jumlah_hari)]
                        target_train = data[jumlah_hari:]

                        model = lstm(
                            input_size=1,
                            hidden_size=32,
                            output_size=1,
                            learning_rate=learning_rate
                            )

                        losses = []
                        for step in range(steps):
                            total_loss = 0.0

                            for i in range(len(input_train)):
                                input_seq = input_train[i]
                                target_train = target[i]

                                pred = model.forward(input_seq)

                                #mse
                                error = pred.item() - target_train
                                loss = 0.5 * error ** 2
                                total_loss += loss

                                dy = np.array([[error]])

                                model.backward(dy)

                            avg_loss = total_loss / len(input)
                            losses.append(avg_loss)

                        if losses:
                            fig, ax = plt.subplots()
                            ax.plot(losses)
                            ax.set_title("Training Loss")
                            ax.set_xlabel("Steps")
                            ax.set_ylabel("Loss")
                            st.pyplot(fig)
                            st.write("Error Pertama: ", losses[0])
                            st.write("Error Terakhir: ", losses[-1])

                            buffer = BytesIO()
                            model.save_model(buffer)
                            buffer.seek(0)

                            st.download_button(
                                label="Download Model",
                                data=buffer,
                                file_name="model.npz",
                                mime="application/octet-stream"
                            )
            
            elif option == "Validasi/Testing":
                model = lstm(input_size=1,
                            hidden_size=32,
                            output_size=1,
                            learning_rate=0.1)
                file = st.file_uploader("Unggah Model")
                scaler = st.file_uploader("Unggah Scaler")

                if jumlah_hari > 0 and "data" in locals() and len(data) > jumlah_hari:
                    input = [np.array(data[i:i+jumlah_hari]).reshape(-1, 1) for i in range(len(data) - jumlah_hari)]
                    target = data[jumlah_hari:]
                    tanggal = tanggal[jumlah_hari:]

                if file is not None and scaler is not None:
                    model.load_model(file)
                    scl = joblib.load(scaler)

                    predictions_normalized = []
                    total_loss = 0.0

                    for i in range(len(input)):
                        pred = model.forward(input[i])
                        predictions_normalized.append(pred.item())

                        #mse
                        error = pred.item() - target[i]
                        loss = 0.5 * error ** 2
                        total_loss += loss

                    total_loss /= len(input)

                    pred_array = np.array(predictions_normalized).reshape(-1, 1)
                    target_array = np.array(target).reshape(-1, 1)

                    pred_denorm = scl.inverse_transform(pred_array).flatten()
                    target_denorm = scl.inverse_transform(target_array).flatten()

                    target_rounded = np.rint(target_denorm).astype(int)

                    results = []
                    for i in range(len(pred_denorm)):
                        results.append({
                            "Tanggal": tanggal[i],
                            "Prediksi": pred_denorm[i],
                            "Target": target_rounded[i]
                        })

                    df_results = pd.DataFrame(results)
                    st.write("Hasil Pengujian")
                    st.dataframe(df_results)

                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(pred_denorm, label="Prediksi", color="blue", linewidth=1)
                    ax.plot(target_denorm, label="Target", color="red", linewidth=1)
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Nilai")
                    ax.set_title("Perbandingan Prediksi dan Target")
                    ax.legend(loc="upper left")
                    st.pyplot(fig)

                    st.write(f"Error Rata-Rata: ", total_loss)
        elif methode == "SVR (RBF)":
            st.subheader("Support Vector Regression (RBF)")
            st.info("SVR digunakan untuk prediksi nilai kontinu (time series)")

            if option == "Training":

                C = st.number_input("C (Regularization)", min_value=0.01, value=10.0)
                epsilon = st.number_input("Epsilon", min_value=0.0, value=0.1)
                gamma = st.number_input("Gamma (0 = auto)", min_value=0.0, value=0.0)
                max_iter = st.number_input("Max Iteration", min_value=100, value=1000)
                lr = st.number_input("Learning Rate", min_value=0.0001, value=0.01)

                if st.button("Mulai"):

                    # =========================
                    # FEATURE ENGINEERING (SVR)
                    # =========================
                    df = dataset.copy()
                    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='mixed', dayfirst=True)
                    df = df.sort_values('Tanggal').set_index('Tanggal')

                    df['Total'].replace(0, np.nan, inplace=True)
                    df['Total'].fillna(method='ffill', inplace=True)

                    # Lag features
                    lags = [1, 2, 3, 7]
                    for lag in lags:
                        df[f'lag_{lag}'] = df['Total'].shift(lag)

                    # Calendar features
                    df['day_of_week'] = df.index.dayofweek
                    df['month'] = df.index.month

                    df = df.dropna()

                    X = df.drop(columns=['Total']).values
                    y = df['Total'].values
                    tanggal = df.index

                    # =========================
                    # Scaling
                    # =========================
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # =========================
                    # Train SVR (FULL DATA)
                    # =========================
                    model = svr(
                        C=C,
                        epsilon=epsilon,
                        gamma=None if gamma == 0 else gamma,
                        max_iter=max_iter,
                        lr=lr
                    )

                    with st.spinner("Training SVR (Full Dataset)..."):
                        model.fit(X_scaled, y)

                    losses = model.loss_history

                    fig, ax = plt.subplots()
                    ax.plot(losses)
                    ax.set_title("SVR Training Loss (MSE)")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss")

                    st.pyplot(fig)

                    st.write("Loss Awal :", losses[0])
                    st.write("Loss Akhir:", losses[-1])

                    # =========================
                    # Save model & scaler
                    # =========================
                    buffer_model = BytesIO()
                    buffer_scaler = BytesIO()

                    model.save(buffer_model)
                    buffer_model.seek(0)

                    joblib.dump(scaler, buffer_scaler)
                    buffer_scaler.seek(0)

                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as z:
                        z.writestr("svr_model.pkl", buffer_model.getvalue())
                        z.writestr("scaler.pkl", buffer_scaler.getvalue())

                    zip_buffer.seek(0)

                    st.download_button(
                        label="Download Model SVR",
                        data=zip_buffer,
                        file_name="svr_model.zip",
                        mime="application/zip"
                    )
            
            elif option == "Validasi/Testing":

                model_file = st.file_uploader("Unggah Model SVR")
                scaler_file = st.file_uploader("Unggah Scaler")

                if model_file and scaler_file:

                    # =========================
                    # Load model & scaler
                    # =========================
                    model = svr()
                    model.load(model_file)

                    scaler = joblib.load(scaler_file)

                    # =========================
                    # SAME FEATURE ENGINEERING AS TRAINING
                    # =========================
                    df = dataset.copy()
                    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='mixed', dayfirst=True)
                    df = df.sort_values('Tanggal').set_index('Tanggal')

                    df['Total'].replace(0, np.nan, inplace=True)
                    df['Total'].fillna(method='ffill', inplace=True)

                    # Lag features (HARUS SAMA)
                    lags = [1, 2, 3, 7]
                    for lag in lags:
                        df[f'lag_{lag}'] = df['Total'].shift(lag)

                    # Calendar features (HARUS SAMA)
                    df['day_of_week'] = df.index.dayofweek
                    df['month'] = df.index.month

                    df = df.dropna()

                    X = df.drop(columns=['Total']).values
                    y = df['Total'].values
                    tanggal = df.index

                    # =========================
                    # Scaling (transform only!)
                    # =========================
                    X_scaled = scaler.transform(X)

                    # =========================
                    # Prediction
                    # =========================
                    y_pred = model.predict(X_scaled)

                    mse = mean_squared_error(y, y_pred)

                    st.success(f"MSE: {mse}")

                    # =========================
                    # Visualization
                    # =========================
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(tanggal, y, label="Target", linewidth=2)
                    ax.plot(tanggal, y_pred, label="Prediksi", linestyle="--")

                    ax.set_title("SVR (RBF) - Target vs Prediksi (Validation)")
                    ax.set_xlabel("Tanggal")
                    ax.set_ylabel("Penjualan")
                    ax.legend()

                    st.pyplot(fig)

                    # =========================
                    # Result Table
                    # =========================
                    hasil = pd.DataFrame({
                        "Tanggal": tanggal,
                        "Prediksi": y_pred,
                        "Target": y
                    })

                    st.write("Hasil Prediksi")
                    st.dataframe(hasil)

