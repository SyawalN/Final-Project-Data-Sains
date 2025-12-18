from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

from functions.models.backpropagation import backpropagation_klasifikasi
from functions.models.lstm import lstm_klasifikasi
from functions.models.svm import svm_rbf_klasifikasi

def render_ui_klasifikasi():
    st.title("Klasifikasi Sentimen")

    file = st.file_uploader("Unggah Dataset Dalam Format CSV")
    if file is not None:
        delimiter = st.selectbox(
            "Format Pemisah Kolom Data (Delimiter):",
            [",", ";"]
        )
        dataset = pd.read_csv(file, sep=delimiter)

        if "show_preview" not in st.session_state:
            st.session_state.show_preview = False

        def preview():
            st.session_state.show_preview = not st.session_state.show_preview

        label_preview = "Hide" if st.session_state.show_preview else "Preview"

        if st.button(label_preview, on_click=preview):
            pass

        if st.session_state.show_preview:
            baris = []

            for label, text in dataset[["Sentiment", "Instagram Comment Text"]].values:
                row = {}
                row["Teks"] = text
                row["Sentimen"] = label
                baris.append(row)

            preview = pd.DataFrame(baris)
            st.write("Preview Data")
            st.dataframe(preview)

    methode = st.selectbox("Metode", ["=== Pilih Metode ===", "Backpropagation", "LSTM", "SVM (RBF)"]) 

    if methode != "=== Pilih Metode ===":
        option = st.selectbox("Mode", ["=== Pilih Mode ===", "Training", "Validasi/Testing"])

        if methode == "Backpropagation":
            all_words = []
            labels = []
            texts = []

            for label, text in dataset[["Sentiment", "Instagram Comment Text"]].values:
                words = text.lower().split()
                all_words.extend(words)
                texts.append(words)
                labels.append(label)

            vocab = {}
            for word in sorted(set(all_words)):
                vocab[word] = len(vocab)

            vocab_size = len(vocab)

            #konversi teks ke binary bag of words
            def text_to_bow(words, vocab):
                vec = np.zeros(len(vocab))
                for word in words:
                    if word in vocab:
                        vec[vocab[word]] = 1
                return vec

            input_vectors = []
            for words in texts:
                vec = text_to_bow(words, vocab)
                input_vectors.append(vec)

            input = np.array(input_vectors)
            target = np.array(labels, dtype=float)

            if option == "Training":
                learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.1)
                steps = st.number_input("Steps", min_value=0, step=100, format="%d")

                model = backpropagation_klasifikasi(learning_rate, vocab_size)

                if learning_rate > 0 and steps > 0:
                    if st.button("Mulai"):
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
                            vocab_buffer = BytesIO()

                            joblib.dump(vocab, vocab_buffer)
                            vocab_buffer.seek(0)

                            model.save_model(buffer)
                            buffer.seek(0)

                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w") as z:
                                z.writestr("vocab.pkl", vocab_buffer.getvalue())
                                z.writestr("model.npz", buffer.getvalue())

                            zip_buffer.seek(0)

                            st.download_button(
                                label="Download Model",
                                data=zip_buffer,
                                file_name="model.zip",
                                mime="application/zip"
                                )
            elif option == "Validasi/Testing":
                file = st.file_uploader("Unggah Model")
                vocab_file = st.file_uploader("Unggah vocab")

                if file is not None and vocab_file is not None:
                    vocab = joblib.load(vocab_file)
                    vocab_size = len(vocab)

                    #konversi teks ke binary bag of words
                    def text_to_bow(words, vocab):
                        vec = np.zeros(len(vocab))
                        for word in words:
                            if word in vocab:
                                vec[vocab[word]] = 1
                        return vec

                    input_vectors = []
                    for words in texts:
                        vec = text_to_bow(words, vocab)
                        input_vectors.append(vec)

                    input = np.array(input_vectors)
                    target = np.array(labels, dtype=float)

                    model = backpropagation_klasifikasi(input=vocab_size)
                    model.load_model(file)
                    output, akurasi, rata_rata = model.validasi(input, target)

                    baris = []

                    for i in range(len(input)):
                        row = {}
                        row["Prediksi"] = output[i]
                        row["Target"] = target[i]
                        baris.append(row)

                    output_df = pd.DataFrame(baris)
                    st.write("Hasil Pengujian")
                    st.dataframe(output_df)

                    st.write(f"Akurasi: {akurasi:.2%}")

                    cm = confusion_matrix(target, output)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Aktual")
                    st.pyplot(fig)

                    st.write(f"Error Rata-Rata: ", rata_rata)
        elif methode == "LSTM":
            all_words = []
            labels = []
            texts = []

            for label, text in dataset[["Sentiment", "Instagram Comment Text"]].values:
                words = text.lower().split()
                all_words.extend(words)
                texts.append(words)
                labels.append(label)

            vocab = {"<PAD>": 0, "<UNK>": 1}
            for i in sorted(set(all_words)):
                vocab[i] = len(vocab)

            #maxlen = sequence length
            def text_to_seq(words, maxlen=20):
                seq = [vocab.get(w, vocab["<UNK>"]) for w in words]

                if len(seq) < maxlen: #tambah padding ke text
                    seq = [vocab["<PAD>"]] * (maxlen - len(seq)) + seq
                else:
                    seq = seq[-maxlen:]  #simpan text maxlen terakhir
                return seq

            input = np.array([text_to_seq(words) for words in texts])
            target = np.array(labels).reshape(-1, 1)  # shape: (N, 1)

            if option == "Training":
                learning_rate = st.number_input("Learning Rate", min_value=0.0, max_value=1.0, step=0.01)
                steps = st.number_input("Steps", min_value=0, step=100, format="%d")

                if learning_rate > 0 and steps > 0:
                    if st.button("Mulai"):
                        vocab_size = len(vocab)
                        dimensi_embedding = 32
                        hidden_size = 32
                        output_size = 1

                        model = lstm_klasifikasi(vocab_size, dimensi_embedding, hidden_size, output_size, learning_rate)

                        losses = []
                        for step in range(steps):
                            total_loss = 0
                            for i in range(len(input)):
                                input_seq = input[i]
                                target_train = target[i]

                                # Forward
                                pred = model.forward(input_seq)
                                p = np.clip(pred.item(), 1e-15, 1 - 1e-15)
                                loss = -(target_train * np.log(p) + (1 - target_train) * np.log(1 - p))
                                total_loss += loss

                                # Gradient for BCE
                                dy = np.array([[p - target_train]])
                                dy_seq = [np.zeros_like(dy) for _ in range(len(input_seq) - 1)] + [dy]
                                model.backward(dy_seq)

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
                            vocab_buffer = BytesIO()

                            joblib.dump(vocab, vocab_buffer)
                            vocab_buffer.seek(0)

                            model.save_model(buffer)
                            buffer.seek(0)

                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w") as z:
                                z.writestr("vocab.pkl", vocab_buffer.getvalue())
                                z.writestr("model.npz", buffer.getvalue())

                            zip_buffer.seek(0)

                            st.download_button(
                                label="Download Model",
                                data=zip_buffer,
                                file_name="model.zip",
                                mime="application/zip"
                                )
            elif option == "Validasi/Testing":
                file = st.file_uploader("Unggah Model")
                vocab_file = st.file_uploader("Unggah vocab")

                if file is not None and vocab_file is not None:
                    vocab = joblib.load(vocab_file)

                    vocab_size = len(vocab)
                    dimensi_embedding = 32
                    hidden_size = 32
                    output_size = 1
                    learning_rate = 0.1   

                    model = lstm_klasifikasi(vocab_size, dimensi_embedding, hidden_size, output_size, learning_rate)
                    model.load_model(file)

                    output = []

                    total_loss = 0
                    for i in range(len(input)):
                        pred = model.forward(input[i]).item()
                        target_train = target[i].item()

                        p = np.clip(pred, 1e-15, 1 - 1e-15)
                        loss = -(target_train * np.log(p) + (1 - target_train) * np.log(1 - p))
                        total_loss += loss

                        if pred > 0.5:
                            pred = 1
                        else:
                            pred = 0

                        output.append({
                            "Prediksi": pred,
                            "Target": target_train
                        })

                    df_results = pd.DataFrame(output)
                    st.write("Hasil Pengujian")
                    st.dataframe(df_results)

                    pred_list = [o["Prediksi"] for o in output]
                    target_list = [o["Target"] for o in output]

                    akurasi = accuracy_score(target_list, pred_list)
                    st.write(f"Akurasi: {akurasi:.2%}")

                    cm = confusion_matrix(target_list, pred_list)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Aktual")
                    st.pyplot(fig)

                    st.write(f"Error Rata-Rata: ", total_loss / len(input))
        elif methode == "SVM (RBF)":
            st.subheader("Support Vector Machine (RBF) - Klasifikasi")
            st.info("<!> SVM RBF mendukung fitur numerik dan klasifikasi biner")

            all_cols = dataset.columns.tolist()
            target_options = ["=== Pilih Kolom Target ==="] + all_cols

            # TARGET: boleh teks / numerik
            target_col = st.selectbox(
                "Pilih kolom target (label)",
                target_options,
                index=0
            )

            if target_col == "=== Pilih Kolom Target ===":
                st.warning("Silakan pilih kolom target terlebih dahulu.")
                return

            # FITUR: hanya numerik & bukan target
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

            if "svm_feature_cols" not in st.session_state:
                st.session_state.svm_feature_cols = []

            if "prev_target_col" not in st.session_state:
                st.session_state.prev_target_col = None

            # Jika target berubah → auto isi fitur
            if target_col != st.session_state.prev_target_col:
                st.session_state.svm_feature_cols = [
                    c for c in numeric_cols if c != target_col
                ]
                st.session_state.prev_target_col = target_col
            
            feature_cols = st.multiselect(
                "Pilih kolom fitur (numerik)",
                options=numeric_cols,
                default=st.session_state.svm_feature_cols,
                key="svm_feature_cols"
            )

            if not feature_cols:
                st.warning("Pilih minimal satu kolom fitur numerik.")
                return
            
            # cek binary
            unique_labels = np.sort(dataset[target_col].unique())
            is_numeric_label = pd.api.types.is_numeric_dtype(dataset[target_col])

            if len(unique_labels) > 2:
                st.warning(
                    "Dataset memiliki lebih dari 2 kelas.\n"
                    "SVM hanya mendukung klasifikasi biner.\n\n"
                    "Filter/Mapping data dapat dilakukan!"
                )

                # Numberik ordinal label (filter + local binary)
                if is_numeric_label:
                    min_label = int(unique_labels.min())
                    max_label = int(unique_labels.max())

                    st.info(f"Label ordinal terdeteksi (range {min_label} – {max_label})")

                    selected_value = st.number_input(
                        "Pilih nilai ordinal minimum (nilai di bawah ini diabaikan)",
                        min_value=min_label,
                        max_value=max_label - 1,
                        value=min_label
                    )

                    # step 1: filter data
                    filtered_df = dataset[dataset[target_col] >= selected_value]

                    remaining_labels = np.sort(filtered_df[target_col].unique())

                    if len(remaining_labels) < 2:
                        st.error(
                            "Setelah filtering, hanya tersisa satu kelas.\n"
                            "Pilih nilai ordinal yang lebih kecil."
                        )
                        return

                    # step 2: posisi ordinal seimbang
                    n_labels = len(remaining_labels)
                    split_idx = n_labels // 2  # floor

                    negative_labels = remaining_labels[:split_idx]
                    positive_labels = remaining_labels[split_idx:]

                    st.success(
                        f"""
                        Label tersisa: {remaining_labels.tolist()}\n
                        Kelas Negatif: {negative_labels.tolist()}\n
                        Kelas Positif: {positive_labels.tolist()}
                        """
                    )

                    # step 3: mapping binary
                    filtered_df = filtered_df.copy()

                    label_map = {
                        "__type__": "ordinal_partition",
                        "target_col": target_col,
                        "negative_labels": negative_labels.tolist(),
                        "positive_labels": positive_labels.tolist()
                    }

                    X = filtered_df[feature_cols].values
                    y = filtered_df[target_col].apply(
                        lambda v: 1 if v in positive_labels else 0
                    ).values

                # label text / kategorikal
                else:
                    positive_class = st.selectbox(
                        "Pilih kelas POSITIF",
                        unique_labels
                    )

                    label_map = {
                        c: 1 if c == positive_class else 0
                        for c in unique_labels
                    }

                    X = dataset[feature_cols].values
                    y = dataset[target_col].map(label_map).values

            # label binary (2 kelas)
            else:
                # Encode label ke (0, 1)
                label_map = {
                    unique_labels[0]: 0,
                    unique_labels[1]: 1
                }
            
                y = dataset[target_col].map(label_map).values
                X = dataset[feature_cols].values

            # Mode untuk model SVM-RBF
            if option == "Training":

                st.subheader("Parameter SVM")
                C = st.number_input("C (Regularization)", min_value=0.01, max_value=100.0, value=1.0)
                gamma = st.number_input("Gamma (RBF)", min_value=0.001, max_value=10.0, value=0.1)
                max_passes = st.number_input("Max Passes", min_value=1, max_value=20, value=5)
                tol = st.number_input("Tolerance", value=1e-3, format="%.5f")

                if st.button("Mulai"):
                    # Scaling
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    model = svm_rbf_klasifikasi(C=C, gamma=gamma, tol=tol, max_passes=max_passes)

                    with st.spinner("Training SVM..."):
                        model.fit(X_scaled, y)

                    # Evaluasi (training)
                    y_pred_train = model.predict(X_scaled)
                    acc_train = accuracy_score(y, y_pred_train)

                    st.success(f"Akurasi: {acc_train:.2%}")

                    st.info(
                        f"Jumlah data setelah seleksi ordinal: {len(y)} "
                        f"dari total {len(dataset)} baris"
                    )

                    cm = confusion_matrix(y, y_pred_train)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Aktual")
                    st.pyplot(fig)

                    # safe model
                    buffer_model = BytesIO()
                    buffer_scaler = BytesIO()
                    buffer_label = BytesIO()

                    model.save(buffer_model)
                    buffer_model.seek(0)

                    joblib.dump(scaler, buffer_scaler)
                    buffer_scaler.seek(0)

                    joblib.dump(label_map, buffer_label)
                    buffer_label.seek(0)

                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as z:
                        z.writestr("svm_model.pkl", buffer_model.getvalue())
                        z.writestr("scaler.pkl", buffer_scaler.getvalue())
                        z.writestr("label_map.pkl", buffer_label.getvalue())

                    zip_buffer.seek(0)

                    st.download_button(
                        label="Download Model",
                        data=zip_buffer,
                        file_name="svm_model.zip",
                        mime="application/zip"
                    )

            # validasi / testing
            elif option == "Validasi/Testing":

                model_file = st.file_uploader("Unggah model SVM")
                scaler_file = st.file_uploader("Unggah scaler")
                label_file = st.file_uploader("Unggah label map")

                if model_file and scaler_file and label_file:

                    scaler = joblib.load(scaler_file)
                    label_map = joblib.load(label_file)

                    X_scaled = scaler.transform(X)

                    model = svm_rbf_klasifikasi()
                    model.load(model_file)

                    y_pred = model.predict(X_scaled)

                    results = pd.DataFrame({
                        "Prediksi": y_pred,
                        "Target": y
                    })

                    st.write("Hasil Pengujian")
                    st.dataframe(results)

                    acc = accuracy_score(y, y_pred)
                    st.write(f"Akurasi: {acc:.2%}")

                    cm = confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Prediksi")
                    ax.set_ylabel("Aktual")
                    st.pyplot(fig)