import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# ---------- CONFIGURASI & LOADING ----------
st.set_page_config("Analisis Setimen Aplikasi ALLO BANK", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("ulasan_terproses.xlsx")
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {'score': 'Rating', 'clean_content': 'Stemming'}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if 'Rating' not in df.columns or 'Stemming' not in df.columns:
        st.error(f"❌ Kolom yang dibutuhkan tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")
        st.stop()

    df['Sentimen'] = df['Rating'].apply(lambda x: 'Positif' if x >= 4 else ('Negatif' if x <= 2 else 'Netral'))
    df = df.dropna(subset=['Stemming'])
    df.rename(columns={'Stemming': 'stemming'}, inplace=True)
    return df

data = load_data()

@st.cache_resource
def prepare_model(df):
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_all = tfidf.fit_transform(df['stemming'])
    y_all = df['Sentimen'].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_all, y_all, df.index, test_size=0.2, random_state=42, stratify=y_all)
    train_texts = df.loc[idx_train]
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    return tfidf, svm, X_train, y_train, train_texts

tfidf, svm_model, X_train, y_train, train_texts = prepare_model(data)

def preprocess_text(text):
    text = text.lower()
    return ''.join(c for c in text if c.isalnum() or c.isspace())

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=['Positif', 'Netral', 'Negatif'])
    return acc, prec, rec, f1, cm, y_pred

# ---------- MENU UTAMA ----------
menu = st.sidebar.radio("ANALISIS SENTIMEN APLIKASI ALLO BANK", [
    "Dashboard", "Confusion Matrix", "Wordcloud", "Evaluasi Metrik", "Prediksi Ulasan Baru"
])

# ---------- DASHBOARD ----------
if menu == "Dashboard":
    st.title("Dashboard Ulasan Aplikasi ALLO BANK")
    st.write("Jumlah data setelah preprocessing:", len(data))
    count_sentiment = data['Sentimen'].value_counts()
    st.subheader("Distribusi Sentimen")
    fig = px.pie(names=count_sentiment.index, values=count_sentiment.values, color=count_sentiment.index,
                 color_discrete_map={'Positif': 'green', 'Negatif': 'red', 'Netral': 'purple'}, title="Distribusi Sentimen")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Contoh Data")
    st.dataframe(data[['stemming', 'Rating', 'Sentimen']].sample(10))

# ---------- CONFUSION MATRIX ----------
elif menu == "Confusion Matrix":
    st.title("Confusion Matrix (Data Latih)")
    acc_svm, _, _, _, cm_svm, _ = evaluate_model(svm_model, X_train, y_train)
    st.subheader("SVM")
    st.write(f"Akurasi: {acc_svm:.2f}")
    fig, ax = plt.subplots()
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positif', 'Netral', 'Negatif'], yticklabels=['Positif', 'Netral', 'Negatif'], ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

# ---------- WORDCLOUD ----------
elif menu == "Wordcloud":
    st.title("Wordcloud Prediksi Model (Data Latih)")
    _, _, _, _, _, svm_pred = evaluate_model(svm_model, X_train, y_train)
    def plot_wordcloud(predictions, label, color):
        filtered = train_texts[predictions == label]
        text = ' '.join(filtered['stemming'])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white', colormap=color).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write(f"Tidak ada ulasan untuk label {label}")
    for label, cmap in [('Positif', 'Greens'), ('Netral', 'Purples'), ('Negatif', 'Reds')]:
        st.subheader(f"SVM - {label}")
        plot_wordcloud(svm_pred, label, cmap)

# ---------- METRIK ----------
elif menu == "Evaluasi Metrik":
    st.title("Evaluasi Metrik Model (Data Latih)")
    acc, prec, rec, f1, _, _ = evaluate_model(svm_model, X_train, y_train)
    st.markdown("### Skor Evaluasi SVM:")
    st.markdown(f"- **Akurasi:** {acc:.2f}")
    st.markdown(f"- **Presisi:** {prec:.2f}")
    st.markdown(f"- **Recall:** {rec:.2f}")
    st.markdown(f"- **F1-Score:** {f1:.2f}")
    fig_bar = px.bar(x=['Akurasi', 'Presisi', 'Recall', 'F1-Score'], y=[acc, prec, rec, f1],
                     text=[f"{v:.2f}" for v in [acc, prec, rec, f1]],
                     color_discrete_sequence=['#ff80bf', '#ff99cc', '#ffb3d9', '#ffcce6'],
                     title="Evaluasi Model SVM")
    fig_bar.update_layout(yaxis_range=[0, 1], title_x=0.5)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------- PREDIKSI FILE & MANUAL ----------
elif menu == "Prediksi Ulasan Baru":
    st.title("🔮 Prediksi Sentimen Ulasan")

    # --- File Upload ---
    st.header("📂 Prediksi dari File CSV / Excel")
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            st.stop()
        st.write("📋 Kolom ditemukan:", df_new.columns.tolist())
        if 'ulasan' not in df_new.columns:
            st.error("❌ Kolom 'ulasan' tidak ditemukan.")
            st.stop()
        df_new = df_new.dropna(subset=['ulasan'])
        df_new['clean'] = df_new['ulasan'].apply(preprocess_text)
        tfidf_new = tfidf.transform(df_new['clean'])
        df_new['Prediksi_SVM'] = svm_model.predict(tfidf_new)
        if 'rating' in df_new.columns:
            df_new['rating'] = pd.to_numeric(df_new['rating'], errors='coerce')
            df_new.dropna(subset=['rating'], inplace=True)
            df_new['label'] = df_new['rating'].apply(lambda x: 'Positif' if x >= 4 else ('Negatif' if x <= 2 else 'Netral'))
        st.subheader("Distribusi Prediksi SVM")
        fig = px.pie(names=df_new['Prediksi_SVM'].value_counts().index,
                     values=df_new['Prediksi_SVM'].value_counts().values,
                     color_discrete_map={'Positif': 'green', 'Netral': 'purple', 'Negatif': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Wordcloud dari Prediksi")
        for label, cmap in [('Positif', 'Greens'), ('Netral', 'Purples'), ('Negatif', 'Reds')]:
            st.markdown(f"**{label}**")
            text = ' '.join(df_new[df_new['Prediksi_SVM'] == label]['clean'])
            if text.strip():
                wc = WordCloud(width=800, height=400, background_color='white', colormap=cmap).generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("Tidak ada data")
        if 'label' in df_new.columns:
            st.subheader("Evaluasi terhadap Label Asli")
            acc = accuracy_score(df_new['label'], df_new['Prediksi_SVM'])
            prec = precision_score(df_new['label'], df_new['Prediksi_SVM'], average='macro')
            rec = recall_score(df_new['label'], df_new['Prediksi_SVM'], average='macro')
            f1 = f1_score(df_new['label'], df_new['Prediksi_SVM'], average='macro')
            st.markdown(f"- **Akurasi:** `{acc:.2f}`")
            st.markdown(f"- **Presisi:** `{prec:.2f}`")
            st.markdown(f"- **Recall:** `{rec:.2f}`")
            st.markdown(f"- **F1-Score:** `{f1:.2f}`")

    # --- Manual Input ---
    st.header("📝 Prediksi Ulasan Manual")
    model_path = "model_svm.pkl"
    vectorizer_path = "vectorizer_tfidf.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.warning("Model atau vectorizer belum ditemukan. Silakan jalankan training terlebih dahulu.")
    else:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        ulasan = st.text_area("Masukkan ulasan pengguna:")
        if st.button("Prediksi Sentimen Manual"):
            if not ulasan.strip():
                st.warning("Teks ulasan tidak boleh kosong.")
            else:
                vec = vectorizer.transform([preprocess_text(ulasan)])
                hasil = model.predict(vec)[0]
                warna = {"positif": "green", "netral": "orange", "negatif": "red"}
                st.success("🎯 Prediksi Selesai")
                st.markdown(f"- **Teks**: _{ulasan}_")
                st.markdown(f"- **Hasil Sentimen**: <span style='color:{warna.get(hasil.lower(), 'black')}; font-weight:bold'>{hasil.upper()}</span>", unsafe_allow_html=True)

