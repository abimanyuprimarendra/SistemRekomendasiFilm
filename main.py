import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# Fungsi: Load CSV dari Google Drive (bukan Sheet)
# ============================================
@st.cache_data
def load_data_from_gdrive():
    file_id = "1muHUAK4oi5A16qj-lNkIREwkyAAhgVE5"  # Ganti dengan ID file CSV di Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("❌ Gagal mengambil data dari Google Drive.")
        return None

    try:
        df = pd.read_csv(io.BytesIO(response.content))
        if 'Rating' in df.columns:
            df = df.sort_values('Rating', ascending=False).head(5000)
        else:
            df = df.head(5000)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memproses file CSV: {e}")
        return None

# ============================================
# Fungsi: Preprocessing teks deskripsi
# ============================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# ============================================
# Fungsi: Rekomendasi berdasarkan cosine similarity
# ============================================
def recommend_cosine(title, df, cosine_sim, n=5):
    title_clean = re.sub(r'[^a-z\s]', '', title.lower())
    try:
        match = df['movie title'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x).lower()))
        idx = df[match == title_clean].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        recommendations = []
        for i, score in sim_scores:
            recommendations.append({
                'Judul'     : df.iloc[i]['movie title'],
                'Rating'    : df.iloc[i].get('Rating', 'N/A'),
                'Generes'   : df.iloc[i].get('Generes', ''),
                'Deskripsi' : df.iloc[i].get('Overview', ''),
                'Writer'    : df.iloc[i].get('Writer', ''),
                'Director'  : df.iloc[i].get('Director', ''),
                'Similarity': round(score, 4)
            })
        return recommendations
    except IndexError:
        return None

# ============================================
# UI Streamlit
# ============================================
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")

# Load data
df = load_data_from_gdrive()
if df is None:
    st.stop()

# Tampilkan data awal (opsional)
with st.expander("🔍 Lihat Data Mentah"):
    st.dataframe(df.head())

# Isi kosong jadi string
features = ['movie title', 'Generes', 'Director', 'Writer']
for feature in features:
    df[feature] = df[feature].fillna('')

# Gabung semua fitur jadi 1 deskripsi
df['deskripsi'] = df.apply(lambda row: ' '.join(row[feature] for feature in features), axis=1)
df['deskripsi'] = df['deskripsi'].apply(preprocess_text)

# TF-IDF & cosine similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Dropdown pilihan film
title_input = st.selectbox(
    "🎞 Pilih Judul Film untuk Rekomendasi:",
    sorted(df['movie title'].dropna().unique())
)

if st.button("🎯 Tampilkan Rekomendasi"):
    recommendations = recommend_cosine(title_input, df, cosine_sim)

    if recommendations:
        st.subheader(f"📌 Rekomendasi untuk: **{title_input}**")
        for rec in recommendations:
            with st.container():
                st.markdown(f"### 🎥 {rec['Judul']}")
                st.write(f"**Genre:** {rec['Generes']}")
                st.write(f"**Rating:** {rec['Rating']}")
                st.write(f"**Director:** {rec['Director']}")
                st.write(f"**Writer:** {rec['Writer']}")
                st.write(f"**Similarity Score:** {rec['Similarity']}")
                st.info(rec['Deskripsi'])
                st.markdown("---")
    else:
        st.warning(f"⚠️ Film '{title_input}' tidak ditemukan dalam dataset.")
