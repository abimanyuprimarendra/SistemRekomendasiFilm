import streamlit as st
import pandas as pd
import re
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ============================================
# Fungsi: Load CSV dari Google Drive
# ============================================
@st.cache_data
def load_data_from_gdrive():
    file_id = "1muHUAK4oi5A16qj-lNkIREwkyAAhgVE5"  # Ganti dengan ID milikmu
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengambil data dari Google Drive.")
        return None
    return pd.read_csv(io.BytesIO(response.content))

# ============================================
# Fungsi: Preprocessing teks deskripsi
# ============================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# ============================================
# Fungsi: Rekomendasi menggunakan Nearest Neighbors
# ============================================
def recommend_nn(title, df, tfidf_matrix, nn_model, n=5):
    title_clean = title.lower()
    match = df['movie title'].str.lower() == title_clean
    if not match.any():
        return None
    idx = df[match].index[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    recommendations = []
    for i in range(1, len(indices[0])):  # skip diri sendiri
        rec = df.iloc[indices[0][i]]
        recommendations.append({
            'Judul': rec['movie title'],
            'Rating': rec.get('Rating', ''),
            'Generes': rec.get('Generes', ''),
            'Deskripsi': rec.get('Overview', '')
        })
    return recommendations

# ============================================
# UI Streamlit
# ============================================
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)

# Load data
df = load_data_from_gdrive()
if df is None:
    st.stop()

# Isi kosong jadi string
for feature in ['movie title', 'Generes', 'Director', 'Writer']:
    df[feature] = df[feature].fillna('')

# Gabung fitur jadi satu deskripsi
df['deskripsi'] = df.apply(lambda row: ' '.join(str(row[feature]) for feature in ['movie title', 'Generes', 'Director', 'Writer']), axis=1)
df['deskripsi'] = df['deskripsi'].apply(preprocess_text)

# TF-IDF & NearestNeighbors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])

nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# Pilihan judul
title_input = st.selectbox("🎞 Pilih Judul Film:", sorted(df['movie title'].dropna().unique()))

# Tampilkan rekomendasi
if st.button("🎯 Tampilkan Rekomendasi"):
    recommendations = recommend_nn(title_input, df, tfidf_matrix, nn_model)

    if recommendations:
        st.markdown(f"<h3>🎯 5 Film Mirip '<span style='color:#950002'>{title_input}</span>'</h3>", unsafe_allow_html=True)

        cols = st.columns(5)  # 5 kolom horizontal

        for i, rec in enumerate(recommendations[:5]):
            with cols[i]:
                genre_clean = rec['Generes']
                if isinstance(genre_clean, str) and genre_clean.startswith('['):
                    try:
                        genre_clean = ', '.join(eval(genre_clean))
                    except:
                        genre_clean = genre_clean.strip("[]").replace("'", "").replace('"', '')

                st.markdown("""
                    <div style='
                        background-color: #fff;
                        padding: 15px;
                        border-radius: 12px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                        height: 100%;
                    '>
                """, unsafe_allow_html=True)

                st.markdown(f"### 🎞 {rec['Judul']}", unsafe_allow_html=True)
                st.markdown(f"**Genre:** {genre_clean}")
                st.markdown(f"**Rating:** {rec['Rating']}")
                st.markdown(f"<span style='font-style: italic; font-size: 13px; color:#555'>{rec['Deskripsi'][:200]}...</span>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Film '{title_input}' tidak ditemukan dalam dataset.")
