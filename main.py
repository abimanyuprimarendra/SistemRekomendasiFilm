import streamlit as st
import pandas as pd
import re
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data dari Google Drive
@st.cache_data
def load_data_from_gdrive():
    file_id = "1muHUAK4oi5A16qj-lNkIREwkyAAhgVE5"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengambil data dari Google Drive.")
        return None
    return pd.read_csv(io.BytesIO(response.content))

# Preprocessing text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# Fungsi rekomendasi
def recommend_nn(title, df, tfidf_matrix, nn_model, n=5):
    title_clean = title.lower()
    match = df['movie title'].str.lower() == title_clean
    if not match.any():
        return None
    idx = df[match].index[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    recommendations = []
    for i in range(1, len(indices[0])):
        rec = df.iloc[indices[0][i]]
        recommendations.append({
            'Judul': rec['movie title'],
            'Rating': rec.get('Rating', ''),
            'Generes': rec.get('Generes', ''),
            'Deskripsi': rec.get('Overview', '')
        })
    return recommendations

# Konfigurasi halaman
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.markdown("<h1 style='text-align:center;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)

# Load data
df = load_data_from_gdrive()
if df is None:
    st.stop()

# Preprocessing
features = ['movie title', 'Generes', 'Director', 'Writer']
for feature in features:
    df[feature] = df[feature].fillna('')
df['deskripsi'] = df.apply(lambda row: ' '.join(str(row[feature]) for feature in features), axis=1)
df['deskripsi'] = df['deskripsi'].apply(preprocess_text)

# TF-IDF + Nearest Neighbors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# Pilihan judul film
title_input = st.selectbox("🎞 Pilih Judul Film:", sorted(df['movie title'].dropna().unique()))

# Tampilkan rekomendasi
if st.button("🎯 Tampilkan Rekomendasi"):
    recommendations = recommend_nn(title_input, df, tfidf_matrix, nn_model)
    if recommendations:
        st.markdown(f"<h3>🎯 5 Film Mirip '<span style='color:#950002'>{title_input}</span>'</h3>", unsafe_allow_html=True)

        # CSS Grid Styling
        st.markdown("""
        <style>
        .card-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            padding: 20px;
            height: 100%;
        }
        .card h4 {
            margin: 0;
            color: #263238;
        }
        .genre {
            font-weight: bold;
            margin-top: 8px;
        }
        .rating {
            font-weight: bold;
            margin-top: 4px;
            color: #444;
        }
        .desc {
            font-style: italic;
            font-size: 14px;
            margin-top: 10px;
            color: #555;
        }
        </style>
        <div class="card-container">
        """, unsafe_allow_html=True)

        for rec in recommendations:
            # Clean genre
            genre_clean = rec['Generes']
            if isinstance(genre_clean, str) and genre_clean.startswith('['):
                try:
                    genre_clean = ', '.join(eval(genre_clean))
                except:
                    genre_clean = genre_clean.strip("[]").replace("'", "").replace('"', '')

            st.markdown(f"""
                <div class="card">
                    <h4>🎞 {rec['Judul']}</h4>
                    <div class="genre">Genre: {genre_clean}</div>
                    <div class="rating">Rating: {rec['Rating']}</div>
                    <div class="desc">{rec['Deskripsi'][:250]}...</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Film '{title_input}' tidak ditemukan dalam dataset.")
