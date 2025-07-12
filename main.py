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
    file_id = "1muHUAK4oi5A16qj-lNkIREwkyAAhgVE5"  # Ganti dengan ID file CSV kamu
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("❌ Gagal mengambil data dari Google Drive.")
        return None

    try:
        df = pd.read_csv(io.BytesIO(response.content))
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
    return ' '.join(text.split())

# ============================================
# Fungsi: Rekomendasi menggunakan NearestNeighbors
# ============================================
def recommend_nn(title, df, tfidf_matrix, nn_model, n=5):
    title_clean = title.lower()
    match = df['movie title'].str.lower() == title_clean
    if not match.any():
        return None

    idx = df[match].index[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)  # +1 untuk dirinya sendiri

    recommendations = []
    for i in range(1, len(indices[0])):  # Mulai dari 1 agar tidak menampilkan film itu sendiri
        rec_idx = indices[0][i]
        rec = df.iloc[rec_idx]
        recommendations.append({
            'Judul': rec['movie title'],
            'Rating': rec.get('Rating', ''),
            'Generes': rec.get('Generes', ''),
            'Deskripsi': rec.get('Overview', ''),
            'Writer': rec.get('Writer', ''),
            'Director': rec.get('Director', '')
        })
    return recommendations

# ============================================
# UI Streamlit
# ============================================
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")

# Load data
df = load_data_from_gdrive()
if df is None:
    st.stop()

# Isi kosong jadi string
features = ['movie title', 'Generes', 'Director', 'Writer']
for feature in features:
    df[feature] = df[feature].fillna('')

# Gabung semua fitur jadi 1 deskripsi
df['deskripsi'] = df.apply(lambda row: ' '.join(str(row[feature]) for feature in features), axis=1)
df['deskripsi'] = df['deskripsi'].apply(preprocess_text)

# TF-IDF & NearestNeighbors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])

nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# Dropdown pilihan film
title_input = st.selectbox(
    "🎞 Pilih Judul Film untuk Rekomendasi:",
    sorted(df['movie title'].dropna().unique())
)

if st.button("🎯 Tampilkan Rekomendasi"):
    recommendations = recommend_nn(title_input, df, tfidf_matrix, nn_model)

    if recommendations:
        st.subheader(f"📌 Rekomendasi untuk: **{title_input}**")
        for rec in recommendations:
            with st.container():
                st.markdown(f"### 🎥 {rec['Judul']}")

                # Bersihkan Genre dari list string
                if rec['Generes']:
                    genre_clean = rec['Generes']
                    if isinstance(genre_clean, str) and genre_clean.startswith('['):
                        try:
                            genre_clean = ', '.join(eval(genre_clean))
                        except:
                            genre_clean = genre_clean.strip("[]").replace("'", "").replace('"', '')
                    st.write(f"**Genre:** {genre_clean}")

                if rec['Rating']:
                    st.write(f"**Rating:** {rec['Rating']}")
                if rec['Director']:
                    st.write(f"**Director:** {rec['Director']}")
                if rec['Writer']:
                    st.write(f"**Writer:** {rec['Writer']}")
                if rec['Deskripsi']:
                    st.info(rec['Deskripsi'])

                st.markdown("---")
    else:
        st.warning(f"⚠️ Film '{title_input}' tidak ditemukan dalam dataset.")
