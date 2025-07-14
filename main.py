import streamlit as st
import pandas as pd
import re
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ============================================
# Load Dataset dari Google Drive
# ============================================
@st.cache_data
def load_data_from_gdrive():
    file_id = "1muHUAK4oi5A16qj-lNkIREwkyAAhgVE5"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("❌ Gagal mengambil data dari Google Drive.")
        return None
    return pd.read_csv(io.BytesIO(response.content))

# ============================================
# Preprocessing teks deskripsi
# ============================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# ============================================
# Rekomendasi Film
# ============================================
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

# ============================================
# Konfigurasi Streamlit
# ============================================
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.markdown("<h1 style='text-align:center;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)

# ============================================
# Load dan proses data
# ============================================
df = load_data_from_gdrive()
if df is None:
    st.stop()

for feature in ['movie title', 'Generes', 'Director', 'Writer']:
    df[feature] = df[feature].fillna('')

df['deskripsi'] = df.apply(lambda row: ' '.join(str(row[feature]) for feature in ['movie title', 'Generes', 'Director', 'Writer']), axis=1)
df['deskripsi'] = df['deskripsi'].apply(preprocess_text)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# ============================================
# Sidebar: Pilih judul dan tombol cari
# ============================================
with st.sidebar:
    st.subheader("🎞 Pilih Judul Film")
    selected_title = st.selectbox("", sorted(df['movie title'].dropna().unique()))
    search = st.button("Cari Rekomendasi")

# ============================================
# Tampilkan Rekomendasi jika tombol diklik
# ============================================
if search:
    recommendations = recommend_nn(selected_title, df, tfidf_matrix, nn_model)
    if recommendations:
        st.markdown(f"<h3>🎯 5 Film Mirip '<span style='color:#950002'>{selected_title}</span>'</h3>", unsafe_allow_html=True)

        image_url = "https://raw.githubusercontent.com/abimanyuprimarendra/SistemRekomendasiFilm/main/gambar.jpeg"

        for rec in recommendations:
            genre_clean = rec['Generes']
            if isinstance(genre_clean, str) and genre_clean.startswith('['):
                try:
                    genre_clean = ', '.join(eval(genre_clean))
                except:
                    genre_clean = genre_clean.strip("[]").replace("'", "").replace('"', '')

            st.markdown(f"""
                <div style='
                    background-color: #ffffff;
                    border-radius: 16px;
                    padding: 20px;
                    margin-bottom: 25px;
                    box-shadow: 0 6px 16px rgba(0,0,0,0.1);
                    display: flex;
                    flex-direction: row;
                    gap: 20px;
                    flex-wrap: wrap;
                '>
                    <div style="flex: 0 0 200px;">
                        <img src="{image_url}" style="width: 100%; height: auto; border-radius: 12px; object-fit: cover;" />
                    </div>
                    <div style="flex: 1;">
                        <h4 style='margin-top: 0;'>🎬 {rec['Judul']}</h4>
                        <p style='margin: 4px 0;'><strong>Genre:</strong> {genre_clean}</p>
                        <p style='margin: 4px 0;'><strong>Rating:</strong> {rec['Rating']}</p>
                        <p style='margin-top: 12px; color: #444; font-size: 15px; line-height: 1.5;'>
                            {rec['Deskripsi'][:400]}...
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠ Film '{selected_title}' tidak ditemukan dalam dataset.")
