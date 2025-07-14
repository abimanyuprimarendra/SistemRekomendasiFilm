import streamlit as st
import pandas as pd
import re
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

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
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.markdown("<h1 style='text-align:center;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)

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
# Sidebar input
with st.sidebar:
    st.subheader("🎞 Pilih Judul Film")
    selected_title = st.selectbox("", sorted(df['movie title'].dropna().unique()))
    search = st.button("Cari Rekomendasi")

# ============================================
# Rekomendasi Film
if search:
    recommendations = recommend_nn(selected_title, df, tfidf_matrix, nn_model)
    if recommendations:
        st.markdown(f"<h3>🎯 5 Film Mirip '<span style='color:#950002'>{selected_title}</span>'</h3>", unsafe_allow_html=True)

        image_url = "https://raw.githubusercontent.com/abimanyuprimarendra/SistemRekomendasiFilm/main/gambar.jpeg"
        cols = st.columns(5)

        for i, rec in enumerate(recommendations[:5]):
            with cols[i]:
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
                        padding: 14px;
                        height: 350px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        display: flex;
                        flex-direction: column;
                        justify-content: flex-start;
                        align-items: center;
                        text-align: left;
                    '>
                        <img src="{image_url}" style="width: 100%; height: 110px; border-radius: 10px; object-fit: cover; margin-bottom: 8px;" />
                        <div style='width: 100%; flex-grow: 1; display: flex; flex-direction: column; '>
                            <div>
                                <h4 style='margin: 0 0 4px 0; font-size: 20px; font-weight: bold;'>🎬 {rec['Judul']}</h4>
                                <p style='margin: 0; font-size: 13px;'><strong>Genre:</strong> {genre_clean}</p>
                                <p style='margin: 0 0 6px 0; font-size: 13px;'><strong>Rating:</strong> {rec['Rating']}</p>
                            </div>
                            <div style='margin-top: 6px;'>
                                <p style='font-size: 12px; color: #444; line-height: 1.25; margin: 0; text-align: justify;'>
                                    {rec['Deskripsi'][:180]}...
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠ Film '{selected_title}' tidak ditemukan dalam dataset.")
