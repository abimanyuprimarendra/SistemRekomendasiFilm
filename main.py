import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# FILE ID Google Drive CSV FIXED
# ===============================
FILE_ID = "1muHUAK4oi5A16qj-lNkIREwkyAAhgVE5"
FILE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ===============================
# Fungsi Load Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_URL)
    return df

# ===============================
# Preprocessing Teks
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# ===============================
# Fungsi Rekomendasi
# ===============================
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

# ===============================
# Streamlit App Layout
# ===============================
st.title("🎬 Rekomendasi Film IMDb")

# Load data
df = load_data()

# Isi kosong jadi string
features = ['movie title', 'Generes', 'Director', 'Writer']
for feature in features:
    df[feature] = df[feature].fillna('')
df['deskripsi'] = df.apply(lambda row: ' '.join(row[feature] for feature in features), axis=1)
df['deskripsi'] = df['deskripsi'].apply(preprocess_text)

# TF-IDF dan cosine similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Input judul film
title_input = st.text_input("Masukkan Judul Film:", "Spider-Man")

if st.button("Tampilkan Rekomendasi"):
    recommendations = recommend_cosine(title_input, df, cosine_sim)

    if recommendations:
        st.subheader(f"Rekomendasi untuk: **{title_input}**")
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
        st.warning(f"Film '{title_input}' tidak ditemukan dalam dataset.")

