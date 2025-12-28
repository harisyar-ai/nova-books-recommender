# app/app.py

import streamlit as st
import pickle
from sklearn.metrics.pairwise import linear_kernel
import ast
import pandas as pd
from collections import Counter
import numpy as np
import gzip
import os
import requests

# PAGE CONFIG
st.set_page_config(
    page_title="Nova Books Recommender",
    page_icon="📚",
    layout="wide"
)

# ================= BACKGROUND IMAGE =================
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.78)),
        url("https://images.unsplash.com/photo-1512820790803-83ca734da794");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)
# ====================================================

# SESSION STATE
for key, val in {
    "preloaded_recs": None,
    "current_book": None,
    "current_genre": None,
    "genre_page": 0,
    "current_author": None,
    "author_page": 0,
    "search_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# LOAD MODEL
@st.cache_resource
def load_artifacts():
    url = "https://huggingface.co/harisyar/nova-books-recommender/resolve/main/tfidf_features.pkl.gz"
    path = "models/tfidf_features.pkl.gz"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(path):
        with st.spinner("Downloading model (first run only)..."):
            r = requests.get(url)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)

    with gzip.open(path, "rb") as f:
        data = pickle.load(f)

    df = data["df"].set_index("Book", drop=False)
    return df, data["tfidf_matrix"]

original_df, tfidf_matrix = load_artifacts()

# HELPERS
@st.cache_data
def get_cover_url(x):
    return x if pd.notna(x) and str(x).strip() else "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

@st.cache_data
def top_two_genres(g):
    try:
        g = ast.literal_eval(str(g))
        if isinstance(g, list):
            return g[:2]
    except:
        pass
    return [str(g)] if g else []

# RECOMMENDERS
@st.cache_data
def get_similar_books(book, n=5):
    idx = original_df.index.get_loc(book)
    sims = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    top = np.argsort(sims)[::-1][1:n+1]
    return original_df.iloc[top].reset_index(drop=True)

@st.cache_data
def get_genre_books(genre, n=50):
    genre_books = original_df[
        original_df["Genres"].str.contains(genre, case=False, na=False)
    ]
    return genre_books.head(n).reset_index(drop=True)

@st.cache_data
def get_author_books(author):
    return original_df[
        original_df["Author"].str.contains(author, case=False, na=False)
    ].reset_index(drop=True)

@st.cache_data
def get_popular_genres():
    all_g = []
    for g in original_df["Genres"].dropna():
        all_g.extend(top_two_genres(g))
    return sorted(Counter(all_g), key=Counter(all_g).get, reverse=True)

popular_genres = get_popular_genres()

# UI STYLES
st.markdown("""
<style>
.book-card {
    background: rgba(255,255,255,0.06);
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    transition: 0.3s;
}
.book-card:hover {
    transform: scale(1.08);
    box-shadow: 0 0 25px rgba(255,80,80,0.55);
}
.glow {
    font-size:48px;
    text-align:center;
    color:#FF4B4B;
    font-weight:700;
}
.sub {
    text-align:center;
    color:#aaa;
    margin-bottom:30px;
}
</style>

<div class="glow">📚 Nova Books Recommender</div>
<div class="sub">We Wish to Recommend Your Desired Books 🧡</div>
""", unsafe_allow_html=True)

def render_card(row):
    st.markdown(f"""
    <div class="book-card">
        <img src="{get_cover_url(row['Cover_URL'])}" style="width:130px;height:200px;border-radius:8px;">
        <h4 style="color:white">{row['Book']}</h4>
        <p style="color:#aaa">{row['Author']}</p>
        <p style="color:#FFD700">⭐ {row.get('Avg_Rating','N/A')}</p>
        <a href="{row['URL']}" target="_blank" style="color:#FF4B4B">Goodreads →</a>
    </div>
    """, unsafe_allow_html=True)

# SIDEBAR
page = st.sidebar.radio("Navigation", [
    "Search by Book",
    "Search by Author",
    "Search by Genre",
    "About Us"
])

# PAGES
if page == "Search by Book":
    book = st.selectbox("Select Book", [""] + list(original_df.index))
    if book:
        cols = st.columns(5)
        for i, r in get_similar_books(book).iterrows():
            with cols[i]:
                render_card(r)

elif page == "Search by Author":
    author = st.selectbox("Select Author", [""] + sorted(original_df["Author"].unique()))
    if author:
        cols = st.columns(5)
        for i, r in get_author_books(author).head(5).iterrows():
            with cols[i]:
                render_card(r)

elif page == "Search by Genre":
    genre = st.selectbox("Select Genre", [""] + popular_genres)
    if genre:
        cols = st.columns(5)
        for i, r in get_genre_books(genre).head(5).iterrows():
            with cols[i]:
                render_card(r)

elif page == "About Us":
    st.write("Built by **Muhammad Haris Afridi** — ML-powered book discovery.")

st.caption("© Built by Muhammad Haris Afridi | Streamlit & Hugging Face")
