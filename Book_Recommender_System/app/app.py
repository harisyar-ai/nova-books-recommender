# app/streamlit_app.py

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

# ===================== BACKGROUND STYLE (NEW) =====================
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)
# ================================================================

# SESSION STATE
if 'preloaded_recs' not in st.session_state:
    st.session_state.preloaded_recs = None
if 'current_book' not in st.session_state:
    st.session_state.current_book = None
if 'current_genre' not in st.session_state:
    st.session_state.current_genre = None
if 'genre_page' not in st.session_state:
    st.session_state.genre_page = 0
if 'current_author' not in st.session_state:
    st.session_state.current_author = None
if 'author_page' not in st.session_state:
    st.session_state.author_page = 0
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    file_url = "https://huggingface.co/harisyar/nova-books-recommender/resolve/main/tfidf_features.pkl.gz"
    local_path = "models/tfidf_features.pkl.gz"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(local_path):
        with st.spinner("Loading book model for the first time (14MB)..."):
            response = requests.get(file_url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
        st.success("✅ Model ready!", icon="📚")

    with gzip.open(local_path, "rb") as f:
        artifacts = pickle.load(f)

    df = artifacts["df"].set_index("Book", drop=False)
    tfidf_matrix = artifacts["tfidf_matrix"]
    return df, tfidf_matrix

original_df, tfidf_matrix = load_artifacts()

# HELPERS
@st.cache_data
def get_cover_url(cover):
    if pd.notna(cover) and str(cover).strip() not in ['', 'nan']:
        return str(cover)
    return "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

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
    indices = np.argsort(sims)[::-1][1:n+1]
    return original_df.iloc[indices].reset_index(drop=True)

@st.cache_data
def get_author_books(author):
    return original_df[original_df["Author"].str.contains(author, case=False, na=False)]

@st.cache_data
def get_genre_books(genre):
    return original_df[original_df["Genres"].str.contains(genre, case=False, na=False)]

@st.cache_data
def get_popular_genres():
    genres = []
    for g in original_df["Genres"].dropna():
        genres.extend(top_two_genres(g))
    return sorted(Counter(genres), key=Counter(genres).get, reverse=True)

popular_genres = get_popular_genres()

# CARD UI
def render_goodreads_card(row):
    st.markdown(f"""
    <div class="book-card">
        <img src="{get_cover_url(row['Cover_URL'])}" style="width:130px;height:200px;border-radius:8px;">
        <h4 style="color:white;margin:8px 0 2px;">{row['Book']}</h4>
        <p style="color:#aaa;font-size:13px;">{row['Author']}</p>
        <p style="color:#FFD700;">⭐ {row.get('Avg_Rating','N/A')}</p>
        <a href="{row['URL']}" target="_blank" style="color:#FF4B4B;">View on Goodreads →</a>
    </div>
    """, unsafe_allow_html=True)

# GLOBAL STYLES
st.markdown("""
<style>
.glow {font-size:48px;text-align:center;color:#FF4B4B;font-weight:700;}
.sub {text-align:center;color:#aaa;font-size:18px;margin-bottom:30px;}
.book-card {
    background:rgba(255,255,255,0.06);
    padding:12px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.25);
    transition:0.3s;
}
.book-card:hover {
    transform:scale(1.08);
    box-shadow:0 0 25px rgba(255,80,80,0.55);
}
</style>

<div class="glow">📚 Nova Books Recommender</div>
<div class="sub">We Wish to Recommend Your Desired Books 🧡</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    page = st.radio("Navigate", [
        "Search by Book",
        "Search by Author",
        "Search by Genre",
        "Recent Searches",
        "About Us"
    ])

# PAGES
if page == "Search by Book":
    book = st.selectbox("Select Book", [""] + list(original_df["Book"]))
    if book:
        recs = get_similar_books(book)
        cols = st.columns(5)
        for i, r in recs.iterrows():
            with cols[i]:
                render_goodreads_card(r)

elif page == "Search by Author":
    author = st.selectbox("Select Author", [""] + list(original_df["Author"].unique()))
    if author:
        books = get_author_books(author)
        cols = st.columns(5)
        for i, r in books.head(5).iterrows():
            with cols[i]:
                render_goodreads_card(r)

elif page == "Search by Genre":
    genre = st.selectbox("Select Genre", [""] + popular_genres)
    if genre:
        books = get_genre_books(genre)
        cols = st.columns(5)
        for i, r in books.head(5).iterrows():
            with cols[i]:
                render_goodreads_card(r)

elif page == "Recent Searches":
    st.info("Feature unchanged")

elif page == "About Us":
    st.markdown("### Muhammad Haris Afridi")
    st.write("Nova Books is a machine-learning powered recommender system built with love for books and AI.")

st.caption("© Built by Muhammad Haris Afridi | Streamlit + Hugging Face")    return genre_books.head(n).reset_index(drop=True)

@st.cache_data
def get_author_books(author_name):
    books = original_df[original_df["Author"].str.contains(author_name, case=False, na=False)].copy()
    if 'Avg_Rating' in books.columns:
        books = books.sort_values('Avg_Rating', ascending=False)
    return books.reset_index(drop=True)

@st.cache_data
def get_popular_genres():
    all_genres = []
    for g in original_df["Genres"].dropna():
        all_genres.extend(top_two_genres(g))
    return sorted(Counter(all_genres), key=Counter(all_genres).get, reverse=True)

popular_genres = get_popular_genres()

# BOOK CARD STYLE - YOUR ORIGINAL (NO CHANGES TO SIZES ANYWHERE)
def render_goodreads_card(row):
    cover = get_cover_url(row.get('Cover_URL', ''))
    title = row['Book']
    author = row['Author']
    rating = row.get('Avg_Rating', 'N/A')
    genres = top_two_genres(row.get('Genres', ''))
    url = row.get('URL', '#')

    genre_tags = " ".join([f"<small style='background:#333; color:white; padding:3px 8px; border-radius:12px; margin:2px; font-size:12px;'>{g}</small>" for g in genres])

    card_html = f"""
    <div class="book-card">
        <img src="{cover}" style="width:130px; height:200px; object-fit:cover; border-radius:8px;">
        <h4 style="margin:8px 0 2px; font-size:16px; color:white;">{title}</h4>
        <p style="color:#aaa; margin:2px 0; font-size:13px;">Author: {author}</p>
        <p style="color:#FFD700; margin:4px 0;">⭐ {rating}</p>
        <div style="margin:6px 0;">{genre_tags}</div>
        <a href="{url}" target="_blank" style="color:#FF4B4B; font-size:13px; text-decoration:none;">View on Goodreads →</a>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

# STYLES - YOUR ORIGINAL (NO MOBILE SIZE CHANGES AT ALL)
st.markdown(
    """
    <style>
    .glow {font-size:48px; text-align:center; color:#FF4B4B; font-weight:700;
           animation: glow 2.5s ease-in-out infinite;}
    @keyframes glow {0%,100% {text-shadow:0 0 5px #FF4B4B;} 50% {text-shadow:0 0 25px #FF4B4B;}}
    .sub {text-align:center; color:#aaa; font-size:18px; margin-bottom:30px;}
    .book-card {background:rgba(255,255,255,0.05); padding:12px; border-radius:12px; text-align:center; 
                box-shadow:0 4px 12px rgba(0,0,0,0.2); margin:5px 0; transition:all 0.3s ease; cursor:pointer;}
    .book-card:hover {transform:scale(1.08) translateY(-5px); box-shadow:0 0 25px rgba(255,80,80,0.55);}
    </style>
    <div class="glow">📚 Nova Books Recommender </div>
    <div class="sub">We Wish to Recommend Your Desired Books 🧡</div>
    """,
    unsafe_allow_html=True
)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("<h2 style='color:#FF4B4B; font-size:24px; margin-bottom:20px;'>Navigation</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Choose a page",
        ["Search by Book", "Search by Author", "Search by Genre", "Recent Searches", "About Us"],
        label_visibility="collapsed"
    )

# PAGES - YOUR ORIGINAL 5-COLUMN GRID EVERYWHERE (NO MOBILE CHANGES)
if page == "Search by Book":
    st.header("Find Similar Books")
    book = st.selectbox("Select a Book:", [""] + list(original_df["Book"].unique()))
    
    if book:
        if book != st.session_state.current_book:
            st.session_state.current_book = book
            st.session_state.preloaded_recs = None
            if book not in st.session_state.search_history:
                st.session_state.search_history.insert(0, book)
                st.session_state.search_history = st.session_state.search_history[:20]

        selected = original_df.loc[book]
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(get_cover_url(selected['Cover_URL']), use_container_width=True)
        with col2:
            st.subheader(book)
# app/streamlit_app.py

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

# ===================== BACKGROUND STYLE (NEW) =====================
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)
# ================================================================

# SESSION STATE
if 'preloaded_recs' not in st.session_state:
    st.session_state.preloaded_recs = None
if 'current_book' not in st.session_state:
    st.session_state.current_book = None
if 'current_genre' not in st.session_state:
    st.session_state.current_genre = None
if 'genre_page' not in st.session_state:
    st.session_state.genre_page = 0
if 'current_author' not in st.session_state:
    st.session_state.current_author = None
if 'author_page' not in st.session_state:
    st.session_state.author_page = 0
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    file_url = "https://huggingface.co/harisyar/nova-books-recommender/resolve/main/tfidf_features.pkl.gz"
    local_path = "models/tfidf_features.pkl.gz"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(local_path):
        with st.spinner("Loading book model for the first time (14MB)..."):
            response = requests.get(file_url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
        st.success("✅ Model ready!", icon="📚")

    with gzip.open(local_path, "rb") as f:
        artifacts = pickle.load(f)

    df = artifacts["df"].set_index("Book", drop=False)
    tfidf_matrix = artifacts["tfidf_matrix"]
    return df, tfidf_matrix

original_df, tfidf_matrix = load_artifacts()

# HELPERS
@st.cache_data
def get_cover_url(cover):
    if pd.notna(cover) and str(cover).strip() not in ['', 'nan']:
        return str(cover)
    return "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

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
    indices = np.argsort(sims)[::-1][1:n+1]
    return original_df.iloc[indices].reset_index(drop=True)

@st.cache_data
def get_author_books(author):
    return original_df[original_df["Author"].str.contains(author, case=False, na=False)]

@st.cache_data
def get_genre_books(genre):
    return original_df[original_df["Genres"].str.contains(genre, case=False, na=False)]

@st.cache_data
def get_popular_genres():
    genres = []
    for g in original_df["Genres"].dropna():
        genres.extend(top_two_genres(g))
    return sorted(Counter(genres), key=Counter(genres).get, reverse=True)

popular_genres = get_popular_genres()

# CARD UI
def render_goodreads_card(row):
    st.markdown(f"""
    <div class="book-card">
        <img src="{get_cover_url(row['Cover_URL'])}" style="width:130px;height:200px;border-radius:8px;">
        <h4 style="color:white;margin:8px 0 2px;">{row['Book']}</h4>
        <p style="color:#aaa;font-size:13px;">{row['Author']}</p>
        <p style="color:#FFD700;">⭐ {row.get('Avg_Rating','N/A')}</p>
        <a href="{row['URL']}" target="_blank" style="color:#FF4B4B;">View on Goodreads →</a>
    </div>
    """, unsafe_allow_html=True)

# GLOBAL STYLES
st.markdown("""
<style>
.glow {font-size:48px;text-align:center;color:#FF4B4B;font-weight:700;}
.sub {text-align:center;color:#aaa;font-size:18px;margin-bottom:30px;}
.book-card {
    background:rgba(255,255,255,0.06);
    padding:12px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.25);
    transition:0.3s;
}
.book-card:hover {
    transform:scale(1.08);
    box-shadow:0 0 25px rgba(255,80,80,0.55);
}
</style>

<div class="glow">📚 Nova Books Recommender</div>
<div class="sub">We Wish to Recommend Your Desired Books 🧡</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    page = st.radio("Navigate", [
        "Search by Book",
        "Search by Author",
        "Search by Genre",
        "Recent Searches",
        "About Us"
    ])

# PAGES
if page == "Search by Book":
    book = st.selectbox("Select Book", [""] + list(original_df["Book"]))
    if book:
        recs = get_similar_books(book)
        cols = st.columns(5)
        for i, r in recs.iterrows():
            with cols[i]:
                render_goodreads_card(r)

elif page == "Search by Author":
    author = st.selectbox("Select Author", [""] + list(original_df["Author"].unique()))
    if author:
        books = get_author_books(author)
        cols = st.columns(5)
        for i, r in books.head(5).iterrows():
            with cols[i]:
                render_goodreads_card(r)

elif page == "Search by Genre":
    genre = st.selectbox("Select Genre", [""] + popular_genres)
    if genre:
        books = get_genre_books(genre)
        cols = st.columns(5)
        for i, r in books.head(5).iterrows():
            with cols[i]:
                render_goodreads_card(r)

elif page == "Recent Searches":
    st.info("Feature unchanged")

elif page == "About Us":
    st.markdown("### Muhammad Haris Afridi")
    st.write("Nova Books is a machine-learning powered recommender system built with love for books and AI.")

st.caption("© Built by Muhammad Haris Afridi | Streamlit + Hugging Face")
