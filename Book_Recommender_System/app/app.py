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
import requests  # Added for downloading from Hugging Face

# PAGE CONFIG
st.set_page_config(
    page_title="Nova Books Recommender",
    page_icon="📚",
    layout="wide"
)

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

# LOAD ARTIFACTS - CLEAN & SILENT LOADING
@st.cache_resource
def load_artifacts():
    file_url = "https://huggingface.co/harisyar/nova-books-recommender/resolve/main/tfidf_features.pkl.gz"
    local_path = "models/tfidf_features.pkl.gz"
    
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(local_path):
        with st.spinner("Loading book model for the first time (14MB)... This takes just a moment."):
            response = requests.get(file_url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
        st.success("✅ Model ready! Enjoy discovering books.", icon="📚")
    
    with gzip.open(local_path, "rb") as f:
        artifacts = pickle.load(f)
    
    df = artifacts.get("df")
    tfidf_matrix = artifacts.get("tfidf_matrix")
    if df is None or tfidf_matrix is None:
        raise ValueError("tfidf_features.pkl.gz must contain 'df' and 'tfidf_matrix'")
    
    df = df.set_index("Book", drop=False)
    return df, tfidf_matrix

original_df, tfidf_matrix = load_artifacts()

if "Cover_URL" not in original_df.columns:
    st.error("Your dataframe must contain a 'Cover_URL' column.")
    st.stop()

# HELPERS
@st.cache_data
def get_cover_url(cover_value):
    if pd.notna(cover_value) and str(cover_value).strip() not in ['', 'nan']:
        return str(cover_value).strip()
    return "https://dryofg8nmyqjw.cloudfront.net/images/no-cover.png"

@st.cache_data(ttl=3600)
def top_two_genres(genres_field):
    if genres_field is None:
        return []
    if isinstance(genres_field, (list, tuple)):
        return list(genres_field)[:2]
    s = str(genres_field).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)[:2]
    except Exception:
        pass
    for sep in ["|", ",", ";"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return parts[:2]
    return [s] if s else []

# RECOMMENDERS
@st.cache_data
def get_similar_books(book_title, n=5):
    if book_title not in original_df.index:
        return original_df.sample(n, random_state=42).reset_index(drop=True)
    idx = original_df.index.get_loc(book_title)
    sim_vector = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    top_indices = np.argsort(sim_vector)[::-1][1:n+1]
    return original_df.iloc[top_indices].reset_index(drop=True)

@st.cache_data
def get_genre_books(genre_name, n=50):
    genre_books = original_df[original_df["Genres"].str.contains(genre_name, case=False, na=False)].copy()
    if len(genre_books) == 0:
        return original_df.sample(min(n, len(original_df)), random_state=42).reset_index(drop=True)
    if 'Avg_Rating' in genre_books.columns:
        genre_books = genre_books.sort_values('Avg_Rating', ascending=False)
    return genre_books.head(n).reset_index(drop=True)

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

# BOOK CARD STYLE - ORIGINAL
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

# STYLES - TRUE HORIZONTAL SCROLL, FULL WIDTH, NO EMPTY SPACE
st.markdown(
    """
    <style>
    .glow {font-size:48px; text-align:center; color:#FF4B4B; font-weight:700;
           animation: glow 2.5s ease-in-out infinite;}
    @keyframes glow {0%,100% {text-shadow:0 0 5px #FF4B4B;} 50% {text-shadow:0 0 25px #FF4B4B;}}
    .sub {text-align:center; color:#aaa; font-size:18px; margin-bottom:30px;}

    .book-card {
        background:rgba(255,255,255,0.05);
        padding:12px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 4px 12px rgba(0,0,0,0.2);
        transition:all 0.3s ease;
        cursor:pointer;
        flex: 0 0 auto;
        width: 160px; /* Fixed card width for consistency */
        margin: 0 8px;
    }
    .book-card:hover {
        transform:scale(1.08) translateY(-5px);
        box-shadow:0 0 25px rgba(255,80,80,0.55);
    }

    /* True horizontal carousel - full width, no wrapping */
    .cards-container {
        display: flex;
        overflow-x: auto;
        overflow-y: hidden;
        gap: 0; /* Gap handled by card margin */
        padding: 10px 0 40px 0;
        width: 100%;
        -webkit-overflow-scrolling: touch;
        touch-action: pan-x;
        scroll-snap-type: x mandatory;
    }
    .cards-container > div {
        flex: 0 0 auto;
        scroll-snap-align: start;
    }
    .cards-container::-webkit-scrollbar {
        height: 10px;
    }
    .cards-container::-webkit-scrollbar-thumb {
        background: #FF4B4B;
        border-radius: 10px;
    }

    /* Mobile - smaller text only */
    @media (max-width: 768px) {
        .book-card h4 {font-size:14px !important;}
        .book-card p {font-size:12px !important;}
        .book-card a {font-size:12px !important;}
        .book-card small {font-size:11px !important;}
    }
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

# HELPER TO RENDER HORIZONTAL CARDS - NO COLUMNS, PURE DIVS
def render_horizontal_cards(df_batch):
    st.markdown('<div class="cards-container">', unsafe_allow_html=True)
    for _, row in df_batch.iterrows():
        render_goodreads_card(row)
    st.markdown('</div>', unsafe_allow_html=True)

# PAGES
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
            st.write(f"**Author:** {selected['Author']}")
            st.write(f"**Genres:** {', '.join(top_two_genres(selected['Genres']))}")
            st.write(f"**Rating:** ⭐ {selected.get('Avg_Rating', 'N/A')}")
            if pd.notna(selected.get('URL')):
                st.markdown(f"[View on Goodreads]({selected['URL']})")
        if st.button("SHOW ME 5 SIMILAR BOOKS", type="primary", use_container_width=True):
            with st.spinner("Finding similar books..."):
                recs = get_similar_books(book, n=5)
            st.subheader("Recommended Books")
            render_horizontal_cards(recs)

elif page == "Search by Author":
    st.header("Books by Author")
    author = st.selectbox("Select Author:", [""] + list(original_df["Author"].unique()))
    if author:
        books = get_author_books(author)
        total = len(books)
        page_num = st.session_state.author_page
        start = page_num * 10
        end = min(start + 10, total)
        batch = books.iloc[start:end]
        st.subheader(f"Books by {author} (Page {page_num + 1})")
        render_horizontal_cards(batch)
        
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if page_num > 0 and st.button("← Previous"):
                st.session_state.author_page -= 1
                st.rerun()
        with col_info:
            st.write(f"Showing {start + 1}–{end} of {total}")
        with col_next:
            if end < total and st.button("Next →"):
                st.session_state.author_page += 1
                st.rerun()

elif page == "Search by Genre":
    st.header("Browse by Genre")
    genre = st.selectbox("Select Genre:", [""] + popular_genres)
    if genre:
        books = get_genre_books(genre, n=50)
        total = len(books)
        page_num = st.session_state.genre_page
        start = page_num * 10
        end = min(start + 10, total)
        batch = books.iloc[start:end]
        st.subheader(f"Top Books in {genre} (Page {page_num + 1})")
        render_horizontal_cards(batch)
        
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if page_num > 0 and st.button("← Previous"):
                st.session_state.genre_page -= 1
                st.rerun()
        with col_info:
            st.write(f"Showing {start + 1}–{end} of {total}")
        with col_next:
            if end < total and st.button("Next →"):
                st.session_state.genre_page += 1
                st.rerun()

elif page == "Recent Searches":
    st.header("Recent Searches")
    if st.session_state.search_history:
        recent_books = original_df.loc[[t for t in st.session_state.search_history if t in original_df.index]]
        render_horizontal_cards(recent_books)
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.info("No recent searches yet.")

elif page == "About Us":
    st.markdown("<h2 style='color:#FF4B4B; font-size:24px; margin-bottom:20px;'>Muhammad Haris Afridi</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Hey there! I'm a self-taught developer with a deep love for books, AI, and building things that make life better.
    Nova Books is my passion project — a smart book recommender built from scratch using machine learning.
    I believe great books change lives, and I wanted to create a beautiful way to help people discover their next favorite read.
    """)
    st.markdown("<h3 style='color:#FF4B4B; font-size:20px; margin-top:30px;'>Project Info</h3>", unsafe_allow_html=True)
    st.markdown("""
    <span style='color:#FF6B6B;'>Nova Books</span> is a content-based recommender system using **TF-IDF + cosine similarity** on book title, author, description, and genres.
    It matches books with similar content — simple, fast, and accurate.
    This is an intermediate ML project focused on clean design and real usability.
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='color:#FF4B4B; font-size:20px; margin-top:30px;'>Connect With Me</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:18px; margin-top:25px; font-size:16px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg?color=181717" width="28" height="28">
            <strong>GitHub:</strong> <a href="https://github.com/harisyar-ai" target="_blank" style="color:#FF4B4B;">github.com/harisyar-ai</a>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg?color=0A66C2" width="28" height="28">
            <strong>LinkedIn:</strong> <a href="https://linkedin.com/in/muhammad-haris-afridi" target="_blank" style="color:#FF4B4B;">linkedin.com/in/muhammad-haris-afridi</a>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/gmail.svg?color=EA4335" width="28" height="28">
            <strong>Email:</strong> <a href="mailto:mharisyar.ai@gmail.com" style="color:#FF4B4B;">mharisyar.ai@gmail.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='margin-top:40px; color:#ccc; font-size:16px;'>Thank you for using Nova Books! Let's discover amazing stories together 📚🧡.</p>", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.caption("© Built by Muhammad Haris Afridi | Powered by Streamlit & Hugging Face")
