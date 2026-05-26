import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from model_utils import (
    load_movie_data,
    preprocess_genres,
    train_genre_word2vec,
    compute_genre_embeddings,
    build_embedding_matrix,
    recommend_movies,
)

# Load configuration values from local .env environment layer
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(
    page_title="Find Your Next Movie", 
    page_icon="🎬", 
    layout="wide"
)

# Custom card container styling with a clean dark theme
st.markdown("""
    <style>
    .movie-card {
        background-color: #1e1e2f;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
        border: 1px solid #3a3a52;
        min-height: 180px;
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.2);
    }
    .movie-title {
        color: #ff4b4b;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .movie-meta {
        color: #a0a0b8;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .movie-overview {
        color: #e0e0e6;
        font-size: 14px;
        display: -webkit-box;
        -webkit-line-clamp: 5;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = load_movie_data("movies.csv")
    df = preprocess_genres(df)
    return df


@st.cache_resource
def train_model_and_build_embeddings(df):
    w2v_model = train_genre_word2vec(df, vector_size=50, epochs=100)
    w2v_embeddings = compute_genre_embeddings(df, w2v_model)
    count_matrix = build_embedding_matrix(df)
    return w2v_embeddings, count_matrix


def get_tmdb_movie_details(title_str):
    """Fetches high-quality metadata and ratings from the TMDb API safely."""
    fallback_data = {
        "overview": "",
        "rating": ""
    }
    if not TMDB_API_KEY:
        return fallback_data
        
    try:
        # Extract title cleanly by splitting out trailing bracketed year tags
        clean_title = title_str.split(" (")[0].strip()
        
        # Handle inverted naming patterns in datasets (e.g., "Bulls, The")
        if ", The" in clean_title:
            clean_title = "The " + clean_title.replace(", The", "").strip()
        elif ", A" in clean_title:
            clean_title = "A " + clean_title.replace(", A", "").strip()
        elif ", An" in clean_title:
            clean_title = "An " + clean_title.replace(", An", "").strip()

        url = "https://themoviedb.org"
        params = {"api_key": TMDB_API_KEY, "query": clean_title, "language": "en-US"}
        
        res = requests.get(url, params=params, timeout=5).json()
        if res.get("results"):
            best_match = res["results"][0]  
            overview_text = best_match.get("overview", "").strip()
            vote_avg = best_match.get("vote_average", 0)
            
            return {
                "overview": overview_text,
                "rating": f"{round(vote_avg, 1)} / 10" if vote_avg > 0 else ""
            }
    except Exception:
        pass
    return fallback_data


def render_movie_grid(recs):
    """Renders a clean 4-column textual card matrix completely free of image components."""
    cols_per_row = 4
    for i in range(0, len(recs), cols_per_row):
        row_data = recs.iloc[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        
        for idx, (_, movie) in enumerate(row_data.iterrows()):
            with cols[idx]:
                meta = get_tmdb_movie_details(movie['title'])
                
                # Format text parameters dynamically based on data availability
                rating_element = f"⭐ Score: {meta['rating']} | " if meta['rating'] else ""
                genres_element = movie['genres'].replace('|', ' • ')
                overview_element = f'<div class="movie-overview">{meta["overview"]}</div>' if meta['overview'] else ""
                
                # Render only the text description panel card
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-meta">{rating_element}{genres_element}</div>
                        {overview_element}
                    </div>
                """, unsafe_allow_html=True)


def main():
    # --- Front Page Title Section ---
    st.title("🎬 Find Your Next Movie")
    st.markdown("##### *Discover tailored recommendations matched precisely to your content taste profile.*")
    st.write("---")

    if "df" not in st.session_state:
        st.session_state.df = load_data()

    df = st.session_state.df
    if df.empty:
        st.warning("Please verify your local database source parameters.")
        return

    w2v_embeddings, count_matrix = train_model_and_build_embeddings(df)

    # Initialize a session state token to toggle between Title and Genre searches smoothly
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "movie"

    # ==========================================
    #     SIDEBAR SEARCH WINDOW CONTROLLERS
    # ==========================================

    # --- Section 1: Live Search by Movie Title ---
    st.sidebar.markdown("### 🎬 Search by Movie")
    movie_titles = df["title"].sort_values().unique().tolist()
    
    # FIX: Add a blank option to the front of the list
    dropdown_options = ["-- Select a Movie --"] + movie_titles
    
    selected_title = st.sidebar.selectbox(
        "Select a film you enjoyed:", 
        dropdown_options, 
        index=0, 
        label_visibility="collapsed"
    )
    
    if st.sidebar.button("Search", key="refresh_title"):
        st.session_state.search_mode = "movie"
        st.rerun()

    st.sidebar.markdown("---")

    # --- Section 2: Live Filter by Genre Mixes ---
    st.sidebar.markdown("### 📽️ Search by Genre Filter")
    all_genres = set()
    for g_string in df['genres'].dropna():
        all_genres.update(g_string.split('|'))
    sorted_genres = sorted(list(all_genres))
    
    selected_genres = st.sidebar.multiselect(
        "Pick matching genre tags:", 
        sorted_genres, 
        placeholder="Choose genres..."
    )
    
    if st.sidebar.button("Search", key="refresh_genres"):
        st.session_state.search_mode = "genre"
        st.rerun()

    st.sidebar.markdown("---")

    # --- Section 3: Global Settings (Pinned to the Bottom) ---
    st.sidebar.markdown("### ⚙️ Global Settings")
    engine_choice = st.sidebar.radio(
        "Matching Logic Type:", 
        ("Vibe Match (Conceptual)", "Direct Match (Strict Genre Alignment)")
    )
    top_k = st.sidebar.slider("How many options do you want?", 4, 16, 8, step=4)

    # Determine automated active mode route based on input focus activity
    if selected_genres:
        st.session_state.search_mode = "genre"
    else:
        st.session_state.search_mode = "movie"

    # ==========================================
    #             AUTOMATED ROUTING ENGINE
    # ==========================================
    
    # Check if a true movie has been selected from the options list
    if st.session_state.search_mode == "movie" and selected_title != "-- Select a Movie --":
        selected_matrix = w2v_embeddings if engine_choice == "Vibe Match (Conceptual)" else count_matrix
        try:
            recs_by_name = recommend_movies(df, selected_matrix, selected_title, top_k=top_k)
            if not recs_by_name.empty:
                st.subheader(f"✨ Curated Selection Based on: {selected_title}")
                render_movie_grid(recs_by_name)
        except Exception as e:
            st.error(f"Execution tracking issue: {str(e)}")

    elif st.session_state.search_mode == "genre" and selected_genres:
        mask = df['genres'].apply(lambda x: all(g in str(x).split('|') for g in selected_genres) if pd.notna(x) else False)
        filtered_df = df[mask].copy()
        
        if not filtered_df.empty:
            st.subheader(f"✨ Custom Collection Matching: {', '.join(selected_genres)}")
            render_movie_grid(filtered_df.head(top_k))
        else:
            st.error("No matches found for that exact combination of genres.")
            
    else:
        # Shows a welcoming instruction state if nothing is selected or if the app is refreshed
        st.info("👈 Use the sidebar to find suggestions either by choosing a movie or selecting a custom genre profile mix!")


if __name__ == "__main__":
    main()
