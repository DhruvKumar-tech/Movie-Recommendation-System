import os
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from the .env file safely
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_tmdb_genre_mapping() -> dict:
    """Helper to convert TMDb numeric IDs to text strings."""
    if not TMDB_API_KEY:
        return {}
    url = f"https://themoviedb.org{TMDB_API_KEY}&language=en-US"
    try:
        res = requests.get(url, timeout=5).json()
        return {genre['id']: genre['name'] for genre in res.get('genres', [])}
    except Exception:
        return {}

def fetch_movie_from_internet(movie_title: str) -> dict:
    """Searches TMDb API for a movie title and returns a structured record."""
    if not TMDB_API_KEY:
        return None

    search_url = "https://themoviedb.org"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
        "language": "en-US"
    }

    try:
        response = requests.get(search_url, params=params, timeout=5)
        if response.status_code == 200 and response.json()['results']:
            movie_data = response.json()['results'][0] # Grab first result

            genre_mapping = fetch_tmdb_genre_mapping()
            genres = [genre_mapping.get(g_id, "Unknown") for g_id in movie_data.get('genre_ids', [])]
            genres_str = "|".join(genres) if genres else "Unknown"

            release_year = f" ({movie_data.get('release_date')[:4]})" if movie_data.get('release_date') else ""
            full_title = f"{movie_data.get('title')}{release_year}"

            return {
                "movieId": int(movie_data.get("id")),
                "title": full_title,
                "genres": genres_str,
                "genres_clean": genres_str.replace("|", " ").lower()
            }
    except Exception:
        return None
    return None


def load_movie_data(csv_path: str = "movies.csv") -> pd.DataFrame:
    """Loads and cleans the initial movie dataset."""
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback empty dataframe matching structure if CSV is missing on GitHub
        return pd.DataFrame(columns=["movieId", "title", "genres", "genres_clean"])

    data = data.dropna(subset=["title", "genres"])
    return data


def preprocess_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the raw genres pipe format into normalized text."""
    df = df.copy()
    df["genres_clean"] = (
        df["genres"].str.replace("|", " ", regex=False).str.lower()
    )
    return df


def fetch_tmdb_genre_mapping() -> dict:
    """Helper to convert TMDb numeric IDs to text strings."""
    if not TMDB_API_KEY:
        return {}
    url = f"https://themoviedb.org{TMDB_API_KEY}&language=en-US"
    try:
        res = requests.get(url, timeout=5).json()
        return {
            genre["id"]: genre["name"] for genre in res.get("genres", [])
        }
    except Exception:
        return {}


def fetch_movie_from_internet(movie_title: str) -> dict:
    """Searches TMDb API for a movie title and returns a structured record."""
    if not TMDB_API_KEY:
        return None

    search_url = "https://themoviedb.org"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title,
        "language": "en-US",
    }

    try:
        response = requests.get(search_url, params=params, timeout=5)
        if response.status_code == 200 and response.json()["results"]:
            movie_data = response.json()["results"][0]

            genre_mapping = fetch_tmdb_genre_mapping()
            genres = [
                genre_mapping.get(g_id, "Unknown")
                for g_id in movie_data.get("genre_ids", [])
            ]
            genres_str = "|".join(genres) if genres else "Unknown"

            release_year = (
                f" ({movie_data.get('release_date')[:4]})"
                if movie_data.get("release_date")
                else ""
            )
            full_title = f"{movie_data.get('title')}{release_year}"

            return {
                "movieId": int(movie_data.get("id")),
                "title": full_title,
                "genres": genres_str,
                "genres_clean": genres_str.replace("|", " ").lower(),
            }
    except Exception:
        return None
    return None


def train_genre_word2vec(
    df, vector_size=50, window=5, min_count=1, workers=4, sg=1, epochs=200
):
    """Trains a Word2Vec Skip-Gram model on movie genres."""
    genre_sentences = []
    for genres in df["genres"]:
        if isinstance(genres, str):
            tokens = genres.split("|")
            genre_sentences.append(tokens)

    model = Word2Vec(
        sentences=genre_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )
    return model


def compute_genre_embeddings(df, w2v_model) -> np.ndarray:
    """Computes averaged genre embeddings for each movie asset."""
    movie_embeddings = []
    for genres in df["genres"]:
        if isinstance(genres, str):
            genre_tokens = genres.split("|")
            vectors = [
                w2v_model.wv[genre]
                for genre in genre_tokens
                if genre in w2v_model.wv
            ]

            if len(vectors) > 0:
                avg_vector = np.mean(vectors, axis=0)
            else:
                avg_vector = np.zeros(w2v_model.vector_size)
        else:
            avg_vector = np.zeros(w2v_model.vector_size)

        movie_embeddings.append(avg_vector)

    return np.array(movie_embeddings)


def build_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    """Builds a dense CountVectorizer matrix array."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df["genres_clean"]).toarray()
    return matrix


def recommend_movies(
    df: pd.DataFrame, embeddings: np.ndarray, movie_title: str, top_k: int = 10
) -> pd.DataFrame:
    """Generates recommendations safely handling 2D target vector reshaping."""
    df = df.reset_index(drop=True)

    mask = df["title"].str.lower() == movie_title.lower()
    if not mask.any():
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    idx = df[mask].index[0]

    # Critical 2D Vector Reshape Fix
    query_vec = embeddings[idx].reshape(1, -1)

    sim_scores = cosine_similarity(query_vec, embeddings)[0]

    # Create safe rankings avoiding self-recommendation mapping anomalies
    sim_scores_mapped = sim_scores.copy()
    sim_scores_mapped[idx] = -1.0

    top_indices = np.argsort(sim_scores_mapped)[::-1][:top_k]

    results = df.loc[top_indices, ["movieId", "title", "genres"]].copy()
    results["similarity"] = sim_scores_mapped[top_indices]

    return results
