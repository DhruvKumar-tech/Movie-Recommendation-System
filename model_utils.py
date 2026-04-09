import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def load_movie_data(csv_path: str = "movies.csv") -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=["title", "genres"])
    return data


def preprocess_genres(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False).str.lower()
    return df


def build_embedding_matrix(df: pd.DataFrame):
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df["genres_clean"])
    return matrix


def recommend_movies(
    df: pd.DataFrame,
    embeddings,
    movie_title: str,
    top_k: int = 10,
) -> pd.DataFrame:

    df = df.reset_index(drop=True)

    mask = df["title"].str.lower() == movie_title.lower()
    if not mask.any():
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    idx = df[mask].index[0]
    query_vec = embeddings[idx]

    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    sim_scores[idx] = -1.0

    top_indices = np.argsort(sim_scores)[::-1][:top_k]

    results = df.loc[top_indices, ["movieId", "title", "genres"]].copy()
    results["similarity"] = sim_scores[top_indices]

    return results
