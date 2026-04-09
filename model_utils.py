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

from gensim.models import Word2Vec


def train_genre_word2vec(
    df,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=200
):
    """
    Train Word2Vec Skip-Gram model on movie genres.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing 'genres' column.

    vector_size : int
        Embedding size.

    window : int
        Context window size.

    min_count : int
        Minimum word frequency.

    workers : int
        CPU cores for training.

    sg : int
        1 = Skip-Gram, 0 = CBOW.

    epochs : int
        Number of training iterations.

    Returns:
    -------
    Word2Vec model
    """

    # Convert genres into tokenized sentences
    genre_sentences = []

    for genres in df['genres']:
        if isinstance(genres, str):
            tokens = genres.split('|')   # split genres like Action|Comedy
            genre_sentences.append(tokens)

    # Train Word2Vec
    model = Word2Vec(
        sentences=genre_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )

    return model

import numpy as np


def compute_genre_embeddings(df, w2v_model):
    """
    Compute averaged genre embeddings for each movie.

    Parameters:
    ----------
    df : pandas.DataFrame
        Must contain 'genres' column.

    w2v_model : gensim Word2Vec model
        Trained genre embedding model.

    Returns:
    -------
    numpy.ndarray
        Array of movie genre embeddings.
    """

    movie_embeddings = []

    for genres in df['genres']:

        if isinstance(genres, str):

            genre_tokens = genres.split('|')

            vectors = []

            for genre in genre_tokens:
                if genre in w2v_model.wv:
                    vectors.append(w2v_model.wv[genre])

            if len(vectors) > 0:
                avg_vector = np.mean(vectors, axis=0)

            else:
                avg_vector = np.zeros(w2v_model.vector_size)

        else:
            avg_vector = np.zeros(w2v_model.vector_size)

        movie_embeddings.append(avg_vector)

    return np.array(movie_embeddings)


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
