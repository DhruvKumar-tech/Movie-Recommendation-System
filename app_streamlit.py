import streamlit as st
import numpy as np

from model_utils import (
    load_movie_data,
    preprocess_genres,
    train_genre_word2vec,
    compute_genre_embeddings,
    build_embedding_matrix,
    recommend_movies,
)


@st.cache_data
def load_data():
    df = load_movie_data("movies.csv")
    df = preprocess_genres(df)
    return df


@st.cache_resource
def train_model_and_build_embeddings(df):
    # Train skip-gram model on genres
    w2v_model = train_genre_word2vec(
        df,
        vector_size=50,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=200,  # you can tune this
    )

    df_emb = compute_genre_embeddings(df, w2v_model)
    emb_matrix = build_embedding_matrix(df_emb)
    return df_emb, emb_matrix


def main():
    st.title("Movie Recommendation System (Genre-based, Skip-gram Word2Vec)")
    st.markdown(
        """
        This app recommends movies based on **genre similarity** using a skip‑gram Word2Vec 
        model trained on the dataset's genres.
        """
    )

    df = load_data()
    df_emb, emb_matrix = train_model_and_build_embeddings(df)

    movie_titles = df_emb["title"].sort_values().unique().tolist()
    default_movie = "Toy Story (1995)" if "Toy Story (1995)" in movie_titles else movie_titles[0]

    st.sidebar.header("Select Movie")
    selected_title = st.sidebar.selectbox("Choose a movie you like:", movie_titles, index=movie_titles.index(default_movie))
    top_k = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10, step=1)

    if st.sidebar.button("Get Recommendations"):
        try:
            recs = recommend_movies(df_emb, emb_matrix, selected_title, top_k=top_k)
            st.subheader(f"Movies similar to: **{selected_title}**")
            st.dataframe(recs.reset_index(drop=True))
        except ValueError as e:
            st.error(str(e))

    st.markdown("---")
    st.markdown(
        """
        **How it works:**  
        - Genres for each movie are tokenized (e.g., `Comedy`, `Drama`, `Romance`).  
        - A **skip‑gram Word2Vec** model is trained on these genre sequences.  
        - Each genre gets a learned embedding, and a movie vector is the average of its genres.  
        - Cosine similarity between movie vectors is used to find the most similar titles.
        """
    )


if __name__ == "__main__":
    main()
