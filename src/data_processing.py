"""Utilidades para preparar el dataset de recomendaciones MovieLens."""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None


REQUIRED_RATINGS_COLUMNS = {"userId", "movieId", "rating"}
REQUIRED_MOVIES_COLUMNS = {"movieId", "title", "genres"}


def load_movielens_data(data_dir="data", ratings_file="ratings.csv", movies_file="movies.csv"):
    """Carga los archivos ratings y movies desde la carpeta local data/."""
    data_path = Path(data_dir)
    ratings_path = data_path / ratings_file
    movies_path = data_path / movies_file

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "No se encontraron ratings.csv y movies.csv en la carpeta data/. "
            "Descarga MovieLens y coloca ambos archivos en data/."
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    missing_ratings = REQUIRED_RATINGS_COLUMNS - set(ratings.columns)
    missing_movies = REQUIRED_MOVIES_COLUMNS - set(movies.columns)

    if missing_ratings:
        raise ValueError(f"Faltan columnas en ratings.csv: {sorted(missing_ratings)}")
    if missing_movies:
        raise ValueError(f"Faltan columnas en movies.csv: {sorted(missing_movies)}")

    return ratings, movies


def get_movie_rating_stats(ratings):
    """Calcula el número de ratings y la valoración media por película."""
    return ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()


def filter_ratings(ratings, min_movie_ratings=20, min_user_ratings=10):
    """Filtra películas populares y usuarios activos para mejorar la fiabilidad."""
    movie_stats = get_movie_rating_stats(ratings)
    popular_movie_ids = movie_stats.loc[
        movie_stats["count"] >= min_movie_ratings, "movieId"
    ]

    filtered_ratings = ratings[ratings["movieId"].isin(popular_movie_ids)].copy()

    user_counts = filtered_ratings["userId"].value_counts()
    active_user_ids = user_counts[user_counts >= min_user_ratings].index

    final_ratings = filtered_ratings[
        filtered_ratings["userId"].isin(active_user_ids)
    ].copy()

    return final_ratings


def build_user_movie_matrix(ratings):
    """Construye la matriz usuario-película."""
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0,
    )


def build_item_similarity_matrix(user_movie_matrix):
    """Calcula la similitud del coseno entre películas."""
    movie_user_matrix = user_movie_matrix.T
    matrix_values = movie_user_matrix.values

    if cosine_similarity is not None and csr_matrix is not None:
        matrix_values = csr_matrix(matrix_values)
        similarity_matrix = cosine_similarity(matrix_values)
    elif cosine_similarity is not None:
        similarity_matrix = cosine_similarity(matrix_values)
    else:
        norms = np.linalg.norm(matrix_values, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = matrix_values / norms
        similarity_matrix = normalized_matrix @ normalized_matrix.T

    movie_ids = movie_user_matrix.index.tolist()
    return pd.DataFrame(similarity_matrix, index=movie_ids, columns=movie_ids)


def build_dataset_summary(ratings, movies, filtered_ratings, user_movie_matrix, similarity_df):
    """Genera métricas descriptivas del dataset procesado."""
    total_possible = user_movie_matrix.shape[0] * user_movie_matrix.shape[1]
    non_zero = int((user_movie_matrix.values > 0).sum())
    density = non_zero / total_possible if total_possible else 0

    return {
        "original_ratings": int(len(ratings)),
        "original_movies": int(len(movies)),
        "original_users": int(ratings["userId"].nunique()),
        "filtered_ratings": int(len(filtered_ratings)),
        "filtered_movies": int(similarity_df.shape[0]),
        "active_users": int(filtered_ratings["userId"].nunique()),
        "matrix_density": float(density),
        "matrix_sparsity": float(1 - density),
    }


def prepare_recommender_data(
    data_dir="data",
    min_movie_ratings=20,
    min_user_ratings=10,
):
    """Ejecuta el pipeline completo de preparación del recomendador."""
    ratings, movies = load_movielens_data(data_dir=data_dir)
    filtered_ratings = filter_ratings(
        ratings,
        min_movie_ratings=min_movie_ratings,
        min_user_ratings=min_user_ratings,
    )
    user_movie_matrix = build_user_movie_matrix(filtered_ratings)
    similarity_df = build_item_similarity_matrix(user_movie_matrix)
    summary = build_dataset_summary(
        ratings,
        movies,
        filtered_ratings,
        user_movie_matrix,
        similarity_df,
    )

    return movies, filtered_ratings, user_movie_matrix, similarity_df, summary
