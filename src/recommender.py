"""Sistema de recomendación de películas basado en similitud entre ítems."""

import pandas as pd


class MovieRecommender:
    """Genera recomendaciones usando similitud del coseno entre películas."""

    def __init__(self, similarity_df, movies_df):
        self.similarity_df = similarity_df
        self.movies_df = movies_df.copy()
        self.available_movie_ids = set(similarity_df.index)

    def get_movie_info(self, movie_id):
        """Obtiene el título y géneros de una película."""
        movie = self.movies_df[self.movies_df["movieId"] == movie_id]
        if movie.empty:
            return None

        row = movie.iloc[0]
        return {
            "movie_id": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
        }

    def search_movies(self, title_query, max_results=10):
        """Busca películas disponibles por coincidencia parcial del título."""
        if not title_query:
            return []

        available_movies = self.movies_df[
            self.movies_df["movieId"].isin(self.available_movie_ids)
        ]
        matches = available_movies[
            available_movies["title"].str.contains(
                title_query,
                case=False,
                na=False,
                regex=False,
            )
        ].head(max_results)

        return [
            {
                "movie_id": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
            }
            for _, row in matches.iterrows()
        ]

    def recommend_by_id(self, movie_id, top_n=10):
        """Devuelve las películas más similares a la seleccionada."""
        if movie_id not in self.available_movie_ids:
            raise ValueError(f"La película con ID {movie_id} no está disponible.")

        similarities = self.similarity_df[movie_id].sort_values(ascending=False)
        similarities = similarities.drop(movie_id, errors="ignore").head(top_n)

        recommendations = []
        for similar_movie_id, score in similarities.items():
            movie_info = self.get_movie_info(similar_movie_id)
            if movie_info:
                movie_info["similarity_score"] = float(score)
                recommendations.append(movie_info)

        return recommendations

    def recommend_by_title(self, title_query, top_n=10):
        """Busca una película y genera recomendaciones asociadas."""
        matches = self.search_movies(title_query, max_results=1)
        if not matches:
            raise ValueError(f"No se encontraron películas con el título '{title_query}'.")

        selected_movie = matches[0]
        recommendations = self.recommend_by_id(selected_movie["movie_id"], top_n=top_n)

        return selected_movie, recommendations


def recommendations_to_dataframe(recommendations):
    """Convierte las recomendaciones en un DataFrame para visualización."""
    return pd.DataFrame(
        [
            {
                "Título": rec["title"],
                "Géneros": rec["genres"],
                "Score de similitud": round(rec["similarity_score"], 4),
            }
            for rec in recommendations
        ]
    )
