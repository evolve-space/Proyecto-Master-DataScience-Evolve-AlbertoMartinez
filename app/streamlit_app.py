"""Streamlit demo for CineMatch AI."""

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import prepare_recommender_data  # noqa: E402
from src.recommender import MovieRecommender, recommendations_to_dataframe  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"


@st.cache_resource(show_spinner="Preparando matriz de similitud...")
def load_recommender():
    movies, _, _, similarity_df, summary = prepare_recommender_data(data_dir=DATA_DIR)
    recommender = MovieRecommender(similarity_df=similarity_df, movies_df=movies)
    return recommender, summary


st.set_page_config(
    page_title="CineMatch AI",
    layout="wide",
)

st.title("CineMatch AI")
st.caption("Sistema de recomendación inteligente basado en patrones de valoración")

with st.sidebar:
    st.header("Configuración")
    top_n = st.radio("Número de recomendaciones", options=[5, 10], horizontal=True)
    st.markdown(
        "El sistema usa filtrado colaborativo basado en ítems. "
        "Compara películas según patrones históricos de valoración de los usuarios."
    )

try:
    recommender, summary = load_recommender()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Coloca los archivos `ratings.csv` y `movies.csv` en la carpeta `data/` "
        "y vuelve a ejecutar la app."
    )
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Ratings usados", f"{summary['filtered_ratings']:,}")
metric_cols[1].metric("Películas", f"{summary['filtered_movies']:,}")
metric_cols[2].metric("Usuarios activos", f"{summary['active_users']:,}")
metric_cols[3].metric("Densidad matriz", f"{summary['matrix_density']:.2%}")

st.divider()

search_query = st.text_input(
    "Busca una película",
    placeholder="Ejemplo: Scarface, Toy Story, Matrix, Star Wars...",
)

if search_query:
    matches = recommender.search_movies(search_query, max_results=20)

    if not matches:
        st.warning("No se encontraron películas disponibles con ese título.")
        st.stop()

    selected_movie = st.selectbox(
        "Selecciona una película",
        options=matches,
        format_func=lambda movie: f"{movie['title']} | {movie['genres']}",
    )

    recommendations = recommender.recommend_by_id(
        selected_movie["movie_id"],
        top_n=top_n,
    )

    st.subheader(f"Recomendaciones para {selected_movie['title']}")
    st.write(f"Géneros base: `{selected_movie['genres']}`")

    recommendations_df = recommendations_to_dataframe(recommendations)
    st.dataframe(recommendations_df, hide_index=True, use_container_width=True)
else:
    st.info("Escribe parte del título de una película para generar recomendaciones.")

st.divider()

st.subheader("Cómo funciona")
st.write(
    "El prototipo construye una matriz usuario-película a partir de ratings. "
    "Después calcula la similitud del coseno entre películas: dos películas son "
    "más similares si reciben patrones de valoración parecidos por parte de los "
    "usuarios. Al seleccionar una película, se muestran las más cercanas en esa "
    "matriz de similitud."
)
