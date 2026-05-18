"""Optional FastAPI endpoint for CineMatch AI."""

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import prepare_recommender_data  # noqa: E402
from src.recommender import MovieRecommender  # noqa: E402


app = FastAPI(
    title="CineMatch AI Recommendation API",
    description=(
        "API sencilla para consultar recomendaciones de películas a partir "
        "de un título usando filtrado colaborativo basado en ítems."
    ),
    version="0.1.0",
)

recommender = None
summary = None


@app.on_event("startup")
def startup_event():
    """Load the recommender once when the API starts."""
    global recommender, summary
    movies, _, _, similarity_df, summary = prepare_recommender_data(
        data_dir=PROJECT_ROOT / "data"
    )
    recommender = MovieRecommender(similarity_df=similarity_df, movies_df=movies)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "CineMatch AI Recommendation API"}


@app.get("/recommend")
def recommend(title: str = Query(..., min_length=1), top_n: int = Query(5, ge=1, le=10)):
    """Recommend movies from a title query."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="El recomendador no esta cargado.")

    try:
        selected_movie, recommendations = recommender.recommend_by_title(
            title_query=title,
            top_n=top_n,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "query": title,
        "selected_movie": selected_movie,
        "recommendations": recommendations,
    }
