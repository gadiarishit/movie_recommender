# 🎬 CineMatch – Movie Recommender

A content-based movie recommendation web app built with Python and Streamlit.

**Live app:** https://movierecommender-excfcx9zkzyjyjtsh8g5qq.streamlit.app

<!-- Add a screenshot: save it as screenshot.png in this folder, then uncomment the next line -->
<!-- ![CineMatch screenshot](screenshot.png) -->

## Features

- **Popular movies** shown on the home page when nothing is searched
- **Search a movie** to get similar ones, ranked by a similarity score
- **Genre bar** to browse movies by genre, sorted by popularity or rating
- **Posters and links** fetched live from the TMDB API

## How it works

1. Each movie's overview, tagline and genres were cleaned into a text field (`tags`).
2. A TF-IDF vectorizer (50,000 features) turned every movie into a vector.
3. For a searched movie, cosine similarity against all 45,447 movies gives the closest matches.
4. Weak matches are filtered out with a minimum similarity threshold.
5. If the search text matches no title, it is treated as a plot description and matched the same way.

## Tech stack

Python · Streamlit · pandas · scikit-learn · SciPy · TMDB API

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `recommender.py` | Loads the model files and computes recommendations |
| `tmdb_api.py` | TMDB API client for posters and links |
| `df.pkl`, `vector_matrix.pkl`, `vectorizer.pkl` | Movie data and TF-IDF model from the analysis |

## Run locally

```bash
pip install -r requirements.txt
```

Create a `.env` file next to `app.py` with your free [TMDB API key](https://www.themoviedb.org/settings/api):

```
TMDB_API_KEY=your_key_here
```

```bash
streamlit run app.py
```

## Notes

- Recommendations are based on plot text and genres only, so results can be weak for movies with little unique text.
- Data source: movie metadata dataset (45k movies); posters and links from TMDB.

*This product uses the TMDB API but is not endorsed or certified by TMDB.*
