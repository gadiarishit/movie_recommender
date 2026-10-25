"""Wraps your pickled data-analysis artifacts (df, indices, TF-IDF vectorizer & matrix)."""
import difflib
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent

# These genres were split on spaces in your data - glue them back together.
_MULTIWORD = {"Science Fiction": "Science_Fiction", "TV Movie": "TV_Movie"}


def _load(name):
    with open(DATA_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def _genre_list(s) -> list[str]:
    s = str(s or "")
    for k, v in _MULTIWORD.items():
        s = s.replace(k, v)
    return [g.replace("_", " ") for g in s.split() if g]


class Recommender:
    def __init__(self):
        df = _load("df").reset_index(drop=True)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        df["genre_list"] = df["genres"].apply(_genre_list)
        df["title_lc"] = df["title"].fillna("").str.lower().str.strip()
        self.df = df
        self.matrix = _load("vector_matrix")      # (n_movies, 50000) sparse TF-IDF
        self.vectorizer = _load("vectorizer")
        # Duplicate titles exist -> map each title to its most popular row.
        best = df.sort_values("popularity", ascending=False).drop_duplicates("title_lc")
        self.title_to_pos = dict(zip(best["title_lc"], best.index))
        self._all_titles = list(self.title_to_pos.keys())
        self.genres = sorted({g for gl in df["genre_list"] for g in gl})

    # ---------- listing ----------
    def popular(self, n=20, offset=0):
        return self.df.sort_values("popularity", ascending=False).iloc[offset: offset + n]

    def by_genre(self, genre, n=20, offset=0, sort="Popularity"):
        sub = self.df[self.df["genre_list"].apply(lambda g: genre in g)]
        col = "popularity" if sort == "Popularity" else "vote_average"
        sub = sub.sort_values(col, ascending=False)
        return sub.iloc[offset: offset + n], len(sub)

    # ---------- search ----------
    def find_titles(self, query, limit=8):
        """Positions of movies whose title matches the query (exact > contains > fuzzy)."""
        q = query.lower().strip()
        if not q:
            return []
        if q in self.title_to_pos:
            exact = [self.title_to_pos[q]]
        else:
            exact = []
        contains = [p for t, p in self.title_to_pos.items() if q in t and p not in exact]
        contains.sort(key=lambda p: -self.df.at[p, "popularity"])
        out = exact + contains
        if len(out) < limit:
            for t in difflib.get_close_matches(q, self._all_titles, n=5, cutoff=0.6):
                p = self.title_to_pos[t]
                if p not in out:
                    out.append(p)
        return out[:limit]

    def recommend(self, pos, n=15):
        """Content-based: cosine similarity on TF-IDF (rows are L2-normalised)."""
        sims = self.matrix[pos].dot(self.matrix.T).toarray().ravel()
        sims[pos] = -1
        # skip other entries with the same title (duplicates)
        sims[self.df["title_lc"].values == self.df.at[pos, "title_lc"]] = -1
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        res = self.df.iloc[top].copy()
        res["similarity"] = sims[top]
        return res

    def search_by_text(self, text, n=15):
        """Fallback: treat the query as a plot description and match it."""
        vec = self.vectorizer.transform([text])
        if vec.nnz == 0:
            return self.df.iloc[0:0]
        sims = self.matrix.dot(vec.T).toarray().ravel()
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        res = self.df.iloc[top].copy()
        res["similarity"] = sims[top]
        return res[res["similarity"] > 0]
