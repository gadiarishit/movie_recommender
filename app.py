"""Movie recommender UI.   Run:  streamlit run app.py"""
import streamlit as st

from recommender import Recommender
from tmdb_api import API_KEY, fetch_many

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
.sub {color: #8b93a3; margin-bottom: 1rem;}
.mtitle {font-weight: 600; font-size: .95rem; margin-top: .4rem; line-height: 1.2;}
.meta {color: #8b93a3; font-size: .8rem;}
.badge {background:#e50914; color:#fff; border-radius:6px; padding:1px 7px; font-size:.75rem;}
[data-testid="stImage"] img {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading movie data…")
def get_model():
    return Recommender()


@st.cache_data(show_spinner=False, ttl=86400)
def posters(titles: tuple):
    return fetch_many(list(titles))


model = get_model()
PAGE = 20
COLS = 5


def show_grid(df, show_sim=False):
    """Render movies as a poster grid."""
    if df.empty:
        st.info("No movies found.")
        return
    infos = posters(tuple(df["title"]))
    rows = list(zip(df.to_dict("records"), infos))
    for i in range(0, len(rows), COLS):
        cols = st.columns(COLS)
        for col, (m, info) in zip(cols, rows[i:i + COLS]):
            with col:
                st.image(info["poster"])
                year = f" ({info['year']})" if info["year"] else ""
                st.markdown(f"<div class='mtitle'>{m['title']}{year}</div>", unsafe_allow_html=True)
                extra = f" · <span class='badge'>{m['similarity']*100:.0f}% match</span>" if show_sim else ""
                st.markdown(f"<div class='meta'>⭐ {m['vote_average']:.1f} · {', '.join(m['genre_list'][:2])}{extra}</div>",
                            unsafe_allow_html=True)
                with st.expander("Overview"):
                    st.write(m["overview"] or "No overview available.")
                    if info["tmdb_url"]:
                        st.markdown(f"[Open on TMDB]({info['tmdb_url']})")


# ---------------- header & search ----------------
st.markdown("<div class='title'>🎬 CineMatch</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Search a movie you like and get similar ones.</div>", unsafe_allow_html=True)
if not API_KEY:
    st.warning("TMDB_API_KEY not found - posters will be placeholders. Put it in a `.env` file next to app.py.")

query = st.text_input("Search", placeholder="Search a movie, e.g. Inception, Toy Story…",
                      label_visibility="collapsed")

# ---------------- genre bar ----------------
genre = st.pills("Genre", ["All"] + model.genres, default="All", label_visibility="collapsed")
genre = genre or "All"

# ---------------- main views ----------------
if query.strip():
    # 1) SEARCH -> use the TF-IDF similarity model
    matches = model.find_titles(query)
    if matches:
        labels = {p: f"{model.df.at[p, 'title']}" for p in matches}
        pos = matches[0]
        if len(matches) > 1:
            pos = st.selectbox("Did you mean…", matches, format_func=lambda p: labels[p])
        movie = model.df.loc[pos]
        st.subheader(f"Because you searched for “{movie['title']}”")
        recs = model.recommend(pos, n=15)
        if genre != "All":
            recs = recs[recs["genre_list"].apply(lambda g: genre in g)]
            st.caption(f"Filtered to {genre}")
        show_grid(recs, show_sim=True)
    else:
        st.subheader(f"No title matched “{query}” - movies matching that description")
        show_grid(model.search_by_text(query), show_sim=True)

elif genre != "All":
    # 2) GENRE BAR -> movies of that genre from your data
    c1, c2 = st.columns([3, 1])
    sort = c2.selectbox("Sort by", ["Popularity", "Rating"])
    key = f"shown_{genre}_{sort}"
    st.session_state.setdefault(key, PAGE)
    page, total = model.by_genre(genre, n=st.session_state[key], sort=sort)
    c1.subheader(f"{genre} movies ({total:,})")
    show_grid(page)
    if st.session_state[key] < total:
        if st.button("Load more"):
            st.session_state[key] += PAGE
            st.rerun()

else:
    # 3) NOTHING SEARCHED -> popular movies
    st.subheader("🔥 Popular right now")
    show_grid(model.popular(n=20))
