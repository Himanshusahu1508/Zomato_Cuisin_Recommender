from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------- Paths ----------
ROOT = Path(__file__).parent.resolve()
TFIDF_PATH = ROOT / "tfidf.joblib"
KMEANS_PATH = ROOT / "kmeans_8.joblib"
DF_PATH     = ROOT / "zomato_with_clusters.pkl"

# ---------- Load artifacts (cached) ----------
@st.cache_resource(show_spinner="Loading artifacts...")
def load_artifacts():
    tfidf = joblib.load(TFIDF_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    df = pd.read_pickle(DF_PATH)
    return tfidf, kmeans, df

tfidf, kmeans, df = load_artifacts()

# ---------- Column detection & guards ----------
def pick_first_present(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        st.error(f"Required column missing. Expecting one of: {candidates}. Found: {list(df.columns)[:10]} ...")
        st.stop()
    return None

name_col   = pick_first_present(df, ['restaurant_name','restaurant','name'], required=True)
rating_col = pick_first_present(df, ['aggregate_rating','rating','rate'], required=False)
text_col   = 'clean_text' if 'clean_text' in df.columns else None
cluster_col = pick_first_present(df, ['cluster'], required=True)

# Ensure numeric ratings if present (handles '4.1/5' etc.)
if rating_col is not None:
    df[rating_col] = (df[rating_col].astype(str)
                      .str.extract(r'(\d+\.?\d*)', expand=False)
                      .astype(float))

# ---------- Precompute cluster keywords ----------
def compute_top_terms(tfidf, kmeans, topn=15):
    terms = np.array(tfidf.get_feature_names_out())
    centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    return {c: terms[centroids[c, :topn]].tolist() for c in range(kmeans.n_clusters)}

TOP_TERMS = compute_top_terms(tfidf, kmeans, 15)

# ---------- Utils ----------
def clean_q(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def recommend_from_text(query: str, top_n=10):
    Xq = tfidf.transform([clean_q(query)])
    c = kmeans.predict(Xq)[0]
    sub = df[df[cluster_col] == c].copy()
    if rating_col:
        sub = sub.sort_values(rating_col, ascending=False)
    cols = [name_col] + ([rating_col] if rating_col else [])
    return c, sub[cols].head(top_n)

# ---------- UI ----------
st.set_page_config(page_title="Zomato Cuisine Clustering", layout="wide")
st.title("Zomato Cuisine Clustering Recommender")
st.caption("TF‑IDF + K‑Means over review text to discover cuisine/experience themes and suggest similar restaurants.")

with st.sidebar:
    st.header("About")
    st.write("- Enter cravings or a short review snippet to get similar restaurants drawn from the most relevant cluster.")
    st.write("- Artifacts are precomputed and cached for quick responses.")
    st.write("- Columns detected:")
    st.code({
        "name_col": name_col,
        "rating_col": rating_col,
        "text_col": text_col,
        "cluster_col": cluster_col
    })

tab_rec, tab_clusters, tab_vis = st.tabs(["Recommendations", "Cluster keywords", "Quick visuals"])

# ----- Recommendations tab -----
with tab_rec:
    q = st.text_input("Describe what is desired (e.g., 'spicy biryani with tender chicken'):", "")
    topn = st.slider("How many suggestions?", 5, 20, 10)
    if st.button("Recommend") and q.strip():
        c, recs = recommend_from_text(q, top_n=topn)
        st.subheader(f"Cluster {c}")
        st.write("Top keywords: " + ", ".join(TOP_TERMS.get(c, [])))
        st.dataframe(recs.reset_index(drop=True), use_container_width=True)

# ----- Cluster keywords tab -----
with tab_clusters:
    st.subheader("Top keywords per cluster")
    data = [{"cluster": c, "keywords": ", ".join(kw)} for c, kw in TOP_TERMS.items()]
    st.dataframe(pd.DataFrame(data), use_container_width=True)

# ----- Visuals tab -----
with tab_vis:
    st.subheader("PCA snapshot (optional)")
    st.write("Projects TF‑IDF to 2D with PCA for a qualitative ‘point view’. Expect some overlap due to dimensionality.")
    show_pca = st.checkbox("Render PCA scatter", value=False)
    if show_pca:
        # Build a lightweight sample for plotting
        N = min(3000, len(df))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), size=N, replace=False)

        # Prefer clean text if present; otherwise fallback to name (not ideal but works)
        if text_col:
            texts = df.iloc[idx][text_col].astype(str)
        else:
            texts = df.iloc[idx][name_col].astype(str)

        X_sample = tfidf.transform(texts)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        # Convert only the sample to dense for PCA
        X2 = pca.fit_transform(X_sample.toarray())
        labs = df.iloc[idx][cluster_col].to_numpy()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,4))
        sc = ax.scatter(X2[:,0], X2[:,1], c=labs, s=6, cmap="tab10", alpha=0.85, edgecolor='none')
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA projection of TF‑IDF (colored by cluster)")
        legend = ax.legend(*sc.legend_elements(), title="cluster", loc="best", markerscale=2)
        st.pyplot(fig)

st.caption("Deployed with Streamlit Community Cloud. If artifacts fail to load, ensure tfidf.joblib, kmeans_8.joblib, and zomato_with_clusters.pkl sit at the repo root and match scikit‑learn/joblib versions pinned in requirements.txt.")
