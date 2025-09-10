import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("data/tfidf.joblib")
    kmeans = joblib.load("data/kmeans_8.joblib")
    df = pd.read_pickle("data/zomato_with_clusters.pkl")
    return tfidf, kmeans, df

tfidf, kmeans, df = load_artifacts()

# Infer column names
name_col = [c for c in ['restaurant_name','restaurant','name'] if c in df.columns]
rating_col = 'aggregate_rating' if 'aggregate_rating' in df.columns else None

# ---------- Helpers ----------
def clean_q(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\\S+|www\\.\\S+", " ", s)
    s = re.sub(r"[^a-z\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def top_terms_per_cluster(tfidf, kmeans, k=15):
    terms = np.array(tfidf.get_feature_names_out())
    centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    return {c: terms[centroids[c, :k]].tolist() for c in range(kmeans.n_clusters)}

TOP_TERMS = top_terms_per_cluster(tfidf, kmeans, 15)

def recommend_from_text(query: str, top_n=10):
    Xq = tfidf.transform([clean_q(query)])
    c = kmeans.predict(Xq)
    sub = df[df['cluster'] == c].copy()
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
    st.write("Enter cravings or a short review snippet, and the app finds similar restaurants from the most relevant cluster.")
    st.write("Artifacts are precomputed for fast responses.")
    st.write("Built for Streamlit Community Cloud.")

tab_rec, tab_clusters, tab_vis = st.tabs(["Recommendations", "Cluster keywords", "Quick visuals"])

with tab_rec:
    q = st.text_input("Describe what is desired (e.g., 'spicy biryani with tender chicken'):", "")
    topn = st.slider("How many suggestions?", 5, 20, 10)
    if st.button("Recommend") and q.strip():
        c, recs = recommend_from_text(q, top_n=topn)
        st.subheader(f"Cluster {c}")
        st.write("Top keywords: " + ", ".join(TOP_TERMS[c]))
        st.dataframe(recs.reset_index(drop=True), use_container_width=True)

with tab_clusters:
    st.subheader("Top keywords per cluster")
    data = [{"cluster": c, "keywords": ", ".join(kw)} for c, kw in TOP_TERMS.items()]
    st.dataframe(pd.DataFrame(data), use_container_width=True)

with tab_vis:
    st.subheader("PCA snapshot (optional)")
    show_pca = st.checkbox("Render PCA scatter (slower for large data)", value=False)
    if show_pca:
        from sklearn.decomposition import PCA
        N = min(3000, len(df))
        idx = np.random.RandomState(42).choice(len(df), size=N, replace=False)
        X_sample = tfidf.transform(df.iloc[idx]['clean_text'] if 'clean_text' in df.columns else df.iloc[idx][name_col].astype(str))
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X_sample.toarray())
        labs = df.iloc[idx]['cluster'].to_numpy()
        import matplotlib.pyplot as plt
        import matplotlib
        fig, ax = plt.subplots(figsize=(6,4))
        sc = ax.scatter(X2[:,0], X2[:,1], c=labs, s=6, cmap="tab10", alpha=0.8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA projection")
        st.pyplot(fig)
