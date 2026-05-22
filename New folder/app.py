import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Recommendation system using K-Means Clustering + Cosine Similarity")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

movies = load_data()

# -----------------------------
# COMBINE FEATURES
# -----------------------------
movies["combined"] = (
    movies["genre"] + " " +
    movies["keywords"] + " " +
    movies["rating"].astype(str)
)

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(movies["combined"])

# -----------------------------
# K-MEANS CLUSTERING
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
movies["cluster"] = kmeans.fit_predict(features)

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
similarity = cosine_similarity(features)

# -----------------------------
# MOVIE RECOMMENDATION FUNCTION
# -----------------------------
def recommend(movie_name):

    if movie_name not in movies['title'].values:
        return []

    index = movies[movies['title'] == movie_name].index[0]

    scores = list(enumerate(similarity[index]))

    sorted_scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in sorted_scores[1:6]:
        recommendations.append(
            movies.iloc[i[0]]["title"]
        )

    return recommendations

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Settings")

selected_movie = st.sidebar.selectbox(
    "Choose a movie",
    movies["title"].values
)

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
st.subheader("🎯 Recommended Movies")

recommendations = recommend(selected_movie)

for movie in recommendations:
    st.success(movie)

# -----------------------------
# DATASET TABLE
# -----------------------------
st.subheader("📊 Movie Dataset")

st.dataframe(movies)

# -----------------------------
# PCA VISUALIZATION
# -----------------------------
pca = PCA(n_components=2)

reduced_features = pca.fit_transform(
    features.toarray()
)

plot_df = pd.DataFrame({
    "x": reduced_features[:, 0],
    "y": reduced_features[:, 1],
    "title": movies["title"],
    "cluster": movies["cluster"].astype(str)
})

fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["title"],
    title="Movie Clusters Visualization"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# CLUSTER DETAILS
# -----------------------------
st.subheader("📌 Cluster Details")

for cluster_num in sorted(
    movies["cluster"].unique()
):

    st.write(f"### Cluster {cluster_num}")

    cluster_movies = movies[
        movies["cluster"] == cluster_num
    ]

    st.write(
        cluster_movies[
            ["title", "genre", "rating"]
        ]
    )