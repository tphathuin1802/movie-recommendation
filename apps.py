import pickle
import sys

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Add src directory to path for imports
sys.path.append("src")
from reccomendation import get_recommendations


def load_data():
    try:
        with open("data/processed/processed_data.pkl", "rb") as f:
            tfidf_matrix, tfidf = pickle.load(f)

        movies_df = pd.read_csv("data/raw/movies.csv")
        return tfidf_matrix, movies_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def get_movie_poster(poster_path):
    """Get movie poster URL from TMDB"""
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image+Available"


def display_recommendations(recommendations):
    """Display movie recommendations in a grid"""
    cols = st.columns(min(5, len(recommendations)))
    for i, (col, (_, row)) in enumerate(zip(cols, recommendations.iterrows())):
        with col:
            st.image(get_movie_poster(row["poster_path"]), width=150)
            st.markdown(f"**{row['title']}**")
            release_date = (
                row["release_date"].split("-")[0]
                if pd.notna(row["release_date"])
                else "Unknown"
            )
            st.caption(f"Released: {release_date}")


def main():
    st.set_page_config(layout="wide")
    st.write(
        "<h2> <b style='color:red; text-align: center; font-size: 30px;'> Movies Recommendation System </b> </h2>",
        unsafe_allow_html=True,
    )
    st.write(
        "This is a movie recommender system that suggests films "
        "similar to your favorites."
    )
    st.write("Author: Huynh Tan Phat")
    st.link_button("Visit my website", "https://stephen-huynh.vercel.app/")

    # Load data
    tfidf_matrix, movies_df = load_data()

    if tfidf_matrix is None or movies_df is None:
        st.error("Failed to load movie data. Please check data files.")
        return

    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get list of movie titles
    movie_titles = movies_df["title"].tolist()

    # Create movie selection dropdown
    selected_movie = st.selectbox("Tap to Select a Movie  🌐️", movie_titles)

    if st.button("Show Recommendations"):
        with st.spinner("Finding similar movies..."):
            # Get recommendations
            recommendations = get_recommendations(
                selected_movie, cosine_sim, movies_df, top_n=10
            )

            # Display recommendations
            st.subheader("You might also like:")
            display_recommendations(recommendations)


if __name__ == "__main__":
    main()
