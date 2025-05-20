import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations(title, cosine_sim, movies_df, top_n=5):
    idx = movies_df.index[movies_df["title"] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df.iloc[movie_indices][["title", "release_date", "poster_path"]]
