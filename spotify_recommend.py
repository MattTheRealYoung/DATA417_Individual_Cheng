import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def check_name_in_library(artists, library_name):
    result= []
    for item in library_name:
        if artists in item:
            result.append(0)
        else:
            result.append(1)
    return result

def get_diversity_scores(similarity_scores):
    result = []
    for item in similarity_scores:
        result.append(0.5 * item)
    return result

song_library = pd.read_csv("dataset.csv")
user_history = song_library.iloc[58, 8 : 18].values
singer = song_library.iloc[58]['artists']

# Calculate similarity scores
similarity_scores = cosine_similarity([user_history], song_library.iloc[0:, 8: 18])

# Apply diversification penalty
diversification_penalty = get_diversity_scores(similarity_scores)

# Introduce bonus
bonus = 0.2 * np.array(check_name_in_library(singer, song_library['artists'].values))

# Final scores
final_scores = np.array(similarity_scores[0]) - np.array(diversification_penalty[0]) + bonus

song_library['score'] = final_scores.tolist()

# Sort and recommend
recommended_songs = song_library.sort_values(by='score', ascending=False)

# Print recommended songs
print(recommended_songs)
