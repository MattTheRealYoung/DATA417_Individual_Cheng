import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Check if the names are in the library, artist attribute can take multiple singers'name
def check_name_in_library(artists, library_name):
    result= []
    for item in library_name:
        found = False
        for artist in artists:
            if artist in item:
                result.append(0)
                found = True
                break
        if not found:
            result.append(1)
    return result

def get_diversity_scores(similarity_scores):
    result = []
    for item in similarity_scores:
        result.append(0.5 * item)
    return result

#Get similarity score between listened songs and song library and calculate the average similiarity score
def get_similarity_scores(user_history, song_library):
    result = []
    similarity = cosine_similarity(user_history, song_library)
    for song_index in range(len(song_library)):
        total = 0
        for index in range(len(user_history)):
            total += similarity[index][song_index]
        result.append(total / len(user_history))
    return result

def recommend_songs(user_history, singers, recommendNumber, deleteHistory = False):
    # Calculate similarity scores
    similarity_scores = get_similarity_scores(user_history, song_library.iloc[0:, 8:18])

    # Apply diversification penalty
    diversification_penalty = get_diversity_scores(similarity_scores)

    # Introduce bonus
    bonus = 0.2 * np.array(check_name_in_library(singers, song_library['artists'].values))

    # Final scores
    final_scores = np.array(similarity_scores[0]) - np.array(diversification_penalty[0]) + bonus

    song_library['score'] = final_scores.tolist()

    # Sort and recommend
    recommended_songs = song_library.sort_values(by='score', ascending=False)
    if deleteHistory:
        user_history.clear()
    if(recommendNumber > 0):
        return recommended_songs.iloc[0:recommendNumber]
    else:
        return recommended_songs

song_library = pd.read_csv("dataset.csv")
user_history = [song_library.iloc[58, 8 : 18].values, song_library.iloc[67, 8 : 18].values]
singers = [song_library.iloc[58]['artists'],  song_library.iloc[67]['artists']]

top_recommend_songs = recommend_songs(user_history, singers, 5, True)

# Print recommended songs
print(top_recommend_songs)
