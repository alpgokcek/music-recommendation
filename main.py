import pandas as pd
import numpy as np
import json
from NaiveBayesClassifier import NaiveBayesClassifier

# CONSTANTS
LIKE, DISLIKE = 1, 0
N_TRACKS = 100

"""
** Importing the dataset and extracting features **
In this part, the aim is to import the dataset and extract the features in it.
"""

DATASET_PATH = './misc/data.csv'
data_df = pd.read_csv(DATASET_PATH)

data_df.head()

features = data_df.drop(
    labels=['artists', 'id', 'name', 'year', 'release_date'], axis=1).to_numpy()

features[:5]

features.shape

"""
** Normalizing the features of the dataset. **
In this part, the aim is to normalize the features using standard score formula.
"""


def normalize_features(arr):
    # standard score
    cols_mean, cols_stdev = arr.mean(axis=0), arr.std(axis=0)
    output = (arr - cols_mean) / cols_stdev
    return output


normalized_features = normalize_features(features)

"""
** Importing the user input. **
In this part, the aim is to import the user input.
"""

USER_FAVORITES_INPUT = './misc/out/mood-pop.json'
USER_DISLIKES_INPUT = './misc/out/rock-classics.json'

user_favorites_ids, user_dislikes_ids = None, None
with open(USER_FAVORITES_INPUT, 'r') as f:
    user_favorites_ids = json.load(f)

with open(USER_DISLIKES_INPUT, 'r') as f:
    user_dislikes_ids = json.load(f)


def get_track_features(idx_arr):
    track_features = []
    for idx in idx_arr['trackIds']:
        track = normalized_features[data_df['id'].str.contains(idx, na=False)]
        if len(track) > 0:
            track_features.append(track)
    return np.array(track_features)


user_favorites = get_track_features(user_favorites_ids)
user_dislikes = get_track_features(user_dislikes_ids)

# extracting samples and its labels
samples = np.concatenate((user_favorites, user_dislikes), axis=0)
samples = samples.reshape(samples.shape[0], -1)
labels = np.array([LIKE for _ in range(len(user_favorites))] +
                  [DISLIKE for _ in range(len(user_dislikes))])


"""
** Testing **
In this part, the aim is to test the algorithm.
"""

nb = NaiveBayesClassifier(samples, list(labels))
predictions = nb.predict(normalized_features)

highest_probabilities = sorted(
    enumerate(predictions), key=lambda tup: tup[1], reverse=True)
like_predictions = sorted(highest_probabilities, key=lambda tup: list(tup[1])[
                          1], reverse=True)[:N_TRACKS]

# printing predictions
for idx, (p, c) in like_predictions:
    print("probability: \t{}\n{}\n".format(p, data_df.iloc[idx]))


"""
** Sklearn Naive Bayes **
In this part, the aim is to run the NB algorithm of Sklearn on the dataset.


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(samples, labels)
predictions = clf.predict_proba(normalized_features)

count = 0
for idx, p in enumerate(predictions):
  if p[1] >= p[0]:
    count += 1
print(count)
"""
