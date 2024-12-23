import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load data from text file
with open("E:/13届市调分析大赛(研究生)/nb的评语.txt", 'r',encoding='utf-8') as f:
    data = f.readlines()

# Create count vectorizer object
vectorizer = CountVectorizer()

# Fit and transform the data
X = vectorizer.fit_transform(data)

# Create PLSA model object
plsa_model = NMF(n_components=4, random_state=42)

# Fit the model
plsa_model.fit(X)

# Print the top words for each topic
feature_names = vectorizer.get_feature_names()
for topic_idx, topic in enumerate(plsa_model.components_):
    print("Topic %d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-10 - 1:-1]]),'\n')

# import numpy as np
# import pandas as pd
# from sklearn.decomposition import NMF
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer

# # Load data from text file
# with open("E:/13届市调分析大赛(研究生)/result.txt", 'r',encoding='utf-8') as f:
#     data = f.readlines()

# # Create count vectorizer object
# vectorizer = CountVectorizer()

# # Fit and transform the data
# X = vectorizer.fit_transform(data)

# # Define parameter grid for grid search
# param_grid = {'n_components': [5, 10, 15, 20],
#               'random_state': [42, 123, 456]}

# # Create NMF model object
# nmf_model = NMF()

# # Define custom scoring function
# def nmf_score(estimator, X):
#     """Return the negative reconstruction error of NMF"""
#     W = estimator.fit_transform(X)
#     H = estimator.components_
#     return -np.sum(X * np.log(W.dot(H)))

# # Create scorer object using custom scoring function
# scorer = make_scorer(nmf_score, greater_is_better=False)

# # Create grid search object
# grid_search = GridSearchCV(nmf_model, param_grid=param_grid, scoring=scorer)

# # Fit the grid search object to the data
# grid_search.fit(X)

# # Print the best hyperparameters
# print("Best hyperparameters: ", grid_search.best_params_)

# # Get the best NMF model from the grid search
# best_nmf_model = grid_search.best_estimator_

# # Print the top words for each topic
# feature_names = vectorizer.get_feature_names()
# for topic_idx, topic in enumerate(best_nmf_model.components_):
#     print("Topic #%d:" % topic_idx)
#     print(" ".join([feature_names[i]
#                     for i in topic.argsort()[:-10 - 1:-1]]))
