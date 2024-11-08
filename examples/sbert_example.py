"""Example to run Probabilistic Matrix Factorization (PMF) model with Ratio Split evaluation strategy"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import SBERT


# Load the MovieLens 100K dataset
ml_100k = movielens.load_feedback()

# Instantiate an evaluation method.
ratio_split = RatioSplit(
    data=ml_100k, test_size=0.2, rating_threshold=4.0, exclude_unknowns=False
)

# Instantiate a PMF recommender model.
sbert = SBERT(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
cornac.Experiment(
    eval_method=ratio_split,
    models=[sbert],
    metrics=[mae, rmse, rec_20, pre_20],
    user_based=True,
).run()
