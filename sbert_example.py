"""Example to run SBERT model with Ratio Split evaluation strategy"""

import cornac
from cornac.datasets import citeulike,movielens,amazon_clothing
from cornac.eval_methods import RatioSplit
from cornac.models import SBERT
from cornac.data import Reader
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer
# Load the MovieLens 100K dataset
plots, movie_ids = movielens.load_plot()

# movies without plots are filtered out by `cornac.data.Reader`
ml_100k = movielens.load_feedback(reader=Reader(item_set=movie_ids))

# Instantiate an evaluation method.

item_text_modality = TextModality(corpus=plots, ids=movie_ids, 
                                  tokenizer=BaseTokenizer(sep='\\t', stop_words='english'),
                                  max_vocab=5000, max_doc_freq=0.5).build()

ratio_split = RatioSplit(data=ml_100k, test_size=0.9,
                         item_text=item_text_modality,
                         exclude_unknowns=True, 
                         verbose=True,
                         seed=123)
# Instantiate a sbert recommender model.
sbert = SBERT(k=10)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
cornac.Experiment(
    eval_method=ratio_split,
    models=[sbert],
    verbose=True,
    metrics=[mae, rmse, rec_20, pre_20],
    user_based=True,
).run()
