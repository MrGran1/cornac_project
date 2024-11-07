import numpy as np
from cornac.models import Recommender
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SBERT(Recommender):
    def __init__(self, name="SBERT", model_name='paraphrase-MiniLM-L6-v2', k=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_name = model_name  # SBERT model name
        self.k = k  # Number of recommendations to return
        self.model = SentenceTransformer(self.model_name)  # Load SBERT model


    def fit(self, train_set, val_set=None):
        """Fit the SBERT model to the dataset."""
        # Initialize base Recommender
        Recommender.fit(self, train_set, val_set)

        # Ensure item texts are available in the train set
        if not hasattr(train_set, 'item_text') or train_set.item_text is None:
            raise ValueError("train_set must contain item_text with text data for items.")

        # Extract item texts to generate embeddings
        n_items = train_set.num_items
        item_texts = train_set.item_text.batch_seq(np.arange(n_items))
        print(train_set.item_text.batch_seq())
        # Generate item embeddings using SBERT
        self.item_embeddings = self.model.encode(item_texts.tolist())
        return self

    def score(self, user_id, item_idx=None):
        """Calculate scores for items based on cosine similarity with the user's preferences."""
        # Get user feedback: two lists (item IDs and ratings)
        user_history = self.train_set.item_data[user_id]
        if not user_history or len(user_history[0]) == 0:
            return np.zeros(self.train_set.num_items)

        # Extract item IDs and ratings
        item_ids, ratings = user_history
        ratings = np.array(ratings)

        # Get embeddings for items the user has interacted with
        item_embeddings = np.array([self.item_embeddings[item] for item in item_ids])

        # Calculate weighted user embedding (weighted by ratings)
        user_embedding = np.average(item_embeddings, axis=0, weights=ratings)

        # Compute similarity between user embedding and all item embeddings
        scores = cosine_similarity([user_embedding], self.item_embeddings)[0]
        return scores[item_idx]

