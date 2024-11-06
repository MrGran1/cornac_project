import numpy as np
from cornac.models import Recommender
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SBERT(Recommender):
    def __init__(self, name="SBERTRecommender", model_name='paraphrase-MiniLM-L6-v2', k=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_name = model_name  # SBERT model name
        self.k = k  # Number of recommendations to return
        self.model = SentenceTransformer(self.model_name)  # Load SBERT model

    def fit(self, train_set):
        """Compute embeddings for all items."""
        super().fit(train_set)
        self.item_texts = [train_set.item_text(i) for i in range(train_set.num_items)]
        self.item_embeddings = self.model.encode(self.item_texts)
        return self

    def score(self, user_id, item_id=None):
        """Calculate scores for items based on cosine similarity with the user's preferences."""
        user_history = self.train_set.user_feedback(user_id)
        if not user_history:
            return np.zeros(self.train_set.num_items)

        # Get embeddings for items the user has interacted with
        user_embedding = np.mean([self.item_embeddings[item] for item in user_history], axis=0)
        scores = cosine_similarity([user_embedding], self.item_embeddings)[0]
        return scores

    def recommend(self, user_id, k=None):
        """Recommend top-k items for a user."""
        k = k or self.k
        scores = self.score(user_id)
        recommended_items = np.argsort(-scores)[:k]  # Top k items
        return recommended_items

