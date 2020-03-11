from sklearn.feature_extraction.text import CountVectorizer
from dirty_cat import SimilarityEncoder
from sklearn.model_selection import train_test_split
import numpy as np


class FitCountVectorizer:
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, df, n_tokens=100):
        self.vectorizer = CountVectorizer(max_features=n_tokens)
        self.vectorizer.fit(df[self.col_name])

    def transform(self, df):
        return self.vectorizer.transform(df[self.col_name]).toarray()

class FitSimilarityEncoder:
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, df, similarity="ngram", categories="most_frequent", n_prototypes=100):
        # Initialaze the similarity encoder
        self.similarity_encoder = SimilarityEncoder(
            similarity=similarity,
            dtype=np.float32,
            categories=categories,
            n_prototypes=n_prototypes,
            random_state=1006
        )

        # Fit the similarity encoder
        self.similarity_encoder.fit(df[self.col_name].values.reshape(-1, 1))

    def transform(self, df):
        return self.similarity_encoder.transform(df[self.col_name].values.reshape(-1, 1))


def create_train_test_set(df, y, test_size):
    return train_test_split(
        df, y,
        train_size=1 - test_size,
        test_size=test_size,
        stratify=y,
        random_state=1006,
    )



