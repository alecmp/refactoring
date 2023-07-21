from .base import AbstractDataset

import pandas as pd

from datetime import date


class AmazonReviewsDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'amazon-reviews'   

    def load_ratings_df(self):
        df = pd.read_csv('data/ratings.csv', header=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        mv_df = pd.read_csv('data/products.csv', header=0)
        mv_df.columns = ['sid', 'title', 'categories']
        mv_df = mv_df[["sid", "categories"]]
        df = pd.merge(df, mv_df)
        return df