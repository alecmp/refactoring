from .base import AbstractDataset

import pandas as pd

from datetime import date


class AmazonReviewsDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'amazon-reviews'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
        

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    @classmethod
    def all_raw_file_names(cls):
        return ['genome-scores.csv',
                'genome-tags.csv',
                'links.csv',
                'movies.csv',
                'ratings.csv',
                'README.txt',
                'tags.csv']

    def load_ratings_df(self):
        #folder_path = Path('alecmp/KeBERT4Rec/data/amazon_us_product.zip') 
        #file_path = folder_path.joinpath('rating_50k.csv') #ratings
        df = pd.read_csv('data/ratings.csv', header=0)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        #c_file_path = folder_path.joinpath('output_50kk.csv') #movies
        mv_df = pd.read_csv('data/products.csv', header=0)
        mv_df.columns = ['sid', 'title', 'categories']
        mv_df = mv_df[["sid", "categories"]]
        df = pd.merge(df, mv_df)
        return df