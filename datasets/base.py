from content_tokenizers import tokenizer_factory
from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.split = args.split
        self.content_ids = args.content_encoding_type
        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()  

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self.preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        df = self.load_ratings_df()
        df = self.make_implicit(df)
        df, tokenizers = self.content_to_id_list(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap,
                   'tokenizers': tokenizers}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def make_implicit(self, df):
        print('Turning into implicit ratings')
        df["rating"] = df["rating"].astype('int32')
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df


    def content_to_id_list(self, df):
        tokenizer = self._get_content_tokenizer_class()
        tokenizers = []
        if tokenizer:
            print("Converting content columns to list of ids")
            for c in df.columns:
                if c.startswith("c"):

                    content_tokenizer = tokenizer(df[c])
                    df[c] = content_tokenizer.convert_column_to_ids(df[c])
                    tokenizers.append(content_tokenizer)
        return df, tokenizers

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i for i, s in enumerate(set(df['sid']))}

        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df, user_count):
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            user2item_content = []
            for c in df.columns:
                if c.startswith("c"):
                    content2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')[c]))
                    user2item_content.append(content2items)
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = [items[:-2]], [items[-2:-1]], [items[-1:]]
                for c_maps in user2item_content:
                    content = c_maps[user]
                    train[user].append(content[:-2])
                    val[user].append(content[-2:-1])
                    test[user].append(content[-1:])
            return train,val,test

        elif self.args.split == 'holdout':
            print('Splitting')
            sys.exit()
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[                :-2*eval_set_size]
            val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
            test_user_index  = permuted_index[  -eval_set_size:                ]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df   = df.loc[df['uid'].isin(val_user_index)]
            test_df  = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def _get_content_tokenizer_class(self):
        return tokenizer_factory(self.args)

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}-content_id_type{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split, self.content_ids)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

    def get_preprocessed_dataset_as_pandas(self):

        dataset_path = self._get_preprocessed_dataset_path()
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        df = self.load_ratings_df()
        df = self.make_implicit(df)
        return df

