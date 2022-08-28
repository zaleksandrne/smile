import pickle
from pathlib import Path
from typing import Union, Dict, Callable

import numpy as np
import pandas as pd
from dask import dataframe as dd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from .preprocessing import date_processing, vector_processing, urls_processing, quantitative_processing


def get_random_message():
    messages = [
        'Oops, it will be too much, run something simpler mode, for example "time"',
        'Maybe a little later, for now you can run something simpler mode, for example "time"',
    ]
    return np.random.choice(messages)


class SmileModel:
    _DEFAULT_MODE_OPTIONS: list = ['time', 'score', 'very_score', 'super_score']

    def __init__(self, mode: str, sequence_bin_path: str = None, verbose: bool = False):
        """

        :param mode: a mode for data preparing. For now it use only "time" mode
        :param sequence_bin_path: It can be, but it is not using
        :param verbose: if True, to print logs
        """
        if mode not in self._DEFAULT_MODE_OPTIONS:
            raise ValueError(f'Wrong mode: {mode}. Must be one of: {self._DEFAULT_MODE_OPTIONS}')
        
        if mode in {'score', 'very_score'}:
            msg = get_random_message()
            raise ValueError(msg)

        if mode == 'super_score':
            raise ValueError('Sorry, but we have not done it... Maybe next time :)')

        self._mode: str = mode
        self._sequence_bin_path = sequence_bin_path

        self._model: XGBClassifier = XGBClassifier()
        self._train_columns = None

        self._verbose = verbose

    def _log(self, log: str):
        if self._verbose:
            print(log)

    def with_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise Exception(f'The model by path "{model_path}" is not exists')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self._model = model

        # Yeh. These are out best (ha-ha) features, which we did not dropped
        self._train_columns = [
            'year', 'month', 'day', 'day_of_week',
            'tokens_len_source', 'urls_len_source', 'tokens_sum_source', 'urls_sum_source',
            'urls_lower', 'urls_middle', 'urls_up', 'urls_single',
            'url_cnt_1', 'url_cnt_2', 'url_cnt_3', 'url_cnt_4', 'url_cnt_5', 'url_cnt_6', 'url_cnt_7', 'url_cnt_8', 'url_cnt_9', 'url_cnt_10',
            'url_freq_1', 'url_freq_2', 'url_freq_3', 'url_freq_4', 'url_freq_5', 'url_freq_6', 'url_freq_7', 'url_freq_8', 'url_freq_9', 'url_freq_10'
        ]

    def check_input_data(self, X: dd.DataFrame):
        input_columns = X.columns.tolist()
        if extra_columns := ({'CLIENT_ID', 'RETRO_DT', 'tokens', 'urls_hashed'} - set(input_columns)):
            raise Exception('Are you sure, that your X has all required column. '
                            f'It looks like it miss {extra_columns}')

    def _update_data_dict(self, df_: dd.DataFrame, features: Dict[str, dd.Series]):
        for key, feature in features.items():
            df_[key] = feature
        del features

        return df_

    def prepare_input_data_time(self, X: dd.DataFrame):
        self._log('Time preprocessing is starting were converted into tuple of tuple types')
        self.check_input_data(X)  # raise Exception if it is not validate

        # from str to tuple of tuples
        # Yeah. I really did not want to do this. But it was too late, and I really
        # wanted to sleep. Sorry...
        tokens = X['tokens'].compute()
        tokens = vector_processing.to_tuples(tokens)
        X = X.drop('tokens', axis=1).compute()

        X['urls_hashed'] = vector_processing.to_tuples(X['urls_hashed'])
        self._log('Tokens were converted into tuple of tuple types')

        # # Generate date features
        date_features: Dict[str, dd.Series] = date_processing.generate_features(X['RETRO_DT'])
        self._update_data_dict(X, features=date_features)
        self._log('Date features were taken')

        # # Generate count and freq features
        quantitative_features = quantitative_processing.generate_quantitative_features(X, tokens)
        self._update_data_dict(X, features=quantitative_features)
        self._log('Quantitative features were taken')

        del tokens  # it is too heavy right now. Dask is beatiful

        # Generate range features
        ranges: Dict[str, dd.Series] = urls_processing.generate_ranges(X['urls_hashed'])
        self._update_data_dict(X, features=ranges)
        self._log('Range features were taken')

        # Generate urls hash tf-idf features
        dfs = urls_processing.create_tfidf_urls(X['urls_hashed'])
        X = pd.concat([X.reset_index(drop=True)] + list(dfs.values()), axis=1)

        self._log('Urls hash frequencies features were taken\n')

        return X

    def prepare_input_data_score(self, X: dd.DataFrame):
        X = self.prepare_input_data_time(X)
        return X

    def prepare_input_data_very_score(self, X: dd.DataFrame):
        X = self.prepare_input_data_score(X)
        return X

    def prepare_input_data_super_score(self, X: dd.DataFrame):
        X = self.prepare_input_data_score(X)
        return X

    def _get_prepare_methods(self) -> Dict[str, Callable]:
        return dict(zip(self._DEFAULT_MODE_OPTIONS,
                        [self.prepare_input_data_time, self.prepare_input_data_score,
                         self.prepare_input_data_very_score, self.prepare_input_data_super_score]))

    def fit(self, X: dd.DataFrame, y: Union[dd.Series, np.ndarray]):
        df_ = X.reset_index(drop=True)
        prepare_methods = self._get_prepare_methods()

        df_prepared = prepare_methods[self._mode](df_)
        df_prepared = df_prepared.drop(['CLIENT_ID', 'RETRO_DT', 'urls_hashed'], axis=1)

        self._log(f'Before fitting. Features: {df_prepared.columns.tolist()}')

        self._train_columns = df_prepared.columns.tolist()
        self._model.fit(df_prepared, y)
        self._log(f'Complete fit. score: {round(self._score(df_prepared, y), 3)}')

        del df_prepared

        return self

    def predict(self, X):
        proba: np.ndarray = self.predict_proba(X)

        return (proba[:, 1] > 0.5).astype('uint8')

    def predict_proba(self, X) -> np.ndarray:
        df_ = X  # it is just I do not like X name
        prepare_methods = self._get_prepare_methods()

        df_prepared: pd.DataFrame = prepare_methods[self._mode](df_)
        return self.predict_proba(df_prepared)

    def _predict_proba(self, df_prepared: pd.DataFrame):

        if miss_columns := set(self._train_columns) - set(df_prepared.columns):
            raise ValueError('It is impossible predict. The columns different from train data columns. '
                             f'Missed columns: {miss_columns}')

        return self._model.predict_proba(df_prepared[self._train_columns])

    def score(self, X, y_true):
        proba = self.predict_proba(X)

        return roc_auc_score(y_true, proba[:, 1])

    def _score(self, df_prepared: pd.DataFrame, y_true):
        """
        When you have prepared data
        :param df_prepared:
        :param y_true:
        :return:
        """
        proba = self._predict_proba(df_prepared)

        return roc_auc_score(y_true, proba[:, 1])
