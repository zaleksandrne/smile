from typing import Union, Dict, Callable

import numpy as np
import pandas as pd
from dask import dataframe as dd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from .preprocessing import date_processing, vector_processing, urls_processing, quantitative_processing


class SmileModel:
    _DEFAULT_MODE_OPTIONS: list = ['time', 'score', 'very_score', 'super_score']

    def __init__(self, mode: str, sequence_bin_path: str = None, verbose: bool = False):
        if mode not in self._DEFAULT_MODE_OPTIONS:
            raise Exception(f'Wrong mode: {mode}. Must be one of: {self._DEFAULT_MODE_OPTIONS}')

        if mode == 'super_score':
            raise Exception('Sorry, but we have not done it... Maybe next time :)')

        self._mode: str = mode
        self._sequence_bin_path = sequence_bin_path

        self._model: XGBClassifier = XGBClassifier()

        self._verbose = verbose

    def _log(self, log: str):
        if self._verbose:
            print(log)

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
        # tokens = vector_processing.to_tuples(X['tokens'])
        X = X.drop('tokens', axis=1).compute()

        X['urls_hashed'] = vector_processing.to_tuples(X['urls_hashed'])
        self._log('Tokens were converted into tuple of tuple types')

        # # Generate date features
        date_features: Dict[str, dd.Series] = date_processing.generate_features(X['RETRO_DT'])
        self._update_data_dict(X, features=date_features)
        self._log('Date features were taken')

        # # Generate count and freq features
        # quantitative_features = quantitative_processing.generate_quantitative_features(X, tokens)
        # self._update_data_dict(X, features=quantitative_features)
        # self._log('Quantitative features were taken')

        # Generate range features
        ranges: Dict[str, dd.Series] = urls_processing.generate_ranges(X['urls_hashed'])
        self._update_data_dict(X, features=ranges)
        self._log('Range features were taken')

        # Generate urls hash tf-idf features
        dfs = urls_processing.create_tfidf_urls(X['urls_hashed'])
        X = pd.concat([X.reset_index(drop=True)] + list(dfs.values()), axis=1)

        self._log('Urls hash frequencies features were taken')
        self._log('')

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
        self._model.fit(df_prepared, y)

        del df_prepared

        return self

    def predict(self, X):
        proba: np.ndarray = self.predict_proba(X)

        return (proba[:, 1] > 0.5).astype('uint8')

    def predict_proba(self, X) -> np.ndarray:
        df_ = X  # it is just I do not like X name
        prepare_methods = self._get_prepare_methods()

        df_prepared = prepare_methods[self._mode](df_)
        proba = self._model.predict_proba(df_prepared)

        return proba

    def score(self, X, y_true):
        proba = self.predict_proba(X)

        return roc_auc_score(y_true, proba[:, 1])
