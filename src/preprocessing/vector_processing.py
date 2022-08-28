from typing import Union, Tuple

import pandas as pd
from dask import dataframe as dd


def _l_str_to_tuples(tokens: Union[str, Tuple[Tuple[str, int]]]):
    if pd.isna(tokens):
        return tuple()

    if isinstance(tokens, str):
        if not tokens or ' ' not in tokens:
            out = tuple()  # outliers
        else:
            tokens_cnt = tokens.split(' ')
            out = tuple(zip(tokens_cnt[::2], list(map(int, tokens_cnt[1::2]))))
    else:
        out = tokens

    # do it sure that the number has numeric type
    out = list(map(lambda v: (v[0], int(v[1])), out))

    # to sort it from high to low by count
    return tuple(sorted(out, key=lambda v: v[1], reverse=True))


def to_tuples(s: Union[dd.Series, pd.Series]) -> dd.Series:
    if isinstance(s, pd.Series):
        return s.apply(_l_str_to_tuples)
    else:
        return s.apply(_l_str_to_tuples, meta=(s.name, 'object'))
