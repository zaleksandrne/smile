from typing import Tuple

from dask import dataframe as dd


def _l_sum_tokens_count(tokens: Tuple[Tuple[str, int]]):
    print('_l_sum_tokens_count', tokens)
    return sum(map(lambda v: v[1], tokens))


def generate_quantitative_features(df_: dd.DataFrame, tokens: dd.Series):
    # tokens - are really heavy
    res = {
        'tokens_len_source': tokens.apply(lambda v: len(v)),
        'urls_len_source': df_['urls_hashed'].apply(lambda v: len(v)),

        'tokens_sum_source': tokens.apply(_l_sum_tokens_count),
        'urls_sum_source': df_['tokens'].apply(_l_sum_tokens_count),
    }

    return res
