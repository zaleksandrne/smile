from typing import Dict

import pandas as pd
from dask import dataframe as dd


def generate_ranges(urls: dd.Series) -> Dict[str, dd.Series]:
    hashed = []

    def _l_hash_complite(hashes: tuple):
        hashed.extend(list(map(lambda v: v[0], hashes)))

    # right now
    urls.apply(_l_hash_complite)

    hash_cnt = pd.Series(hashed).value_counts()

    # These constants were found by histograms
    hash_dict_low = hash_cnt[(hash_cnt >= 2) & (hash_cnt < 35)].to_dict()
    hash_dict_middle = hash_cnt[(hash_cnt >= 35) & (hash_cnt < 100)].to_dict()
    hash_dict_up = hash_cnt[hash_cnt >= 100].to_dict()

    hash_set_low = set(hash_dict_low.keys())
    hash_set_middle = set(hash_dict_middle.keys())
    hash_set_up = set(hash_dict_up.keys())

    def _l_count_hash_in(urls_: tuple, hash_set: set):
        hashes_ = set(map(lambda v: v[0], urls_))
        return len(hashes_ & hash_set)

    def _l_single_url(urls_: tuple):
        sum_ = 0
        for url, cnt in urls_:
            # если хэша нет ни в одной
            sum_ += int(url not in hash_set_low and url not in hash_set_middle and url not in hash_set_up)

        return sum_

    ranges = {
        'urls_lower': urls.apply(lambda v: _l_count_hash_in(v, hash_set_low)),
        'urls_middle': urls.apply(lambda v: _l_count_hash_in(v, hash_set_middle)),
        'urls_up': urls.apply(lambda v: _l_count_hash_in(v, hash_set_up)),
        'urls_single': urls.apply(_l_single_url)
    }

    del hashed, hash_dict_low, hash_dict_middle, hash_dict_up
    del hash_set_low, hash_set_middle, hash_set_up

    return ranges


def create_tfidf_urls(urls: dd.Series, cnt=10):
    hash_count_global = {}

    def _l_count_hash_count(urls_: tuple):
        for u, cnt_ in urls_:
            if u not in hash_count_global:
                hash_count_global[u] = 0

            hash_count_global[u] += cnt_

    urls.apply(_l_count_hash_count)

    def _l_count_firsts_urls(urls_: tuple):
        res_ = [0] * cnt
        for i, (u, cnt_) in enumerate(list(urls_)[:cnt]):
            res_[i] = cnt_
        return res_

    df_urls_cnt = pd.DataFrame(urls.apply(_l_count_firsts_urls).tolist(),
                               columns=list(map(lambda i: f'url_cnt_{i + 1}', range(cnt))))

    def _l_freq_firsts_urls(urls_: tuple):
        res_ = [0] * cnt
        for i, (u, cnt_) in enumerate(list(urls_)[:cnt]):
            res_[i] = cnt_ / hash_count_global[u]
        return res_

    df_urls_freq = pd.DataFrame(urls.apply(_l_freq_firsts_urls).tolist(),
                                columns=list(map(lambda i: f'url_freq_{i + 1}', range(cnt))))

    res = {
        'df_urls_cnt': df_urls_cnt,
        'df_urls_freq': df_urls_freq,
    }
    return res
