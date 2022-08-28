from typing import Dict

import pandas as pd
from dask import dataframe as dd


def preprocess_date(s: dd.Series):
    s = s.astype(str)
    s = dd.to_datetime(s, format='%Y%m%d')

    return s


def generate_features(s: dd.Series) -> Dict[str, dd.Series]:
    if pd.api.types.is_numeric_dtype(s.dtype):
        s = preprocess_date(s)

    features = {
        'year': s.dt.year,
        'month': s.dt.month,
        'day': s.dt.day,
        'day_of_week': s.dt.dayofweek,
    }

    return features
