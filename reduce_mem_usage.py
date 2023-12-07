# Adapted from
# https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits

import numpy as np


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    i8 = np.iinfo(np.int8)
    i16 = np.iinfo(np.int16)
    i32 = np.iinfo(np.int32)
    i64 = np.iinfo(np.int64)
    f16 = np.finfo(np.float16)
    f32 = np.finfo(np.float32)
    mem_before = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > i8.min and c_max < i8.max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > i16.min and c_max < i16.max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > i32.min and c_max < i32.max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > i64.min and c_max < i64.max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > f16.min and c_max < f16.max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > f32.min and c_max < f32.max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    mem_after = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        t = "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)"
        print(t.format(mem_after, 100 * (mem_before - mem_after) / mem_before))
    return df
