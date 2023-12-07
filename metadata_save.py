import json
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa


def df_to_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Saves a DataFrame as a .parquet like the Pandas '.to_parquet' normally
    would, but also produces a simplified metadata json, which can be useful in
    a simple version control through git or bash, without having to add the
    actual dataset.

    :param df: A Pandas DataFrame (dataset).
    :param path: Save path for the dataset as a parquet file.  The name of the
        metadata file will be the path name without the '.parquet' extension,
        but instead with a suffix '_metadata' and the extension '.json'
    :param kwargs: Any arguments for the Pandas DataFrame '.to_parquet' method.

    :returns: Nothing, but a parquet and a json file is produced.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)
    metadata = pa.parquet.ParquetFile(path).metadata.to_dict()
    metadata_path = path.parent / f"{path.stem}_metadata.json"
    new_meta = {}
    new_meta["timestamp"] = f"{datetime.now().isoformat()}"
    new_meta["num_columns"] = metadata["num_columns"]
    new_meta["num_rows"] = metadata["num_rows"]
    new_meta["serialized_size"] = metadata["serialized_size"]
    new_meta["total_byte_size"] = metadata["row_groups"][0]["total_byte_size"]
    new_meta["data"] = []
    for info in metadata["row_groups"][0]["columns"]:
        feature = info["path_in_schema"]
        if feature == "index":
            continue

        feature_md = {}
        feature_md["feature"] = feature
        feature_md["physical_type"] = info["physical_type"]
        feature_md["num_values"] = info["statistics"]["num_values"]
        feature_md["null_count"] = info["statistics"]["null_count"]
        feature_md["min"] = info["statistics"]["min"]
        feature_md["max"] = info["statistics"]["max"]
        feature_md["compression"] = info["compression"]
        feature_md["total_compressed_size"] = info["total_compressed_size"]
        feature_md["total_uncompressed_size"] = info["total_uncompressed_size"]
        new_meta["data"].append(feature_md)
    with open(metadata_path, "w") as file:
        json.dump(new_meta, file, indent=4, default=str)


# if __name__ == "__main__":
#     path = (
#         "/home/chris/scratch/ogi/datasets/mlp_score/"
#         "ogi_cp_20220809_165454/dataset.parquet"
#     )
#     df = pd.read_parquet(path)
#     df_to_parquet(df, "~/test_dataset.parquet", compression="zstd")
