from typing import List
import tensorflow as tf
import numpy as np


def get_mapped_tensor(main_ds: tf.data.Dataset, keys: List[str]):
    return main_ds.map(lambda item: {k: item[k] for k in keys})


def get_integer_lookup_layer(ds: tf.data.Dataset, key: str):
    ids_vocabulary = tf.keras.layers.IntegerLookup(mask_token=None)
    unique_ids = np.unique(ds[key])
    ids_vocabulary.adapt(unique_ids)
    return ids_vocabulary
