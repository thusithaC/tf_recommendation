from typing import Dict, Text, Union, List

import numpy as np
import tensorflow as tf
import pandas as pd
from models.recomender_model import DEFAULTS, RecommenderModel
from utils.mongo_helper import insert_one
from utils.common import get_mapped_tensor, get_integer_lookup_layer

tf.random.set_seed(42)
USE_MOD_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
EVENTS_PATH = "/home/thusitha/work/projects/recommendation_take_home/data/sample_1k_users.csv"

params = {
    "defaults": DEFAULTS,
    "train_test_split": 0.90,
    "dense_layers": [16],
    "learning_rate": 0.0005,
    "epochs": 20
}

events_dt = pd.read_csv(EVENTS_PATH)
events_ts_ds = tf.data.Dataset.from_tensor_slices(dict(events_dt[["user_id", "item_id"]]))
print(len(events_ts_ds))

# create mapped tensors
events_ds = get_mapped_tensor(events_ts_ds, ["user_id", "item_id"])
items_ds = get_mapped_tensor(events_ts_ds, ["item_id"])

# create Integer lookup layers
user_ids_vocabulary = get_integer_lookup_layer(events_dt, "user_id")
item_ids_vocabulary = get_integer_lookup_layer(events_dt, "item_id")

# Crete train test data
shuffled_data = events_ds.shuffle(len(events_ds))
n_train = int(len(events_ds)*params["train_test_split"])
n_test = len(events_ds) - n_train
train = shuffled_data.take(n_train)
test = shuffled_data.skip(n_train).take(n_test)
print(len(train), len(test))

# build model
model = RecommenderModel(user_ids_vocabulary, item_ids_vocabulary, items_ds, dense_layers=params["dense_layers"])

# train and eval
cached_train = train.batch(DEFAULTS["batch_size"]).cache()
cached_test = test.batch(DEFAULTS["batch_size"]).cache()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]))
model.fit(cached_train, epochs=params["epochs"])

eval_result = model.evaluate(cached_test, return_dict=True)

# save results
model_save = {
    "desc": "1 k users, simple with dense",
    "eval": eval_result,
    "data": {"n_train": len(train), "n_test": len(test), "path": EVENTS_PATH},
    "params": params
}
print(model_save)
insert_one(model_save)
