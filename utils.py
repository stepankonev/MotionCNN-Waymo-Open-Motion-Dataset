import os
import yaml
import tensorflow as tf


def create_tf_dataset(datapath, n_shards, shard_id):
    files = os.listdir(datapath)
    dataset = tf.data.TFRecordDataset(
        [os.path.join(datapath, f) for f in files], num_parallel_reads=1
    )
    if n_shards > 1:
        dataset = dataset.shard(n_shards, shard_id)
    return dataset


def get_config(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data