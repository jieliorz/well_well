import os
from .tokenization import Tokenizer
import tensorflow as tf
tf.enable_eager_execution()

def parser(string_record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature([], tf.int64,default_value=[]),
        "label": tf.FixedLenFeature([], tf.int64,default_value=[])
    }
    parsed = tf.parse_single_example(record, keys_to_features)


class Dataset:
	def __init__(self,dataset_file):
		# record_iterator = tf.python_io.tf_record_iterator(path=dataset_file)

		# for string_record in record_iterator:
		# 	example = tf.train.Example()
		# 	example.ParseFromString(string_record)

		# 	print(example)

			# Exit after 1 iteration as this is purely demonstrative.
		filenames = [dataset_file]
		dataset = tf.data.TFRecordDataset(filenames)
		# dataset.map(parse_exmp)
		print(dataset.take(1))