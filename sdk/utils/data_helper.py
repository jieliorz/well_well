import os
from .tokenization import Tokenizer
import tensorflow as tf



class Dataset:
	def __init__(self,dataset_file):
		record_iterator = tf.python_io.tf_record_iterator(path=dataset_file)

		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)

			print(example)

			# Exit after 1 iteration as this is purely demonstrative.
			break