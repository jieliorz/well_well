import os
from .tokenization import Tokenizer
import tensorflow as tf
# tf.enable_eager_execution()

# def parser(string_record):
#     keys_to_features = {
#         "image_data": tf.FixedLenFeature([], tf.int64,default_value=[]),
#         "label": tf.FixedLenFeature([], tf.int64,default_value=[])
#     }
#     parsed = tf.parse_single_example(record, keys_to_features)


class Dataset:
    def __init__(self,params):
        self.params=params
        dataset_file=params['dataset_file']
        filenames = [dataset_file]

        # record_iterator = tf.python_io.tf_record_iterator(path=dataset_file)
        # for string_record in record_iterator:
        #     example = tf.train.Example()
        #     example.ParseFromString(string_record)

        #     print(example)

        # # Exit after 1 iteration as this is purely demonstrative.

        raw_dataset = tf.data.TFRecordDataset(filenames)
        max_length=self.params['max_length']
        if self.params['is_tgt_label']:
        	tgt_max_length = 1
        else:
        	tgt_max_length = max_length

        print('tgt_max_length',tgt_max_length)
        def parse(record):
            feature_description = {
               "src": tf.FixedLenFeature([max_length], tf.int64),
               "tgt": tf.FixedLenFeature([tgt_max_length], tf.int64)
                }
            return tf.parse_single_example(record,feature_description)

        dataset=raw_dataset.map(parse)

        iterator=dataset.make_one_shot_iterator()
        next_element=iterator.get_next()



        sess=tf.Session()
        value=sess.run(next_element)
        # example_proto = tf.train.Example.FromString(value)
        print(value)
        # print(type(example_proto))
