import os
import tensorflow as tf
# tf.enable_eager_execution()

# def parser(string_record):
#     keys_to_features = {
#         "image_data": tf.FixedLenFeature([], tf.int64,default_value=[]),
#         "label": tf.FixedLenFeature([], tf.int64,default_value=[])
#     }
#     parsed = tf.parse_single_example(record, keys_to_features)


class DataSet:
    def __init__(self,params):
        self.params=params
        dataset_file=self.params['dataset_file']
        self.filenames = [dataset_file]
        self.buffer_size = self.params['n_observations']
        self.batch_size = self.params["batch_size"]
        self.num_epochs = self.params["num_epochs"] 
        # record_iterator = tf.python_io.tf_record_iterator(path=dataset_file)
        # for string_record in record_iterator:
        #     example = tf.train.Example()
        #     example.ParseFromString(string_record)

        #     print(example)

        # # Exit after 1 iteration as this is purely demonstrative.

    def prepare_dataset(self):
        raw_dataset = tf.data.TFRecordDataset(self.filenames)
        max_length=self.params['max_length']

        if self.params['is_tgt_label']:
        	tgt_max_length = 1
        else:
        	tgt_max_length = max_length

        print('tgt_max_length',tgt_max_length)
        def parse(record):
            feature_description = {
               "src": tf.FixedLenFeature([max_length], tf.int64),
               "tgt": tf.FixedLenFeature([tgt_max_length], tf.int64),
               "src_len":tf.FixedLenFeature([1], tf.int64),
               "tgt_len":tf.FixedLenFeature([1], tf.int64),
                }
            return tf.parse_single_example(record,feature_description)

        dataset = raw_dataset.map(parse)
        dataset = dataset.shuffle(self.buffer_size)
        # dataset = dataset.batch(self.batch_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.repeat(self.num_epochs)

        return dataset

        # iterator=dataset.make_one_shot_iterator()
        # next_element=iterator.get_next()

        # sess=tf.Session()
        # while 1:
        #     try:
        #         value=sess.run(next_element)
        #         print(value)
        #     except tf.errors.OutOfRangeError:
        #         print('over')
        #         break
        # # example_proto = tf.train.Example.FromString(value)
 
        # # # print(type(example_proto))
