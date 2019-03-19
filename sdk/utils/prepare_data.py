import glob
import os
import jieba
from .tokenization import Tokenizer
import tensorflow as tf
##################################################################################################################
def read_raw_files(files):
	for file in files:
		with open(file,'r') as f:
			for line in f:
				line = line.strip()
				src = line.split('\t')[0]
				tgt = line.split('\t')[1]
				yield src,tgt

def produce_semi(params):
	"""
	from raw to semi, produce 2 files, src.txt and tgt.txt
	"""
	raw_data_dir = params['raw_data_dir']
	src_file = params['src_file']
	tgt_file = params['tgt_file']

	raw_files = glob.glob(os.path.join(raw_data_dir,'*.txt'))
	iterator = read_raw_files(raw_files)
	src_writer = open(src_file,'w')
	tgt_writer = open(tgt_file,'w')
	for src,tgt in iterator:
		src_writer.write(src.strip()+'\n')
		tgt_writer.write(tgt.strip()+'\n')
	src_writer.close()
	tgt_writer.close()

##################################################################################################################



def feature_sent_iter(file,tool):
	"""yield list of int"""
	with open(file,'r') as f:
		for line in f:
			line = line.strip()
			feature = tool.encode(line)
			yield feature

def feature_lable_iter(file,labels):
	label2idx = {label:i for i,label in enumerate(labels)}
	with open(file,'r') as f:
		for line in f:
			line = line.strip()
			if line not in label2idx:
				raise ValueError('{} is not in labels'.format(line))
			feature = [label2idx[line]]
			yield feature	


def _int64_feature(value):
	"""
	p.s here value is a list of int
	Returns an int64_list from a bool / enum / int / uint.
	"""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(src, tgt):
	"""
	Creates a tf.Example message ready to be written to a file.
	"""

	# Create a dictionary mapping the feature name to the tf.Example-compatible
	# data type.

	feature = {
		'src': _int64_feature(src),
		'tgt': _int64_feature(tgt),
	}

	# Create a Features message using tf.train.Example.
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()



def semi_to_dataset(params):
	"""
	from semi to tfrecord
	"""
	# Write the `tf.Example` observations to the file.
	src_file = params['src_file']
	tgt_file = params['tgt_file']
	dataset_file = params['dataset_file']
	n_observations = params['n_observations']
	labels = params['labels']
	
	tool = Tokenizer(params)

	with tf.python_io.TFRecordWriter(dataset_file) as writer:
		src_iter = feature_sent_iter(src_file,tool)
		if params['is_tgt_label']:
			tgt_iter  = feature_lable_iter(tgt_file,labels)
		else:
			tgt_iter  = feature_sent_iter(tgt_file,tool)
		for i in range(n_observations):
			try:
		 		example = serialize_example(next(src_iter),next(tgt_iter))
		 		writer.write(example)
			except StopIteration:
				print('over')
				break

