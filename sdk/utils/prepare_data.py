import glob
import os
import jieba
from .tokenization import Tokenizer
from .pre_process import pre_process
import tensorflow as tf
##################################################################################################################
def read_raw_files(files):
	for file in files:
		with open(file,'r') as f:
			for line in f:
				line = line.strip()
				if len(line.split('\t')) != 2:
					continue
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
	n_observations = params['n_observations']
	max_length = params['max_length']

	raw_files = glob.glob(os.path.join(raw_data_dir,'*.txt'))
	print('semi from: {}'.format(raw_files))
	iterator = read_raw_files(raw_files)
	src_writer = open(src_file,'w')
	tgt_writer = open(tgt_file,'w')
	cnt = 0
	for src,tgt in iterator:
		src=pre_process(src,keep_sep=True)
		if not src:
			continue
		tgt=pre_process(tgt,keep_sep=True)
		if not tgt:
			continue		
		src_writer.write(src.strip()+'\n')
		tgt_writer.write(tgt.strip()+'\n')
		cnt+=1
		if n_observations:
			if cnt > n_observations:
				break

	src_writer.close()
	tgt_writer.close()
	print('semi src_file: {} saved'.format(src_file))
	print('semi tgt_file: {} saved'.format(tgt_file))

##################################################################################################################



def feature_sent_iter(file,tool,padding=False,start_mark=False,end_mark=False):
	"""yield list of int"""
	with open(file,'r') as f:
		for line in f:
			line = line.strip()
			line_encode,line_len = tool.encode(line,padding=padding,start_mark=start_mark,end_mark=end_mark)
			yield line_encode,line_len

def feature_lable_iter(file,labels):
	label2idx = {label:i for i,label in enumerate(labels)}
	with open(file,'r') as f:
		for line in f:
			line = line.strip()
			if line not in label2idx:
				raise ValueError('{} is not in labels'.format(line))

			feature = [label2idx[line]]
			yield feature,1


def _int64_feature(value):
	"""
	p.s here value is a list of int
	Returns an int64_list from a bool / enum / int / uint.
	"""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(src, tgt, src_len, tgt_len):
	"""
	Creates a tf.Example message ready to be written to a file.
	"""

	# Create a dictionary mapping the feature name to the tf.Example-compatible
	# data type.
	feature = {
		'src': _int64_feature(src),
		'src_len':_int64_feature([src_len]),
		'tgt': _int64_feature(tgt),
		'tgt_len':_int64_feature([tgt_len])
	}

	# Create a Features message using tf.train.Example.
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()



def semi_to_dataset(params,tokenizer):
	"""
	from semi to tfrecord
	"""
	# Write the `tf.Example` observations to the file.
	src_file = params['src_file']
	tgt_file = params['tgt_file']
	dataset_file = params['dataset_file']
	n_observations = params['n_observations']
	with tf.python_io.TFRecordWriter(dataset_file) as writer:
		src_iter = feature_sent_iter(src_file,tokenizer,padding=True,start_mark=False,end_mark=False)
		if params['is_tgt_label']:
			labels = params['labels']
			tgt_iter  = feature_lable_iter(tgt_file,labels)
		else:
			tgt_iter  = feature_sent_iter(tgt_file,tokenizer,padding=True,start_mark=True,end_mark=True)
		cnt=0
		while 1:
			try:
				src,src_len = next(src_iter)
				tgt,tgt_len = next(tgt_iter)
				if not src:
					continue
				if not tgt:
					continue
				example = serialize_example(src,tgt,src_len,tgt_len)
				writer.write(example)
				cnt+=1
				if n_observations:
					if cnt > n_observations:
						break
			except StopIteration:
				print('over')
				break
		n_observations = cnt - 1
		return n_observations



