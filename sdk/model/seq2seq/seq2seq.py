import sys
sys.path.append('..')
print(sys.path)
import tensorflow as tf
import pickle
from .embedding import get_embedding
from .get_encoder import get_unbi_encoder,get_bi_encoder
from .get_decoder import get_unbi_train_decoder,get_unbi_decoder
from .get_learning_rate_and_loss import get_loss,get_train_op
import logging
from .greedy import greedy_decode
from .beam_search import beam_search_decode
import os
import yaml
from utils.data_helper import DataSet

fpath = os.path.dirname(__file__)
logger = logging.getLogger('seq2seq')



class Seq2SeqModel:
	def __init__(self):

		with open(os.path.join(fpath,'params.yml'),'r') as f:
			self.params=yaml.load(f)

		self.save_file=self.params["save_file"]
		self.num_epochs=self.params["num_epochs"]
		self.batch_size=self.params["batch_size"]
		self.vocab_size=self.params["vocab_size"]
		self.embedding_size=self.params["embedding_size"]
		self.hidden_size=self.params["hidden_size"]
		self.learning_rate=self.params["learning_rate"]
		self.keep_prob=self.params["keep_prob"]
		self.maximum_iterations=self.params['max_length'] - 1
		


	def build_graph(self):

		self.src = tf.placeholder(tf.int64,[None,None],name="src")
		self.tgt = tf.placeholder(tf.int64,[None,None],name="tgt")
		self.tgt_in = tf.strided_slice(self.tgt,[0,0],[tf.shape(self.tgt)[0],tf.shape(self.tgt)[1]-1],[1,1],name="tgt_in")
		self.tgt_out = tf.strided_slice(self.tgt,[0,1],[tf.shape(self.tgt)[0],tf.shape(self.tgt)[1]],[1,1],name="tgt_out")

		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.raw_src_len = tf.placeholder(tf.int64,[None,1])
		self.src_lengths= tf.reshape(self.raw_src_len,shape=[-1],name="src_len")

		self.raw_tgt_len = tf.placeholder(tf.int64,[None,1])
		self.tgt_lengths= tf.reshape(self.raw_tgt_len,shape=[-1],name="tgt_len")

		
		self.global_step=tf.get_variable(name="global_step",initializer=0,trainable=False)
		
		self.embeddings = tf.get_variable(name="embeddings", shape=[self.vocab_size, self.embedding_size],dtype=tf.float32)
		
		with tf.name_scope("output_projection"):
			self.projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False, name="output_projection")
				
		self.src_embedding,self.tgt_in_embedding=get_embedding(self.embeddings,self.src,self.tgt_in)
		
		encoder_outputs,self.encoder_state = get_bi_encoder(self.src_embedding,self.hidden_size,self.src_lengths,self.dropout_keep_prob)
		self.decoder_cell=get_unbi_decoder(self.hidden_size,self.dropout_keep_prob)
	

	def train(self):

		dataset_obj=DataSet(self.params)
		dataset=dataset_obj.prepare_dataset()

		init_train=self.params['init_train']

		g=tf.Graph()
		with g.as_default():
			self.build_graph()
			self.logits=get_unbi_train_decoder(self.tgt_in_embedding,self.encoder_state,self.decoder_cell,self.hidden_size,
				self.batch_size,self.maximum_iterations,self.projection_layer)
			self.loss = get_loss(self.logits,self.tgt_out,self.tgt_lengths,self.maximum_iterations,self.batch_size)
			tf.summary.scalar('loss', self.loss)
			self.output = tf.argmax(self.logits,-1)
			self.train_op = get_train_op(self.learning_rate,self.loss,self.global_step)
			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(fpath + '/train', g)

			self.saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			sess = tf.Session()

			with sess.as_default():
				if init_train:
					sess.run(init)
				else:
					self.saver.restore(sess,self.save_file)

				iterator=dataset.make_one_shot_iterator()
				next_element=iterator.get_next()
				while 1:
					try:
						batch=sess.run(next_element)
						feed_dict = {
								self.src: batch['src'],
								self.tgt: batch['tgt'],
								self.dropout_keep_prob: self.keep_prob,
								self.raw_src_len: batch['src_len'],
								self.raw_tgt_len: batch['tgt_len']
								}
						summary,_,_,loss,step= sess.run([merged,self.output,self.train_op,self.loss,self.global_step],feed_dict)
						train_writer.add_summary(summary, step)
						print('train step:{} loss:{}'.format(step,loss))
					except tf.errors.OutOfRangeError:
						print('over')
						break

			self.saver.save(sess,self.save_file)




	# def infer(self,subtoken):

	# 	self.g=tf.Graph()
	# 	with self.g.as_default():
	# 		self.build_graph()
	# 		decode_mode=self.params['decode_mode']
	# 		sos_id='<s>'
	# 		eos_id='</s>'
	# 		if decode_mode == 'greedy':
	# 			self.output=greedy_decode(self.batch_size,sos_id,eos_id,self.embeddings,
	# 				self.decoder_cell,self.encoder_state,self.projection_layer,
	# 				self.maximum_iterations)
	# 		elif decode_mode == 'beam_search':
	# 			beam_width=self.params['beam_width']
	# 			self.output=beam_search_decode(self.batch_size,sos_id,eos_id,
	# 				self.embeddings,self.encoder_state,self.decoder_cell,beam_width,
	# 				self.projection_layer,self.maximum_iterations)



			# self.saver = tf.train.Saver()
			# self.sess = tf.Session()
			# self.saver.restore(self.sess,self.save_file)

			# droupout=1.0
			# with self.sess.as_default():
			# 	# src_total,tgt_in_total,tgt_out_total,src_lengths,tgt_lengths
			# 	src=[subtoken]
			# 	src_lengths=[len(subtoken)]
			# 	feed_dict = {
			# 			self.src: src,
			# 			self.dropout_keep_prob:droupout,
			# 			self.src_lengths:src_lengths,
			# 			}
			# 	output= self.sess.run([self.output],feed_dict)
			# 	# print(output[0],output[0][0].dtype)
			# 	# logger.info("batch output:{}".format(output))
			# 	return output[0]



			# 		# for i in range(beam_width):
			# 		# 	res=[int(i[0]) for i in list(output[i][0])]
