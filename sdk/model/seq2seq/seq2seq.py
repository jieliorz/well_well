import tensorflow as tf
import pickle
from .embedding import get_embedding
from .get_encoder import get_unbi_encoder,get_bi_encoder
from .get_decoder import get_unbi_train_decoder,get_unbi_decoder
from . import get_learning_rate_and_loss
import logging
from .greedy import greedy_decode
from .beam_search import beam_search_decode
logger = logging.getLogger('seq2seq')



class Seq2SeqModel:
	def __init__(self,params):

		with open('./params.json','r') as f:
			self.params=json.load(f)

		self.save_file=self.params["save_file"]
		self.num_epochs=self.params["num_epochs"]
		self.batch_size=self.params["batch_size"]
		self.vocab_size=self.params["vocab_size"]
		self.embedding_size=self.params["embedding_size"]
		self.hidden_size=self.params["hidden_size"]
		self.learning_rate=self.params["learning_rate"]
		self.keep_prob=self.params["droupout"]
		self.maximum_iterations=self.params['maximum_iterations']
		
		if self.mode == 'infer':
			self.g=tf.Graph()
			with self.g.as_default():
				self.build_graph()
				self.saver = tf.train.Saver()
				self.sess = tf.Session()
				self.saver.restore(self.sess,self.save_file)

	def build_graph(self):

		self.src = tf.placeholder(tf.int32,[self.batch_size,None],name="src")
		self.tgt = tf.placeholder(tf.int32,[self.batch_size,None],name="tgt")
		self.tgt_in = tf.placeholder(tf.int32,[self.batch_size,None],name="tgt_in")
		self.tgt_out = tf.placeholder(tf.int32,[self.batch_size,None],name="tgt_out")

		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.src_lengths= tf.placeholder(tf.int32,[self.batch_size],name="q_lengths")
		self.tgt_lengths= tf.placeholder(tf.int32,[self.batch_size],name="a_lengths")
		
		self.global_step=tf.get_variable(name="global_step",initializer=0,trainable=False)
		
		self.embeddings = tf.get_variable(name="embeddings", shape=[self.vocab_size, self.embedding_size],dtype=tf.float32)
		
		with tf.name_scope("output_projection"):
			self.projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False, name="output_projection")
				
		self.src_embedding,self.tgt_in_embedding=get_embedding(self.embeddings,self.src,self.tgt_in)
		
		encoder_outputs,self.encoder_state = get_bi_encoder(self.src_embedding,self.hidden_size,self.src_lengths,self.dropout_keep_prob)
		self.decoder_cell=get_unbi_decoder(self.hidden_size,self.dropout_keep_prob)
	

		if self.mode == 'train':
			self.logits=get_unbi_train_decoder(self.tgt_in_embedding,self.encoder_state,self.decoder_cell,self.hidden_size,
				self.batch_size,self.maximum_iterations,self.projection_layer)
			self.loss = get_learning_rate_and_loss.get_loss(self.logits,self.tgt_out,self.tgt_lengths,self.maximum_iterations,self.batch_size)
			self.output = tf.argmax(self.logits,-1)
			#self.learning_rate=get_learning_rate_and_loss.get_learning_rate(self.learning_rate,self.hidden_size,learning_rate_warmup_steps=160)
			self.train_op = get_learning_rate_and_loss.get_train_op(self.learning_rate,self.loss,self.global_step)
		elif self.mode == 'infer':
			decode_mode=self.params['decode_mode']
			sos_id=self.params['sos_id']
			eos_id=self.params['eos_id']
			if decode_mode == 'greedy':
				self.output=greedy_decode(self.batch_size,sos_id,eos_id,self.embeddings,
					self.decoder_cell,self.encoder_state,self.projection_layer,
					self.maximum_iterations)
			elif decode_mode == 'beam_search':
				beam_width=self.params['beam_width']
				self.output=beam_search_decode(self.batch_size,sos_id,eos_id,
					self.embeddings,self.encoder_state,self.decoder_cell,beam_width,
					self.projection_layer,self.maximum_iterations)


	def train(self):

		dataset_obj=DataSet(self.params)
		dataset=dataset_obj.prepare_dataset()

		init_train=self.params['init_train']

		g=tf.Graph()
		with g.as_default():
			self.build_graph()
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
								self.raw_seq_len: batch['src_len']
								}
						_, step, loss, accuracy = sess.run(
						[self.train_op, self.global_step, self.loss, self.accuracy],
						feed_dict)
						if step%100 == 0:
							print('train step:{} loss:{} accuracy:{}'.format(step,loss,accuracy))
					except tf.errors.OutOfRangeError:
						print('over')
						break

			self.saver.save(sess,save_file)



					feed_dict = {
							self.src: src,
							self.tgt_in:tgt_in,
							self.tgt_out:tgt_out,
							self.dropout_keep_prob: droupout,
							self.src_lengths:src_lengths,
							self.tgt_lengths:tgt_lengths,
							}
		
					_,_,loss,step= sess.run([self.output,self.train_op,self.loss,self.global_step],feed_dict)
					# logger.info('maximum_iterations:{},tgt_in:{},tgt_out:{},p:{}'.format(self.maximum_iterations,tgt_in[0],tgt_out[0],p[0]))


					if step%100 == 0:
						logger.info("sampling data src:{}, tgt_in:{},tgt_out:{},src_len:{}, tgt_len:{}".format(
							tokenizer.decode(src[0]),tokenizer.decode(tgt_in[0]),tokenizer.decode(tgt_out[0]),src_lengths[0],tgt_lengths[0]))
						#print("train step:{},loss:{}, lr:{}".format(step,loss))
						logger.info("train step:{},loss:{}".format(step,loss))
					if loss < 0:
						logger.info("break step:{},loss:{}".format(step,loss))
						break
				logger.info("last train step:{} loss:{}".format(step,loss))


				
	def infer(self,subtoken):
		droupout=1.0
		with self.sess.as_default():
			# src_total,tgt_in_total,tgt_out_total,src_lengths,tgt_lengths
			src=[subtoken]
			src_lengths=[len(subtoken)]
			feed_dict = {
					self.src: src,
					self.dropout_keep_prob:droupout,
					self.src_lengths:src_lengths,
					}
			output= self.sess.run([self.output],feed_dict)
			# print(output[0],output[0][0].dtype)
			# logger.info("batch output:{}".format(output))
			return output[0]



				# for i in range(beam_width):
				# 	res=[int(i[0]) for i in list(output[i][0])]
