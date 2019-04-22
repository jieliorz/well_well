import tensorflow as tf
from utils.tokenization import Tokenizer,EOS_ID,SOS_ID
import yaml
import numpy as np

class TextRnn:
	def __init__(self):

		with open('./params.yml','r') as f:
			params=yaml.load(f)
		self.params = params
		self.tokenizer = Tokenizer(self.params)
		self.embedding_size=self.params["embedding_size"]
		self.vocab_size=self.params["vocab_size"]
		self.input_size=self.params["max_length"]
		self.num_classes=len(self.params['labels'])
		self.keep_prob=self.params["dropout_keep_prob"]
		self.learning_rate=self.params["learning_rate"]
		self.num_epochs=self.params["num_epochs"]
		self.batch_size=self.params["batch_size"]
		self.hidden_size=self.params["hidden_size"]
		self.mode=self.params["mode"]

	def build(self):
		self.src = tf.placeholder(tf.int32,[None,self.input_size],name="src")

		self.tgt_raw = tf.placeholder(tf.int64,[None,1])

		self.tgt = tf.one_hot(tf.reshape(self.tgt_raw,shape=[-1]),depth=self.num_classes,name='tgt')

		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.raw_seq_len = tf.placeholder(tf.int64,[None,1])
		self.seq_lengths= tf.reshape(self.raw_seq_len,shape=[-1],name="src_len")

		self.global_step=tf.Variable(0,name='global_step',trainable=False)

		with tf.name_scope('embeddings'):
			self.embeddings = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
			self.src_embedding = tf.nn.embedding_lookup(self.embeddings, self.src)
			# . The result of the embedding operation is a 3-dimensional tensor of shape [None,input_size,embedding_size].

		with tf.name_scope('rnn'):
			lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) #forward direction cell
			lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) #backward direction cell		
			lstm_fw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
			lstm_bw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)

			outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.src_embedding,sequence_length=self.seq_lengths,dtype=tf.float32)
			#[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
			output_rnn=tf.concat(outputs,axis=2)#[batch_size,sequence_length,hidden_size*2] 两个tensor按照第2个维度（hidden size）连接
			self.final_outputs=tf.reduce_sum(output_rnn,axis=1)#shape=[batch_size,2*hidden_size]按维度1(即senquence length)相加

		with tf.name_scope('output'):
			W = tf.Variable(tf.truncated_normal([2*self.hidden_size,self.num_classes]))
			b = tf.Variable(tf.constant(0.1,shape=[self.num_classes]))
			self.scores=tf.nn.xw_plus_b(self.final_outputs,W,b,name='scores')
			self.predictions=tf.argmax(self.scores,1,name='predictions')

		with tf.name_scope('loss'):
			losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.tgt)
			self.loss=tf.reduce_mean(losses)

		with tf.name_scope('accuracy'):
			correct_predictions=tf.equal(self.predictions,tf.argmax(self.tgt, 1))
			self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		if self.mode == "train":
			optimizer=tf.train.AdamOptimizer(self.learning_rate)
			grads_and_vars=optimizer.compute_gradients(self.loss)
			#返回A list of (gradient, variable) pairs
			self.train_op=optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
			# global_step可以理解为调用train_op的次数

	def train(self,dataset):
		save_file=self.params["save_file"]
		init_train=self.params['init_train']

		g=tf.Graph()
		with g.as_default():
			self.build()
			self.saver = tf.train.Saver()
			init = tf.global_variables_initializer()
			sess = tf.Session()
			with sess.as_default():
				if init_train:
					sess.run(init)
					print('start to first training')
				else:
					self.saver.restore(sess,save_file)
				
				iterator=dataset.make_one_shot_iterator()
				next_element=iterator.get_next()
				while 1:
					try:
						batch=sess.run(next_element)
						# if batch['src'].shape[0] != self.batch_size:
						# 	break
						feed_dict = {
								self.src: batch['src'],
								self.tgt_raw: batch['tgt'],
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

	def infer(self):
		save_file=self.params["save_file"]
		self.mode = 'infer'
		self.g=tf.Graph()
		with self.g.as_default():
			self.build()
			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.saver.restore(self.sess,save_file)
	

	def predict(self,sentence):
		line_encode,line_len = self.tokenizer.encode(sentence,padding=True)
		with self.sess.as_default():
			feed_dict = {
					self.src: [line_encode],
					self.dropout_keep_prob: 1.0,
					self.raw_seq_len: [[line_len]]
					}
			prediction = self.sess.run(self.predictions,feed_dict)

			return self.params['labels'][int(prediction[0])]
	
	def test(self,sentence_list):
		res = [self.tokenizer.encode(sentence,padding=True) for sentence in sentence_list]
		line_encode_list = [each[0] for each in res]
		line_len_list = [[each[1]] for each in res]
		
		with self.sess.as_default():
			feed_dict = {
					self.src: line_encode_list,
					self.dropout_keep_prob: 1.0,
					self.raw_seq_len: line_len_list
					}
			prediction = self.sess.run(self.predictions,feed_dict)

			return [self.params['labels'][int(prediction[i])] for i in range(len(prediction))]



	def test_batch(self,dataset):
		save_file=self.params["save_file"]
		g=tf.Graph()
		with g.as_default():
			self.build()
			self.saver = tf.train.Saver()
			sess = tf.Session()
			with sess.as_default():
				self.saver.restore(sess,save_file)
				
				iterator=dataset.make_one_shot_iterator()
				next_element=iterator.get_next()
				while 1:
					try:
						batch=sess.run(next_element)
						# if batch['src'].shape[0] != self.batch_size:
						# 	break
						feed_dict = {
								self.src: batch['src'],
								self.tgt_raw: batch['tgt'],
								self.dropout_keep_prob: 1.0,
								self.raw_seq_len: batch['src_len']
								}
						loss, accuracy = sess.run([self.loss, self.accuracy],feed_dict)
						if accuracy != 1:
							predictions=sess.run([self.predictions],feed_dict)
							predictions=list(predictions[0])
							tgt=list(np.reshape(batch['tgt'],[-1]))
							res={self.tokenizer.decode([int(x) for x in batch['src'][i]]): 'real:'+self.params['labels'][int(tgt[i])]+'/predict:'+self.params['labels'][int(predictions[i])] for i in range(len(tgt)) if int(predictions[i]) != int(tgt[i])}
							print(res)
							break
						print('loss:{} accuracy:{}'.format(loss,accuracy))
					except tf.errors.OutOfRangeError:
						print('over')
						break
