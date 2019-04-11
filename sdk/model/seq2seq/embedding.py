import tensorflow as tf

def get_embedding(embedding,src,tgt_in):
	with tf.name_scope('encoder_embeddings'):
		src_embedding = tf.nn.embedding_lookup(embedding, src)
		#####x_train_embedding [input_size, batch_size, embedding_size]
	with tf.name_scope('decoder_embeddings'):
		tgt_in_embedding = tf.nn.embedding_lookup(embedding, tgt_in)
	return src_embedding,tgt_in_embedding
