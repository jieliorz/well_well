import tensorflow as tf


def greedy_decode(
	batch_size,sos_id,eos_id,embeddings,decoder_cell,encoder_state,projection_layer,maximum_iterations):
	start_tokens = tf.fill([batch_size], tf.cast(sos_id,tf.int32),name='start_tokens')
	end_token = tf.cast(eos_id,tf.int32)
	# Helper
	helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens, end_token)
	# Decoder
	decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
											initial_state=encoder_state,output_layer=projection_layer)
	
	outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=maximum_iterations)
	predict = tf.argmax(outputs.rnn_output,axis=-1)
	return predict


