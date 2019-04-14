import tensorflow as tf


def get_unbi_decoder(
	hidden_size,dropout_keep_prob):
	with tf.name_scope('decoder_rnn'):
		decoder_cell= tf.contrib.rnn.LSTMCell(hidden_size) #forward direction cell
		decoder_cell=tf.nn.rnn_cell.DropoutWrapper(decoder_cell,output_keep_prob=dropout_keep_prob)
	return decoder_cell

def get_unbi_train_decoder(
	tgt_in_embedding,encoder_state,decoder_cell,hidden_size,batch_size,maximum_iterations,projection_layer):
	
	# helper
	helper = tf.contrib.seq2seq.TrainingHelper(tgt_in_embedding, batch_size*[maximum_iterations])
	# Decoder
	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,output_layer=projection_layer)
	# # Dynamic decoding

	decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
	#decoder_outputs.rnn_output  [batch_size,sequence_length,vocab_size]; decoder_outputs.sample_id   [batch_size,sequence_length]
	logits = decoder_outputs.rnn_output
	return logits
