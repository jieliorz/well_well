import tensorflow as tf



def get_unbi_encoder(src_embedding,hidden_size,src_lengths,dropout_keep_prob):
	#encoder
	with tf.name_scope('encoder_rnn'):
		encoder_cell= tf.contrib.rnn.LSTMCell(hidden_size)
		encoder_cell=tf.nn.rnn_cell.DropoutWrapper(encoder_cell,output_keep_prob=dropout_keep_prob)
		encoder_outputs,encoder_state=tf.nn.dynamic_rnn(
			encoder_cell,src_embedding,dtype=tf.float32,sequence_length=src_lengths)
	return encoder_outputs,encoder_state



def get_bi_encoder(src_embedding,hidden_size,src_lengths,dropout_keep_prob):
	#encoder
	with tf.name_scope('encoder_rnn'):
		encoder_cell_fw= tf.contrib.rnn.LSTMCell(hidden_size//2)  #forward direction cell
		encoder_cell_bw= tf.contrib.rnn.LSTMCell(hidden_size//2)
		encoder_cell_fw=tf.nn.rnn_cell.DropoutWrapper(encoder_cell_fw,output_keep_prob=dropout_keep_prob)
		encoder_cell_bw=tf.nn.rnn_cell.DropoutWrapper(encoder_cell_bw,output_keep_prob=dropout_keep_prob)
		(encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
		tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,encoder_cell_bw,src_embedding,dtype=tf.float32,sequence_length=src_lengths)
		#time_major=False   encoder_outputs  [batch_size,sequence_length,hidden_size] encoder_state [batch_size,hidden_size]
		encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
		encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

		encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
		
	return tf.concat((encoder_fw_outputs, encoder_bw_outputs),-1),encoder_final_state









def getLayeredCell(layer_size, hidden_size, dropout_keep_prob,
        output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(hidden_size),
        dropout_keep_prob) for i in range(layer_size)])


def get_multi_bi_encoder(src_embedding, src_lengths, hidden_size, layer_size, dropout_keep_prob):
    # encode input into a vector
    bi_layer_size = int(layer_size / 2)
    encode_cell_fw = getLayeredCell(bi_layer_size, hidden_size, dropout_keep_prob)
    encode_cell_bw = getLayeredCell(bi_layer_size, hidden_size, dropout_keep_prob)
    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = encode_cell_fw,
            cell_bw = encode_cell_bw,
            inputs = src_embedding,
            sequence_length = src_lengths,
            dtype = src_embedding.dtype,
            time_major = False)

    # concat encode output and state
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(bi_layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state