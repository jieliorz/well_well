import tensorflow as tf

def beam_search_decode(
	batch_size,sos_id,eos_id,embeddings,encoder_state,decoder_cell,beam_width,projection_layer,maximum_iterations):
	start_tokens = tf.fill([batch_size], tf.cast(sos_id,tf.int32),name='start_tokens')
	end_token = tf.cast(eos_id,tf.int32)
	decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)		
	with tf.name_scope('beam_search'):
		decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
														embedding=embeddings,
														start_tokens=start_tokens,
														end_token=end_token,
														initial_state=decoder_initial_state,
														beam_width=beam_width,
														output_layer=projection_layer)
	# Dynamic decoding
	outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=maximum_iterations)
	predict = outputs.predicted_ids
	answers=[]
	for i in range(beam_width):
		a=tf.strided_slice(predict,[0,0,i],[batch_size, tf.shape(predict)[1],i+1],[1,1,1])
		a=tf.reshape(a,[batch_size,-1])[0]
		answers.append(a)
	#[batch_size, length, beam_width]
	return answers

	
