import tensorflow as tf

def get_learning_rate(learning_rate,hidden_size,learning_rate_warmup_steps=160):

	"""Calculate learning rate with linear warmup and rsqrt decay."""
	with tf.name_scope("learning_rate"):
		learning_rate_warmup_steps=160
		warmup_steps = tf.to_float(learning_rate_warmup_steps)
		step = tf.to_float(tf.train.get_or_create_global_step())
		learning_rate *= (hidden_size ** -0.5)
		# Apply linear warmup
		learning_rate *= tf.minimum(1.0, step / warmup_steps)
		# Apply rsqrt decay
		learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
	return learning_rate




def get_loss(logits,tgt_out,tgt_lengths,maximum_iterations):
	with tf.name_scope('loss'):
		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=tgt_out, logits=logits)
		target_weights = tf.sequence_mask(tgt_lengths, maximum_iterations, dtype=tf.float32)
		loss = tf.reduce_sum(crossent * target_weights)/tf.to_float(tf.shape(tgt_out)[0])
		# tf.to_float(tf.reduce_sum(tgt_lengths))
		return loss

def get_train_op(learning_rate,loss,global_step):
	params = tf.trainable_variables()
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
	
	train_op = tf.train.AdamOptimizer(
		learning_rate=learning_rate).apply_gradients(zip(clipped_gradients,params),global_step=global_step)
	return train_op


