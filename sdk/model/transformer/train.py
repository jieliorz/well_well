# from model.seq2seq.seq2seq import Seq2SeqModel

# model=Seq2SeqModel()
# # model.train()
# model.infer()
# while 1:
# 	sentence = input('you:')
# 	model.predict(sentence)

# # sentence='打开收音机'
# # ret,ret_len = tokenizer.encode(sentence,padding=True)
# # prediction=model.predict(ret,ret_len)
# # print(prediction)





from model.transformer import Transformer

dataset_obj=DataSet(params)
dataset=dataset_obj.prepare_dataset()

def input_fn(dataset):

	return features, labels

def model_fn(features, labels, mode, params):
  """Defines how to train, evaluate and predict from the transformer model."""
  with tf.variable_scope("model"):
    inputs, targets = features, labels

    # Create model and get output logits.
    model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

    logits = model(inputs, targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
      if params["use_tpu"]:
        raise NotImplementedError("Prediction is not yet supported on TPUs.")
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=logits,
          export_outputs={
              "translate": tf.estimator.export.PredictOutput(logits)
          })

    # Explicitly set the shape of the logits for XLA (TPU). This is needed
    # because the logits are passed back to the host VM CPU for metric
    # evaluation, and the shape of [?, ?, vocab_size] is too vague. However
    # it is known from Transformer that the first two dimensions of logits
    # are the dimensions of targets. Note that the ambiguous shape of logits is
    # not a problem when computing xentropy, because padded_cross_entropy_loss
    # resolves the shape on the TPU.
    logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

    # Calculate model loss.
    # xentropy contains the cross entropy loss of every nonpadding token in the
    # targets.
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, targets, params["label_smoothing"], params["vocab_size"])
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    # Save loss as named tensor that will be logged with the logging hook.
    tf.identity(loss, "cross_entropy")

    if mode == tf.estimator.ModeKeys.EVAL:
      if params["use_tpu"]:
        # host call functions should only have tensors as arguments.
        # This lambda pre-populates params so that metric_fn is
        # TPUEstimator compliant.
        metric_fn = lambda logits, labels: (
            metrics.get_eval_metrics(logits, labels, params=params))
        eval_metrics = (metric_fn, [logits, labels])
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, predictions={"predictions": logits},
            eval_metrics=eval_metrics)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, predictions={"predictions": logits},
          eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op, metric_dict = get_train_op_and_metrics(loss, params)

      # Epochs can be quite long. This gives some intermediate information
      # in TensorBoard.
      metric_dict["minibatch_loss"] = loss
      if params["use_tpu"]:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, loss=loss, train_op=train_op,
            host_call=tpu_util.construct_scalar_host_call(
                metric_dict=metric_dict, model_dir=params["model_dir"],
                prefix="training/")
        )
      record_scalars(metric_dict)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



with open('./params.yml','r') as f:
  params=yaml.load(f)





# Train and evaluate transformer model
estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=save_dir, params=params)


estimator.train(
    input_fn,
    steps=None,
    hooks=None)




# import tensorflow as tf

# tgt = tf.placeholder(tf.int32,[None],name="tgt")
# # tgt_in=tf.concat((tf.strided_slice(tgt,[0,0],[tf.shape(tgt)[0],2],[1,1]),tf.strided_slice(tgt,[0,3],[tf.shape(tgt)[0],tf.shape(tgt)[1]],[1,1])),axis=1)
# sess = tf.Session()
# a = tf.reduce_sum(tgt)
# print(sess.run(a,{tgt:[1,2,3]}))
# # print(sess.run(tgt_in,{tgt:[[1,2,3,4,5],[4,5,6,7,8]]}))


