from .model.transformer import Transformer

def input_fn(dataset):
  dataset_obj=DataSet(params)
  dataset=dataset_obj.prepare_dataset()
  return dataset

with open(os.path.join(fpath,'params.yml'),'r') as f:
  params=yaml.load(f)

# Train and evaluate transformer model
estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=save_dir, params=params)


estimator.train(
    input_fn,
    steps=None,
    hooks=None)


