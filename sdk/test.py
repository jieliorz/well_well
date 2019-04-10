from config.params import Config
from utils.prepare_data import semi_to_dataset,produce_semi
from utils.tokenization import Tokenizer
from utils.data_helper import DataSet
from model.rnn.rnnModel import TextRnn

config = Config(project_name='poc',
				model_type='rnn',
				max_length=None,
				is_tgt_label=True,
				update_vocab=False,
				n_observations=None)

# raw to semi
n_observations,max_len = produce_semi(config.params)

if not config.init_params['n_observations']:
	config.set_params({'n_observations':n_observations})
if not config.init_params['max_length']:
	config.set_params({'max_length':max_len})
print(config.params)
tokenizer = Tokenizer(config.params)
config.set_params({'vocab_size':tokenizer.vocab_size})
# semi to tfrecord(including Tokenization)
semi_to_dataset(config.params,tokenizer)

dataset_obj=DataSet(config.params)
dataset=dataset_obj.prepare_dataset()

# model=TextRnn(config.params)
# model.train(dataset)

# sentence='打开收音机'
# ret,ret_len = tokenizer.encode(sentence,padding=True)
# prediction=model.predict(ret,ret_len)
# print(prediction)