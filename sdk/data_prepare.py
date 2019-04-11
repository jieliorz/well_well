from config.params import Config
from utils.prepare_data import semi_to_dataset,produce_semi
from utils.tokenization import Tokenizer
from utils.data_helper import DataSet

config = Config(project_name='moli',
				model_type='seq2seq',
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
print('n_observations',n_observations)
print('max_length',max_len)
print(config.params)
tokenizer = Tokenizer(config.params)
config.set_params({'vocab_size':tokenizer.vocab_size})
# semi to tfrecord(including Tokenization)
semi_to_dataset(config.params,tokenizer)

dataset_obj=DataSet(config.params)
dataset=dataset_obj.prepare_dataset()
