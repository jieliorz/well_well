from config.params import Config
from utils.prepare_data import semi_to_dataset,produce_semi
from utils.tokenization import Tokenizer
from utils.data_helper import DataSet
import os
import yaml
config = Config(project_name='moli',
				model_type='seq2seq',
				update_semi=True,
				update_dataset=True,
				max_length=50,
				is_tgt_label=False,
				update_vocab=True,
				n_observations=10000)

if config.init_params['update_semi']:
	# raw to semi
	produce_semi(config.params)

tokenizer = Tokenizer(config.params)
config.set_params({'vocab_size':tokenizer.vocab_size})

if config.init_params['update_dataset']:
	# semi to tfrecord(including Tokenization)
	n_observations=semi_to_dataset(config.params,tokenizer)

if not config.init_params['n_observations']:
	config.set_params({'n_observations':n_observations})


print(config.params)

final_params_path = os.path.join(config.init_params['model_dir'],'params.yml')
with open(final_params_path,'w') as f:
	yaml.dump(config.init_params,f)

dataset_obj=DataSet(config.params)
dataset=dataset_obj.prepare_dataset()
