from config.params import Config
from utils.prepare_data import semi_to_dataset,produce_semi
from utils.tokenization import Tokenizer
from utils.data_helper import DataSet
import os
import yaml


model_type='rnn'

model_params_file = os.path.join('model',model_type,'params.yml')
with open(model_params_file,'r') as f:
	params=yaml.load(f)


params.update({
	'num_epochs':1,
	'semi_dir':params['semi_dir'].replace('/data/','/test/'),
	'raw_data_dir': params['raw_data_dir'].replace('/data/','/test/'),
	'dataset_dir': params['dataset_dir'].replace('/data/','/test/'),
	'src_file': params['src_file'].replace('/data/','/test/'),
	'tgt_file': params['tgt_file'].replace('/data/','/test/'),
	'dataset_file':params['dataset_file'].replace('/data/','/test/'),
	})

print(params)
produce_semi(params)

tokenizer = Tokenizer(params)
semi_to_dataset(params,tokenizer)

