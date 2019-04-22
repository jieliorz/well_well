import sys
sys.path.append('../../')
print(sys.path)
from model.rnn.rnnModel import TextRnn
from utils.data_helper import DataSet
import glob
import yaml
import os

model_type='rnn'

model_params_file = os.path.join('params.yml')
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

model=TextRnn()

dataset_obj=DataSet(params)
dataset=dataset_obj.prepare_dataset()
model.test_batch(dataset)