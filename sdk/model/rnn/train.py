import sys
sys.path.append('../../')
print(sys.path)
from model.rnn.rnnModel import TextRnn
from utils.data_helper import DataSet
import yaml


with open('params.yml','r') as f:
	params=yaml.load(f)


dataset_obj=DataSet(params)
dataset=dataset_obj.prepare_dataset()


model=TextRnn()
model.train(dataset)

