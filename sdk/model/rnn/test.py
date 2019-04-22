import sys
sys.path.append('../../')
print(sys.path)
from model.rnn.rnnModel import TextRnn
from utils.data_helper import DataSet
import glob
import yaml
import os


model_params_file = os.path.join('params.yml')
with open(model_params_file,'r') as f:
	params=yaml.load(f)



model=TextRnn()


files=glob.glob('test/*.txt')
batch_size = params['batch_size']
sentence_list = []
label_list = []
for file in files:
  with open(file,'r') as f:
    for line in f:
      line=line.strip()
      sentence=line.split('\t')[0]
      label=line.split('\t')[1]
      if len(sentence_list) == batch_size:
        ret=model.test(sentence_list)
        res=[ret[i]==label_list[i] for i in range(batch_size)]
        print(res)
        sentence_list = []
        label_list = []
  
      sentence_list.append(sentence)
      label_list.append(label)