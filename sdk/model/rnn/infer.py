import sys
sys.path.append('../../')
print(sys.path)
from model.rnn.rnnModel import TextRnn
import yaml

with open('./params.yml','r') as f:
  params=yaml.load(f)

model=TextRnn(params)
model.infer()
while 1:
  sentence = input('you:')
  res=model.predict(sentence)
  print('bot:{}\n'.format(res))
