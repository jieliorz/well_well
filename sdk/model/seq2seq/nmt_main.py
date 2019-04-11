import logging
import tensorflow as tf
from model.seq2seq import Seq2SeqModel
import os

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename='seq.log')
logger = logging.getLogger('seq2seq')



def run_train():

	model = Seq2SeqModel(params)
	model.train(batch_iter,data_producer.tokenizer)

def run_infer():
	vocab_dir=params['vocab_dir']
	clean_data_dir=params['clean_data_dir']
	test_dataset_dir=params['test_dataset_dir']
	data_producer = data_helper.DataSet(vocab_dir,clean_data_dir)
	sos_id=data_producer.tokenizer.subtoken_to_id_dict['<s>']
	eos_id=data_producer.tokenizer.subtoken_to_id_dict['</s>']
	params['sos_id']=sos_id
	params['eos_id']=eos_id
	vocab_size = len(data_producer.tokenizer.subtoken_list)
	params['vocab_size']=vocab_size
	params['maximum_iterations']=30
	params["batch_size"]=1
	decode_mode=params['decode_mode']

	model = Seq2SeqModel(params)
	while 1:
		sentence = input('you say: ')
		subtoken = data_producer.tokenizer.encode(pre_process(sentence,keep_sep=True))
		output=model.infer(subtoken)
		if decode_mode == 'greedy':
			output=[int(i) for i in output[0]]
			res = data_producer.tokenizer.decode(output)
			print(res)
			print('\n')
		elif decode_mode == 'beam_search':
			for j in range(len(output)):
				print("output[i]",output[j])
				res=[int(i) for i in list(output[j])]
				res = data_producer.tokenizer.decode(res)
				print('answer {}:{}'.format(j,res))
			print('\n')

if __name__ == '__main__':
	run_train()
