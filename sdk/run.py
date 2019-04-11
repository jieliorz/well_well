from model.seq2seq import Seq2SeqModel

final_params_path = os.join(config.init_params['model_dir'],'params.json')
with open(final_params_path,'w') as f:
	json.dump(config.init_params,f)

# dataset_obj=DataSet(config.params)
# dataset=dataset_obj.prepare_dataset()

# model=TextRnn(config.params)
# model.train(dataset)

# sentence='打开收音机'
# ret,ret_len = tokenizer.encode(sentence,padding=True)
# prediction=model.predict(ret,ret_len)
# print(prediction)