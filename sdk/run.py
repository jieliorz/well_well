from model.seq2seq import Seq2SeqModel

model=Seq2SeqModel()
model.train()

sentence='打开收音机'
ret,ret_len = tokenizer.encode(sentence,padding=True)
prediction=model.predict(ret,ret_len)
print(prediction)