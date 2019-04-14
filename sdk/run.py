from model.seq2seq.seq2seq import Seq2SeqModel

model=Seq2SeqModel()
# model.train()
model.infer()
while 1:
	sentence = input('you:')
	model.predict(sentence)

# sentence='打开收音机'
# ret,ret_len = tokenizer.encode(sentence,padding=True)
# prediction=model.predict(ret,ret_len)
# print(prediction)











# import tensorflow as tf

# tgt = tf.placeholder(tf.int32,[None],name="tgt")
# # tgt_in=tf.concat((tf.strided_slice(tgt,[0,0],[tf.shape(tgt)[0],2],[1,1]),tf.strided_slice(tgt,[0,3],[tf.shape(tgt)[0],tf.shape(tgt)[1]],[1,1])),axis=1)
# sess = tf.Session()
# a = tf.reduce_sum(tgt)
# print(sess.run(a,{tgt:[1,2,3]}))
# # print(sess.run(tgt_in,{tgt:[[1,2,3,4,5],[4,5,6,7,8]]}))


