import glob
import pickle
import jieba
from concurrent.futures import as_completed,ThreadPoolExecutor
from collections import defaultdict
import logging
import os
import re

logger = logging.getLogger('seq2seq')


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().lower()
    # string = emoji.demojize(string)
    # 英文结尾
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"&gt;","",string)
    string = re.sub(r"\《.*\》|\<.*\>|\(.*\)|\{.*\}|\【.*\】|\[.*\]|\（.*\）", "", string)
    string = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9:]", "<sep>", string)
    
    # 数字
    string = re.sub(r"\d+", "<num>", string)
    string = re.sub(r"<num><sep><num>", "<num>", string)

    return string

def pre_process(sentence,keep_sep=False):
    '''
    return prepared word list 
    '''

    reg_sentence = clean_str(strQ2B(sentence))
   
    reg_sentence = re.split(r'(<[a-zA-Z_]+>)',reg_sentence)
    # print(reg_sentence)
    reg_sentence = [word for word in reg_sentence if word != '']
    
    reg_sentence = ' '.join(reg_sentence)
    # print(reg_sentence)

    if len(reg_sentence) != 0 or (not re.fullmatch('<[a-zA-Z_]+>',reg_sentence)):

        reg_sentence = [jieba.lcut(reg) if not re.fullmatch('<[a-zA-Z_]+>',reg) else reg for reg in reg_sentence.split(' ')]
        
        new_reg_sentence = []
        
        for word in reg_sentence:

            if len(word) < 1:
                continue
            if isinstance(word,list):
                for w in word:
                    if w == '<sep>':
                        if keep_sep:
                            if len(new_reg_sentence) and (new_reg_sentence[-1] != '<sep>'):
                                new_reg_sentence.append('<sep>')
                    else:
                        new_reg_sentence.append(w)
            else:
                if word == '<sep>':
                    if keep_sep:
                        if len(new_reg_sentence) and (new_reg_sentence[-1] != '<sep>'):
                            new_reg_sentence.append('<sep>')

                else:
                    new_reg_sentence.append(word)
        if len(new_reg_sentence) > 0:
            if new_reg_sentence[-1] == '<sep>':
                new_reg_sentence = new_reg_sentence[:-1]
            if new_reg_sentence[0] == '<sep>':
                new_reg_sentence = new_reg_sentence[1:]
            return ' '.join(new_reg_sentence)



if __name__ == '__main__':
    string='处理原始,,,数据'
    
    # string=',,,'
    print(pre_process(string,keep_sep=True))
