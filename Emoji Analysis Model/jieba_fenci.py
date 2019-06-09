#encoding=utf-8
from __future__ import unicode_literals
from Data_Option import Data_Option
import sys
import jieba.analyse
import numpy as np
import re
import json

from gensim.models import word2vec
import time

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append("../")
r = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+")


def stopword_filter(stopwords, seq_words):
    filter_words_list = []
    # 停用词过滤
    for word in seq_words:
        if word not in stopwords:
            if word == '！':
                print('GG')
            filter_words_list.append(word)

    return filter_words_list

if __name__ == '__main__':

    start = time.process_time()
    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)

    """ 加载停词表 """
    stopwords = dp.get_stop_words()

    """ 获得词库 """
    data_words = []
    for i in range(train_data.__len__()):
        if(i%1000 == 0):
            print('training:',i)
        sentence = r.sub('', str(train_data[i]))
        seg_list = jieba.lcut(train_data[i])
        data_words.append(stopword_filter(stopwords,seg_list))

    """ 存储结果 （用json 序列化）"""
    with open('files/data_words.txt','w', encoding='UTF-8-sig') as FD:
        FD.write(json.dumps(data_words))

    with open('files/data_words.txt','r', encoding='UTF-8-sig') as FR:
        content = json.loads(FR.read())

    # print(content)
    FD.close()
    FR.close()

    """ 训练模型 """
    model=word2vec.Word2Vec(data_words, size=100)
    end = time.process_time()
    print('Running time: %s Seconds'%(end-start))

    """ 保存（加载）模型 """
    model.save('models/second_model_100d')
    # model = word2vec.Word2Vec.load('models/first_model')