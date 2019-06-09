import json
from Data_Option import  Data_Option
from gensim.models import word2vec
import time
import numpy as np
from gensim.models import word2vec

if __name__ == '__main__':

    d_size = 50
    vec_model_filepath = 'models/all_data_model/all_vec_model_'+str(d_size)+'d'  # 词向量
    sentence_vec_filepath = 'models/all_data_model/all_sentence_vec_'+str(d_size)+'d.npy'  # 句向量

    start = time.process_time()
    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)

    """ 读取句子并载入模型"""
    data_words = []
    d1 = []
    d2 = []
    with open('files/data_words.txt','r', encoding='UTF-8-sig') as FR:
        d1 = json.loads(FR.read())

    with open('files/test_data_words.txt','r', encoding='UTF-8-sig') as FT:
        d2 = json.loads(FT.read())

    data_words = d1+d2

    """ 存储结果 （用json 序列化）"""
    with open('files/all_data_words.txt','w', encoding='UTF-8-sig') as FD:
        FD.write(json.dumps(data_words))

    model=word2vec.Word2Vec(data_words, size=d_size)

    model.save(vec_model_filepath)

    data_words = d1

    """ 得到向量 """
    ematrix = np.zeros([train_data.__len__(),d_size],dtype='float32')
    line = ematrix[0].copy()
    not_in_cnts = 0
    for i in range(data_words.__len__()):
        if(i%1000 == 0):
            print('training:',i)
        size = 0
        line_vec = line.copy()

        # print(line_vec)
        for word in data_words[i]:
            try:
                line_vec = line_vec + model.wv.get_vector(word)
                size = size + 1
            except KeyError:
                print(word,'is not in vocabulary')
                not_in_cnts = not_in_cnts +1

        line_vec = line_vec / size
        ematrix[i] = line_vec

    """ 保存和载入句子向量 """
    np.save(sentence_vec_filepath, ematrix)

    end = time.process_time()
    print('Running time: %s Seconds'%(end-start))

