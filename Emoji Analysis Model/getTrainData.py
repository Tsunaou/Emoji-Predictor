import  json
import sklearn
from Data_Option import  Data_Option
from gensim.models import word2vec
import time
import numpy as np

if __name__ == '__main__':

    d_size = 12
    vec_model_filepath = 'models/second_model_'+str(d_size)+'d'
    sentence_vec_filepath = 'models/sentence_vec_'+str(d_size)+'d.npy'

    start = time.process_time()
    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)

    """ 读取句子并载入模型"""
    data_words = []
    with open('files/data_words.txt','r', encoding='UTF-8-sig') as FR:
        data_words = json.loads(FR.read())
    model = word2vec.Word2Vec.load(vec_model_filepath)

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

