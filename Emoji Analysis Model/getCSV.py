import pickle
import sklearn
from Data_Option import  Data_Option
import time
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer
import re
import  json
import jieba.analyse
from gensim.models import word2vec


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

    d_size = 50

    # ml_model_path = 'models/random_forest_third_tree_20_depth_20'+str(d_size)+'.pickle'
    ml_model_path = 'models/MLPClassifier/true_voting1_50d_model.pickle'

    sentence_vec_filepath = 'models/sentence_vec_'+str(d_size)+'d.npy'
    vec_model_filepath = 'models/second_model_'+str(d_size)+'d'

    #以下是用全部训练集得到的结果的向量
    # sentence_vec_filepath = 'models/all_data_model/all_sentence_vec_'+str(d_size)+'d.npy'  # 句向量
    # vec_model_filepath = 'models/all_data_model/all_vec_model_'+str(d_size)+'d'  # 词向量


    f = open(ml_model_path, 'rb')
    model = pickle.load(f)
    f.close()

    start = time.process_time()
    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)
    ematrix2 = np.load(sentence_vec_filepath)

    print('得到向量，开始训练：')

    x = ematrix2
    y = np.array(list(train_dict.values())).reshape(-1, 1)
    x = Imputer().fit_transform(x)
    y = Imputer().fit_transform(y)

    target_names = list(emojis_dict.keys())
    label = y
    pre = model.predict(x)
    print(classification_report(label, pre))

    # """ 测试停一下 """
    # exit(0)

    test_data = dp.get_test_data()

    vec_model_filepath = vec_model_filepath
    vec_model = word2vec.Word2Vec.load(vec_model_filepath)

    """ 加载停词表 """
    stopwords = dp.get_stop_words()

    """ 获得词库 """
    data_words = []
    # for i in range(test_data.__len__()):
    #     if (i % 1000 == 0):
    #         print('training:', i)
    #     sentence = r.sub('', str(test_data[i]))
    #     seg_list = jieba.lcut(test_data[i])
    #     data_words.append(stopword_filter(stopwords, seg_list))
    #
    #
    #
    # """ 存储结果 （用json 序列化）"""
    # with open('files/test_data_words.txt','w', encoding='UTF-8-sig') as FD:
    #     FD.write(json.dumps(data_words))

    with open('files/test_data_words.txt','r', encoding='UTF-8-sig') as FR:
        data_words = json.loads(FR.read())

    """ 得到向量 """
    ematrix = np.zeros([test_data.__len__(), d_size], dtype='float32')
    line = ematrix[0].copy()
    for i in range(data_words.__len__()):
        if (i % 1000 == 0):
            print('training:', i)
        size = 0
        line_vec = line.copy()

        # print(line_vec)
        for word in data_words[i]:
            try:
                line_vec = line_vec + vec_model.wv.get_vector(word)
                size = size + 1
            except KeyError:
                print(word, 'is not in vocabulary')

        line_vec = line_vec / size
        ematrix[i] = line_vec

    x = ematrix
    x = Imputer().fit_transform(x)
    pre = model.predict(x)
