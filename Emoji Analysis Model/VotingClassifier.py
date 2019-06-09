from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

import pickle
import sklearn
from Data_Option import  Data_Option
import time
import numpy as np
from sklearn import tree
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

    start = time.process_time()


    models = []
    sentence_vec_filepath = 'models/sentence_vec_'+str(d_size)+'d.npy'
    # sentence_vec_filepath = 'models/all_data_model/all_sentence_vec_'+str(d_size)+'d.npy'  # 句向量


    # for i in range(10):
    #     f = open('models/MLPClassifier/all_data_model/allnn_default_50d_model'+str(i)+'.pickle', 'rb')
    #     model = pickle.load(f)
    #     models.append(model)
    #     f.close()


    start = time.process_time()

    models.append(MLPClassifier(hidden_layer_sizes=(75,)))  # 1
    models.append(MLPClassifier(hidden_layer_sizes=(125,)))  # 2
    models.append(MLPClassifier())  # 3
    models.append(RandomForestClassifier(max_depth=20,n_estimators=20))  # 4
    models.append(RandomForestClassifier())  # 5
    models.append(LogisticRegression(multi_class='ovr',solver='sag'))  # 6
    models.append(GradientBoostingClassifier())  # 7
    models.append(LogisticRegression())  # 8
    models.append(BernoulliNB())  # 9
    models.append(KNeighborsClassifier)  # 10


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

    # x = x[0:1000]
    # y = y[0:1000]

    estimators = []
    zips_label = []
    for i in range(10):
        estimators.append(('MLP'+str(i),models[i]))
        zips_label.append('MLP'+str(i))

    eclf = VotingClassifier(estimators=estimators, voting='hard')
    models.append(eclf)
    zips_label.append('Ensemble')

    target_names = list(emojis_dict.keys())
    label = y
    eclf.fit(x,y)
    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))

    for i in range(models.__len__()):
        pre = models[i].predict(x)
        score = sklearn.metrics.f1_score(label, pre, average="micro")
        print(zips_label[i],'scores=',score)
        # print(classification_report(label, pre))



