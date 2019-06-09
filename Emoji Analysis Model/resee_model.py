import numpy as np
from Data_Option import Data_Option
import time
import sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer

"""
    d = 50 :  max = 1, depth = 20, trees = 20, score = 0.1458808809388649
    d = 300 : max = 2, score = 0.13375117300184206 
    
"""

if __name__ == '__main__':

    start = time.process_time()

    d_size = 300

    sentence_vec_filepath = 'models/sentence_vec_'+str(d_size)+'d.npy'
    # ml_model_path = 'models/random_forest_second_' + str(d_size) + '.pickle'
    ml_model_path = 'models/MLPClassifier/nn_default_300d_model_Ensemble_hard.pickle'

    print('resee model,dsize=',d_size)
    print(sentence_vec_filepath)

    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)
    ematrix2 = np.load(sentence_vec_filepath)

    print('读取训练向量，准备复现：')

    x = ematrix2
    y = np.array(list(train_dict.values())).reshape(-1, 1)

    # x = x[0:1000]
    # y = y[0:1000]


    x = Imputer().fit_transform(x)
    y = Imputer().fit_transform(y)

    scores = []
    kf = KFold(n_splits=10, shuffle=False)
    index_round = 1
    index_max = 1
    tests_xdatas = []
    tests_ydatas = []
    pre_datas = []

    max_score = 0

    f = open(ml_model_path, 'rb')
    model = pickle.load(f)
    f.close()

    for train_index, test_index in kf.split(x, y):
        end = time.process_time()
        print('round', index_round,'-'*20)
        print('Running time: %s Seconds' % (end - start))
        index_round = index_round + 1

        # if(index_round != 2):
        #     continue

        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        y_predict = model.predict(x_test)

        tests_xdatas.append(x_test)
        tests_ydatas.append(y_test)
        pre_datas.append(y_predict)

        score = sklearn.metrics.f1_score(y_test, y_predict, average="micro")

        print("score = " + str(score))
        scores.append(score)




    target_names = list(emojis_dict.keys())
    label = tests_ydatas[0]
    pre = pre_datas[0]
    print(classification_report(label, pre))

