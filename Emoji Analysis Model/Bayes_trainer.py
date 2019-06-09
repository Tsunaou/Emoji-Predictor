import numpy as np
import time
import sklearn
import pickle
from Data_Option import  Data_Option
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer

if __name__ == '__main__':

    start = time.process_time()

    d_size = 50

    sentence_vec_filepath = 'models/sentence_vec_'+str(d_size)+'d.npy'
    # model_save_path = 'models/random_forest_second_'+str(d_size)+'.pickle'
    model_save_path = 'models/GaussianNB'+str(d_size)+'.pickle'

    print('dsize=',d_size)
    print(sentence_vec_filepath)
    print(model_save_path)

    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)
    ematrix2 = np.load(sentence_vec_filepath)

    print('得到向量，开始训练：')

    x = ematrix2
    y = np.array(list(train_dict.values())).reshape(-1, 1)

    # x = x[0:1000]
    # y = y[0:1000]


    dtc = BernoulliNB()
    # dtc = MultinomialNB()
    x = Imputer().fit_transform(x)
    y = Imputer().fit_transform(y)

    scores = []
    kf = KFold(n_splits=10, shuffle=False)
    index_round = 0
    index_max = 0
    tests_xdatas = []
    tests_ydatas = []
    pre_datas = []

    max_score = 0

    models = []

    for train_index, test_index in kf.split(x, y):
        end = time.process_time()
        print('round', index_round,'-'*20)
        print('Running time: %s Seconds' % (end - start))
        index_round = index_round + 1

        # if index_round > 3:
        #     break

        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model = dtc.fit(x_train, y_train.ravel())
        models.append(model)

        y_predict = model.predict(x_test)

        tests_xdatas.append(x_test)
        tests_ydatas.append(y_test)
        pre_datas.append(y_predict)

        score = sklearn.metrics.f1_score(y_test, y_predict, average="micro")

        print("score = " + str(score))

        if score > max_score:
            max_score = score
            index_max = index_round
            # with open(model_save_path, 'wb') as fp:
            #     print('save model:',index_max)
            #     pickle.dump(model, fp)
        scores.append(score)




    target_names = list(emojis_dict.keys())
    label = tests_ydatas[index_max]
    pre = pre_datas[index_max]
    print(classification_report(label, pre))

