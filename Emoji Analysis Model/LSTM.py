#coding:utf-8

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

import numpy as np
import time
import sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import Imputer



class Data_Option():


    def __init__(self) -> None:
        super().__init__()

        self.emojis_filename = 'EmojiDataSet/emoji.data'
        self.train_small_filename = 'EmojiDataSet/small_data/small_train.data'
        self.label_small_filename = 'EmojiDataSet/small_data/small_train.solution'
        self.train_filename = 'EmojiDataSet/train.data'
        self.test_filename = 'EmojiDataSet/test.data'
        self.label_filename = 'EmojiDataSet/train.solution'

        self.emojis_dict = {}  # 表情和数字的字典 {(No, Emoji)}
        self.train_dict = {}
        self.train_data = []
        self.train_label = []
        self.emojis_classfier = {}
        self.cxk_dict = {}
        self.stopwords = []
        self.test_data = []

    def get_test_data(self):
        """ 测试 """
        with open(self.test_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.test_data.append(line.strip('\n').split('\t')[1])
            f.close()
        return self.test_data

    def get_stop_words(self):
        buff = []
        with open('files/stop', encoding='UTF-8-sig') as fp:
            for ln in fp:
                buff.append(ln.strip('\n'))
        self.stopwords = buff
        self.stopwords.append(' ')
        return self.stopwords

    def get_train_data(self,isSmall=False):
        if isSmall:
            self.train_filename = self.train_small_filename
            self.label_filename = self.label_small_filename

        """ 读取emoji对应关系 """
        with open(self.emojis_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip('\n').split('\t')
                self.emojis_dict[tmp[1]] = tmp[0]
            f.close()

        """ 读取训练数据 """
        with open(self.train_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.train_data.append(line.strip('\n'))
            f.close()

        """ 读取训练标签 """
        with open(self.label_filename, encoding='UTF-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                self.train_label.append(line.strip('\n').strip('{}'))
            f.close()

        assert (self.train_label.__len__() == self.train_data.__len__())

        for i in range(self.train_data.__len__()):
            '''这里就是用数字代替表情了'''
            self.train_dict[self.train_data[i]] = self.emojis_dict[self.train_label[i]]
            # self.train_dict[self.train_data[i]] = self.train_label[i]

        """ 根据表情对数据集分类 """
        for k, v in self.train_dict.items():

            """ 虚假的 """
            # if k.__contains__('cxk') or k.__contains__('蔡徐坤') or k.__contains__('kun'):
            #     # print(k,' --> ',v)
            #     if cxk_dict.get(v)==None:
            #         cxk_dict[v] = 1
            #     else:
            #         cxk_dict[v] = cxk_dict[v] + 1

            """ 真实的 """
            if self.emojis_classfier.get(v) == None:
                sentence_list = []
                sentence_list.append(k)
                self.emojis_classfier[v] = sentence_list
            else:
                self.emojis_classfier[v].append(k)

        # for k, v in self.emojis_classfier.items():
        #     if k != '心':
        #         continue
        #     print(k, ' --> ')
        #     for sentence in v:
        #         print('\t' + sentence)

        return self.emojis_dict,self.train_data,self.train_label,self.train_dict,self.emojis_classfier



if __name__ == '__main__':

    start = time.process_time()

    d_size = 300

    sentence_vec_filepath = 'models/sentence_vec_'+str(d_size)+'d.npy'
    model_save_path = 'models/random_forest_second_'+str(d_size)+'.pickle'

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


    dtc = RandomForestClassifier(max_depth=20)
    x = Imputer().fit_transform(x)
    y = Imputer().fit_transform(y)

    scores = []
    kf = KFold(n_splits=10, shuffle=False)
    index_round = 1
    index_max = 1
    tests_datas = []
    pre_datas = []

    max_score = 0

    for train_index, test_index in kf.split(x, y):
        end = time.process_time()
        print('round', index_round,'-'*20)
        print('Running time: %s Seconds' % (end - start))
        index_round = index_round + 1

        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        print('Build model...')
        model = Sequential()
        model.add(Embedding(len(dict) + 1, 256, input_length=50))
        model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

        model.fit(xa, ya, batch_size=16, nb_epoch=10)  # 训练时间为若干个小时

        classes = model.predict_classes(xa)
        acc = np_utils.accuracy(classes, ya)
        print('Test accuracy:', acc)

        y_predict = model.predict(x_test)

        tests_datas.append(y_test)
        pre_datas.append(y_predict)

        score = sklearn.metrics.f1_score(y_test, y_predict, average="micro")

        print("score = " + str(score))

        if score > max_score:
            max_score = score
            index_max = index_round
            with open(model_save_path, 'wb') as fp:
                pickle.dump(model, fp)
        scores.append(score)




    target_names = list(emojis_dict.keys())
    label = tests_datas[index_max]
    pre = pre_datas[index_max]
    print(classification_report(label, pre))




