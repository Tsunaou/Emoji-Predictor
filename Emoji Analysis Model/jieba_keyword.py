#encoding=utf-8
from __future__ import unicode_literals
from Data_Option import Data_Option
import sys
import jieba.analyse
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append("../")


if __name__ == '__main__':

    """ 读取训练数据 """
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)

    """ 用 TF-IDF 计算关键词 """
    s = train_data.__str__()
    i = 0
    vector_dict = {}
    for x, w in jieba.analyse.extract_tags(s, withWeight=True,topK=3000):
        # print(i,':%s %s' % (x, w))
        # print(i,':',x,w)
        vector_dict[x] = i
        i = i+1

    print('关键字数量为：',i)

    """ 得到向量 """
    ematrix = np.zeros([train_data.__len__(),i],dtype='int8')
    for i in range(train_data.__len__()):
        seg_list = jieba.lcut(train_data[i])
        for word in seg_list:
            j = vector_dict.get(word)
            if j != None:
                ematrix[i][j] = 1

    print('得到向量，开始训练：')

    x = ematrix
    y = np.array(list(train_dict.values()))
    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    clf = dtc.fit(x_train, y_train)
    print(clf.predict(x_test))
    print(y_test)

    target_names = list(emojis_dict.keys())
    label = y_test
    pre = clf.predict(x_test)
    print(classification_report(label, pre))
