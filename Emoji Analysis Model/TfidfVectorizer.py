from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import sklearn
import jieba.analyse
from Data_Option import Data_Option
import numpy as np
import json



if __name__ == '__main__':
    dp = Data_Option()
    emojis_dict, train_data, train_label, train_dict, emojis_classfier \
        = dp.get_train_data(isSmall=False)

    with open('files/data_words.txt','r', encoding='UTF-8-sig') as FR:
        content = json.loads(FR.read())

    with open('files/test_data_words.txt','r', encoding='UTF-8-sig') as FT:
        content_test = json.loads(FT.read())

    data_words = []
    for sentence in content:
        data_words.append(" ".join(sentence))

    data_test_words = []
    for sentence in content_test:
        data_test_words.append(" ".join(sentence))

    emoji_data = data_words
    emoji_target = np.array(list(train_dict.values())).reshape(-1, 1)


    # 2 分割训练数据和测试数据
    max_scores = 0
    my_pre = []
    for i in range(2):
        print("Round ",i,'-'*10)
        x_train, x_test, y_train, y_test = train_test_split(emoji_data,
                                                            emoji_target,
                                                            test_size=0.1, shuffle=True)

        tfid_stop_vec = TfidfVectorizer()
        # tfid_stop_vec = CountVectorizer()
        x_tfid_stop_train = tfid_stop_vec.fit_transform(x_train)
        x_tfid_stop_test = tfid_stop_vec.transform(x_test)

        feature = tfid_stop_vec.transform(data_test_words)

        # mnb_tfid_stop = MultinomialNB()
        mnb_tfid_stop = MLPClassifier()
        mnb_tfid_stop.fit(x_tfid_stop_train, y_train.ravel())  # 学习
        mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(x_tfid_stop_test)  # 预测
        pre = mnb_tfid_stop.predict(feature)

        # print("更加详细的评估指标:\n", classification_report(y_test, mnb_tfid_stop_y_predict))
        score = sklearn.metrics.f1_score(y_test, mnb_tfid_stop_y_predict, average="micro")
        print("score = " + str(score))
        print(pre)
        if score > max_scores:
            max_scores = score
            my_pre = pre