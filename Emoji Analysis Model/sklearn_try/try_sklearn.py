from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn import tree
from sklearn.model_selection import train_test_split,KFold
from sklearn.naive_bayes import MultinomialNB
import sklearn
import graphviz
import os
import pickle

iris = load_diabetes()
x = iris['data']
for i in range(len(x)):
    for j in range(len(x[i])):
        if(x[i][j]<0):
            x[i][j] = -x[i][j]

y = iris['target']
# dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc = MultinomialNB()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
clf = dtc.fit(x_train, y_train)
print(clf.predict(x_test))
print(y_test)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# os.environ["PATH"] += os.pathsep + 'F:/Program Files/Graphviz2.38/bin/'
# graph.render("iris", view=True)

scores = []
kf = KFold(n_splits=10,shuffle=False)
index_round = 1
index_max = 1
max_score = 0

for train_index, test_index in kf.split(x, y):
    print('round',index_round)
    index_round = index_round+1

    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    model = dtc.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = sklearn.metrics.f1_score(y_test, y_predict, average="micro")

    print("score = " + str(score))

    if score > max_score:
        max_score = score
        index_max = index_round
        # with open('../models/clf.pickle', 'wb') as fp:
        #     pickle.dump(model, fp)
    scores.append(score)

