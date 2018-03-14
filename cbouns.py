from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.ensemble import *
import numpy as np
from sklearn.metrics import *
import pandas as pd
from sklearn.tree import *
from threading import Thread
from xgboost.sklearn import *
from sklearn.neighbors import *

data = pd.read_csv("aps_failure_training_set_processed_8bit.csv")
data["class"][data["class"] > 0] = 1
data["class"][data["class"] < 0] = 0
skf = StratifiedKFold(n_splits=5)
x = data.iloc[:, 1:data.shape[1]]
y = data.iloc[:, 0]

predict = np.zeros((7, y.shape[0]))


def classifier(classy, train, test):
    global predict, x, y
    x_train, y_train, x_test, y_test = x.iloc[train, :], y[train], x.iloc[test, :], y[test]

    c = classy[1]
    c.fit(x_train, y_train)
    predict[classy[0]][test] = c.predict(x_test)
    print("Index of", classy[0], accuracy_score(y, predict[classy[0]]))


threads = []
for train, test in skf.split(x, y):
    x_train, y_train, x_test, y_test = x.iloc[train, :], y[train], x.iloc[test, :], y[test]

    for cl in [[0, GradientBoostingClassifier()], [1, LogisticRegression()], [2, ExtraTreesClassifier()],
               [3, RandomForestClassifier()], [4, DecisionTreeClassifier()], [5, XGBClassifier()],
               [6, KNeighborsClassifier()]]:
        th = Thread(target=classifier, args=(cl.copy(), train.copy(), test.copy()))
        th.start()
        threads.append(th)
for th in threads:
    th.join()
y_predict = []
for p in range(predict.shape[1]):
    sum = 0
    for i in range(predict.shape[0]):
        sum += predict[i][p]
    # print(sum)
    y_predict.append(1 if (sum / predict.shape[0] >= .4) else 0)

print(accuracy_score(y, y_predict) * 100)
