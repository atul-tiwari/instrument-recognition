import __Feature_selection as Feature_selection
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import matplotlib
import matplotlib.pyplot as plt

per = []
best_features = []

# 0  =  f_classif   
# 1  =  mutual_info_classif   
# 2  =  f_regression
# 3  =  mutual_info_regression  

Feature_Class = 0

for i_features in range(1,110,2):

    best_features.append(i_features)

    X = Feature_selection._get_Data(i_features,Feature_Class)
    Y = Feature_selection._get_target()

    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification


    clf = RandomForestClassifier(max_depth=30, random_state=0, n_estimators=52)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))

fig, ax = plt.subplots()
ax.plot(best_features, per,'b')

ax.set(xlabel='No of Best Features', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()


plt.show()


per = []
est = []
X = Feature_selection._get_Data(88,Feature_Class)
Y = Feature_selection._get_target()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

for i_est in range(5,150,2):

    est.append(i_est)

    clf = RandomForestClassifier(max_depth=30, random_state=0, n_estimators=i_est)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))


fig, ax = plt.subplots()
ax.plot(est, per,'g')

ax.set(xlabel='No of estimators', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()

plt.show()

per = []
depth = []
X = Feature_selection._get_Data(88,Feature_Class)
Y = Feature_selection._get_target()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

for i_dep in range(5,100,2):

    depth.append(i_dep)

    clf = RandomForestClassifier(max_depth=i_dep, random_state=0, n_estimators=52)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))


fig, ax = plt.subplots()
ax.plot(depth, per,'r')

ax.set(xlabel='Deapth of tree', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()


plt.show()