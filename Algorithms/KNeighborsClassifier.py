import __Feature_selection as Feature_selection
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt

per = []
best_features = []


# 0  =  f_classif   
# 1  =  mutual_info_classif   
# 2  =  f_regression
# 3  =  mutual_info_regression  

Feature_Class = 0

for i_features in range(10,110,5):

   # print(i_features)

    best_features.append(i_features)

    X = Feature_selection._get_Data(i_features,Feature_Class)
    Y = Feature_selection._get_target()

    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)


    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))

fig, ax = plt.subplots()
ax.plot(best_features, per)

ax.set(xlabel='No of Best Features', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()


plt.show()


per = []
neighbors = []
X = Feature_selection._get_Data(85,Feature_Class)
Y = Feature_selection._get_target()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

for i_neighbors in range(2,100,2):

    neighbors.append(i_neighbors)

    clf = KNeighborsClassifier(n_neighbors=i_neighbors)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))


fig, ax = plt.subplots()
ax.plot(neighbors, per,'g')

ax.set(xlabel='No of neighbors', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()

plt.show()
