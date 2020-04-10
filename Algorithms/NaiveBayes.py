from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
import __Feature_selection as Feature_selection
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 0  =  f_classif   
# 1  =  mutual_info_classif   
# 2  =  f_regression
# 3  =  mutual_info_regression  

Feature_Class = 1


per = []
best_features = []

for i_features in range(10,100,5):
    #print(i_features)
    best_features.append(i_features)

    X = Feature_selection._get_Data(i_features,Feature_Class)
    Y = Feature_selection._get_target()

    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

    clf = GaussianNB()

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))

fig, ax = plt.subplots()
ax.plot(best_features, per,'r')

ax.set(xlabel='No of Best Features', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()

plt.show()

