from __Data_Analysis import plot_conf_matrix
import __Feature_selection as Feature_selection

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrices
from sklearn.metrics import accuracy_score

Y = Feature_selection._get_target()

from sklearn.metrics import accuracy_score



## ------------------------ BaggingClassifier ------------------------ #
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

X = Feature_selection._get_Data(85,2)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

clf = BaggingClassifier(base_estimator=SVC(gamma="scale"),n_estimators=20, random_state=0)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
       
cm = metrices.confusion_matrix(y_test,y_pred)

plot_conf_matrix(cm)

print("BaggingClassifier")

print("Accuracy : %.2f"%accuracy_score(y_test,y_pred))

## ---------------------- DecisionTreeClassifier -------------------------- #
from sklearn import tree

X = Feature_selection._get_Data(36,1)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =1)

clf = tree.DecisionTreeClassifier(max_depth=30)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
       
cm = metrices.confusion_matrix(y_test,y_pred)

plot_conf_matrix(cm)

print("DecisionTreeClassifier")

print("Accuracy : %.2f"%accuracy_score(y_test,y_pred))

## ---------------------- KNeighborsClassifier ---------------------------- #

from sklearn.neighbors import KNeighborsClassifier

X = Feature_selection._get_Data(85,3)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
       
cm = metrices.confusion_matrix(y_test,y_pred)

plot_conf_matrix(cm)

print("KNeighborsClassifier")

print("Accuracy : %.2f"%accuracy_score(y_test,y_pred))

## ---------------------- NaiveBayes ---------------------------

from sklearn.naive_bayes import GaussianNB

X = Feature_selection._get_Data(85,3)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

clf = GaussianNB()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

cm = metrices.confusion_matrix(y_test,y_pred)

plot_conf_matrix(cm)

print("NaiveBayes")

print("Accuracy : %.2f"%accuracy_score(y_test,y_pred))

## ---------------------------- RandomForestClassifier -----------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X = Feature_selection._get_Data(88,0)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state =0)

clf = RandomForestClassifier(max_depth=30, random_state=0, n_estimators=52)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

cm = metrices.confusion_matrix(y_test,y_pred)

plot_conf_matrix(cm)

print("RandomForestClassifier")

print("Accuracy : %.2f"%accuracy_score(y_test,y_pred))