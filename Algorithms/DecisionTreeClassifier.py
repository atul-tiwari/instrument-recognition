from sklearn import tree
import __Feature_selection as Feature_selection
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 0  =  f_classif   
# 1  =  mutual_info_classif   
# 2  =  f_regression
# 3  =  mutual_info_regression  

Feature_Class = 0

per = []
best_features = []

for i_features in range(10,100,2):
   # print(i_features)
    best_features.append(i_features)

    X = Feature_selection._get_Data(i_features,Feature_Class)
    Y = Feature_selection._get_target()

    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.01,random_state =1)

    clf = tree.DecisionTreeClassifier(max_depth=20)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    #print(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))
    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))


fig, ax = plt.subplots()
ax.plot(best_features, per)

ax.set(xlabel='No of Best Features', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()

plt.show()

per = []
Max_Depth = []

X = Feature_selection._get_Data(36,Feature_Class)
Y = Feature_selection._get_target()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.01,random_state =1)


for i_Dep in range(10,100,2):
   # print(i_features)
    Max_Depth.append(i_Dep)

    clf = tree.DecisionTreeClassifier(max_depth=i_Dep)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn.metrics import accuracy_score

    #print(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))
    per.append(np.round(accuracy_score(y_test,y_pred)*100,decimals=2))

#f =36 , d =30


clf = tree.DecisionTreeClassifier(max_depth=30)

tree.plot_tree(clf.fit(X,Y))

fig, ax = plt.subplots()
ax.plot(Max_Depth, per,'r')

ax.set(xlabel='Depth of tree', ylabel='Percentage',
       title='Accuracy graph')
ax.grid()

plt.show()