#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
pd.options.display.max_columns = 999



df0 = pd.read_csv (r'FILEPATH' )
df1 = pd.read_csv (r'FILEPATH')
df= df0.append(df1, ignore_index=True)


X = df.drop(['Mayo Score'],axis=1)
 
y = df['Mayo Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)


train_df = pd.DataFrame(X_train)
test_df = pd.DataFrame(X_test)
vectorizer = CountVectorizer(ngram_range=(1, 3))
train_matrix = vectorizer.fit_transform(train_df['Notes'])
test_matrix = vectorizer.transform(test_df['Notes'])

logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(train_matrix,y_train)
y_pred_test = logreg.predict(test_matrix)
acc = accuracy_score(y_test,y_pred_test)
fscore=f1_score(y_test,y_pred_test,average='weighted')
print("Accuracy of LogisticRegression: %.2f%%" % (acc * 100.0))
print("Fscore of LogisticRegression: %.2f%%" % (fscore * 100.0))


gnb = GaussianNB()
X=train_matrix.toarray()
test_matrix=test_matrix.toarray()
gnb.fit(X,y_train)

y_pred_test = gnb.predict(test_matrix)

fscore=f1_score(y_test,y_pred_test,average='weighted')
acc = accuracy_score(y_test,y_pred_test)
print("Accuracy of GaussianNB: %.2f%%" % (acc * 100.0))
print("Fscore of GaussianNB: %.2f%%" % (fscore * 100.0))

clf = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')

clf.fit(train_matrix,y_train)
y_pred3 = clf.predict(test_matrix)
acc3 =   accuracy_score(y_test,y_pred3)
fscore=f1_score(y_test,y_pred3,average='weighted')
print("Accuracy of KNN: %.2f%%" % (acc3 * 100.0))
print("Fscore of KNN: %.2f%%" % (fscore * 100.0))

svc1 = SVC(C=50,kernel='rbf',gamma=1)     

svc1.fit(train_matrix,y_train)
y_pred4 = svc1.predict(test_matrix)

fscore=f1_score(y_test,y_pred4,average='weighted')
acc4=    accuracy_score(y_test,y_pred4)
print("Accuracy of SVM: %.2f%%" % (acc4 * 100.0))
print("Fscore of SVM: %.2f%%" % (fscore * 100.0))

dt = DecisionTreeClassifier()
dt.fit(train_matrix,y_train)

y_pred67 = dt.predict(test_matrix)
fscore=f1_score(y_test,y_pred67)
acc2 = accuracy_score(y_test,y_pred67)
print("Accuracy of Decision Tree: %.2f%%" % (acc2 * 100.0))
print("Fscore of Decision Tree: %.2f%%" % (fscore * 100.0))


model = XGBClassifier(max_depth=5 ,num_classes=4)
model.fit(train_matrix,y_train)  
y_pred = model.predict(test_matrix) 


predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions) 
fscore=f1_score(y_test,predictions,average='weighted')
print("Accuracy of XGBoost: %.2f%%" % (accuracy * 100.0))
print("Fscore of XGBoost: %.2f%%" % (fscore * 100.0))

train_matrix = train_matrix.astype('float32')
test_matrix = test_matrix.astype('float32')   
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
clf = lgb.LGBMClassifier(objective='binary', verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000,)
clf.fit(train_matrix,y_train)
y_pred5=clf.predict(test_matrix)
acc5 =   accuracy_score(y_test,y_pred5)
fscore=f1_score(y_test,y_pred5)
print("Accuracy of LightGBM: %.2f%%" % (acc5 * 100.0))
print("Fscore of LightGBM: %.2f%%" % (fscore * 100.0))


rf = RandomForestClassifier()
rf.fit(train_matrix,y_train)
y_pred2 = rf.predict(test_matrix)

fscore=f1_score(y_test,y_pred2,average='weighted')
acc2 = accuracy_score(y_test,y_pred2)

print("Accuracy of Random Forest: %.2f%%" % (acc2 * 100.0))
print("Fscore of Random Forest: %.2f%%" % (fscore * 100.0))


et = ExtraTreesClassifier()
et.fit(train_matrix,y_train)
y_pred21 = et.predict(test_matrix)

fscore=f1_score(y_test,y_pred21,average='weighted')
acc2 = accuracy_score(y_test,y_pred21)

print("Accuracy of Extra Tree: %.2f%%" % (acc2 * 100.0))
print("Fscore of Extra Tree: %.2f%%" % (fscore * 100.0))


ab= AdaBoostClassifier()
ab.fit(train_matrix,y_train)
y_pred22 = ab.predict(test_matrix)

fscore2=f1_score(y_test,y_pred22,average='weighted')
acc22 = accuracy_score(y_test,y_pred22)

print("Accuracy of AdaBoost: %.2f%%" % (acc22 * 100.0))
print("Fscore of AdaBoost: %.2f%%" % (fscore2 * 100.0))


train_matrix = train_matrix.astype('float32')
test_matrix = test_matrix.astype('float32')   
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
clf = lgb.LGBMClassifier(objective='binary', verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000,)
clf.fit(train_matrix,y_train)
y_pred5=clf.predict(test_matrix)
acc5 =   accuracy_score(y_test,y_pred5)
fscore=f1_score(y_test,y_pred5)
print("Accuracy of LightGBM: %.2f%%" % (acc5 * 100.0))
print("Fscore of LightGBM: %.2f%%" % (fscore * 100.0))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred5)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred5))


CM = confusion_matrix(y_test, y_pred5)

TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


print("Sensitivity, hit rate, recall, or true positive rate",TPR)
print("Specificity or true negative rate",TNR)
print("Precision or positive predictive value",PPV)
print("Negative predictive value",NPV)
print("Fall out or false positive rate",FPR)
print("False negative rate",FNR)
print("False discovery rate",FDR)
print("Overall accuracy",ACC)



from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred5)



#Multi Classifier




df0 = pd.read_csv (r'FILEPATH')
df1 = pd.read_csv (r'FILEPATH')
df= df0.append(df1, ignore_index=True)


X = df.drop(['Mayo Score'],axis=1)
 
y = df['Mayo Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)



train_df = pd.DataFrame(X_train)
test_df = pd.DataFrame(X_test)
vectorizer = CountVectorizer(ngram_range=(1, 3))
train_matrix = vectorizer.fit_transform(train_df['Notes'])
test_matrix = vectorizer.transform(test_df['Notes'])


logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(train_matrix,y_train)
y_pred_test = logreg.predict(test_matrix)
acc = accuracy_score(y_test,y_pred_test)
fscore=f1_score(y_test,y_pred_test,average='weighted')
print("Accuracy of LogisticRegression: %.2f%%" % (acc * 100.0))
print("Fscore of LogisticRegression: %.2f%%" % (fscore * 100.0))



gnb = GaussianNB()
X=train_matrix.toarray()
test_matrix=test_matrix.toarray()
gnb.fit(X,y_train)

y_pred_test = gnb.predict(test_matrix)

fscore=f1_score(y_test,y_pred_test,average='weighted')
acc = accuracy_score(y_test,y_pred_test)
print("Accuracy of GaussianNB: %.2f%%" % (acc * 100.0))
print("Fscore of GaussianNB: %.2f%%" % (fscore * 100.0))



clf = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')

clf.fit(train_matrix,y_train)
y_pred3 = clf.predict(test_matrix)
acc3 =   accuracy_score(y_test,y_pred3)
fscore=f1_score(y_test,y_pred3,average='weighted')
print("Accuracy of KNN: %.2f%%" % (acc3 * 100.0))
print("Fscore of KNN: %.2f%%" % (fscore * 100.0))



dt = DecisionTreeClassifier()
dt.fit(train_matrix,y_train)

y_pred67 = dt.predict(test_matrix)
fscore=f1_score(y_test,y_pred67,average='weighted')
acc2 = accuracy_score(y_test,y_pred67)
print("Accuracy of Decision Tree: %.2f%%" % (acc2 * 100.0))
print("Fscore of Decision Tree: %.2f%%" % (fscore * 100.0))


model = XGBClassifier(max_depth=5 ,num_classes=4)
model.fit(train_matrix,y_train)  
y_pred = model.predict(test_matrix) 


predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions) 
fscore=f1_score(y_test,predictions,average='weighted')
print("Accuracy of XGBoost: %.2f%%" % (accuracy * 100.0))
print("Fscore of XGBoost: %.2f%%" % (fscore * 100.0))



svc1 = SVC(C=50,kernel='rbf',gamma=1)     

svc1.fit(train_matrix,y_train)
y_pred4 = svc1.predict(test_matrix)

fscore=f1_score(y_test,y_pred4,average='weighted')
acc4=    accuracy_score(y_test,y_pred4)
print("Accuracy of SVM: %.2f%%" % (acc4 * 100.0))
print("Fscore of SVM: %.2f%%" % (fscore * 100.0))



train_matrix = train_matrix.astype('float32')
test_matrix = test_matrix.astype('float32')   
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
clf = lgb.LGBMClassifier(objective='multiclass', verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000,)
clf.fit(train_matrix,y_train)
y_pred5=clf.predict(test_matrix)
acc5 =   accuracy_score(y_test,y_pred5)
fscore=f1_score(y_test,y_pred5,average='weighted')
print("Accuracy of LightGBM: %.2f%%" % (acc5 * 100.0))
print("Fscore of LightGBM: %.2f%%" % (fscore * 100.0))



rf = RandomForestClassifier()
rf.fit(train_matrix,y_train)
y_pred2 = rf.predict(test_matrix)

fscore=f1_score(y_test,y_pred2,average='weighted')
acc2 = accuracy_score(y_test,y_pred2)

print("Accuracy of Random Forest: %.2f%%" % (acc2 * 100.0))
print("Fscore of Random Forest: %.2f%%" % (fscore * 100.0))



et = ExtraTreesClassifier()
et.fit(train_matrix,y_train)
y_pred21 = et.predict(test_matrix)

fscore=f1_score(y_test,y_pred21,average='weighted')
acc2 = accuracy_score(y_test,y_pred21)

print("Accuracy of Extra Tree: %.2f%%" % (acc2 * 100.0))
print("Fscore of Extra Tree: %.2f%%" % (fscore * 100.0))



ab= AdaBoostClassifier()
ab.fit(train_matrix,y_train)
y_pred22 = ab.predict(test_matrix)

fscore2=f1_score(y_test,y_pred22,average='weighted')
acc22 = accuracy_score(y_test,y_pred22)

print("Accuracy of AdaBoost: %.2f%%" % (acc22 * 100.0))
print("Fscore of AdaBoost: %.2f%%" % (fscore2 * 100.0))



train_matrix = train_matrix.astype('float32')
test_matrix = test_matrix.astype('float32')   
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
clf = lgb.LGBMClassifier(objective='multiclass', verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000,)
clf.fit(train_matrix,y_train)
y_pred5=clf.predict(test_matrix)
acc5 =   accuracy_score(y_test,y_pred5)
fscore=f1_score(y_test,y_pred5,average='weighted')
print("Accuracy of LightGBM: %.2f%%" % (acc5 * 100.0))
print("Fscore of LightGBM: %.2f%%" % (fscore * 100.0))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred5)
print('Confusion matrix\n\n', cm)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred5))



from sklearn.metrics import multilabel_confusion_matrix
multilabel_confusion_matrix(y_test, y_pred5)




import numpy as np
cnf_matrix = confusion_matrix(y_test, y_pred5)
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)



# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
mean_absolute_error(y_test, y_pred5)


print("Sensitivity, hit rate, recall, or true positive rate",TPR)
print("Specificity or true negative rate",TNR)
print("Precision or positive predictive value",PPV)
print("Negative predictive value",NPV)
print("Fall out or false positive rate",FPR)
print("False negative rate",FNR)
print("False discovery rate",FDR)
print("Overall accuracy",ACC)
