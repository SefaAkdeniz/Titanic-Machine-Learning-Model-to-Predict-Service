# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:59:52 2020

@author: sefa
"""

import pandas as pd
import numpy as np
import pickle

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_result_df = pd.read_csv("test_result.csv")
test_df = pd.merge(test_df, test_result_df, on='PassengerId')

df = pd.concat([train_df, test_df], ignore_index=True)
df.drop(columns=['Name','Ticket','Cabin'],inplace=True)

df["Embarked"] = df["Embarked"].fillna("C")
df["Fare"] = df["Fare"].fillna(np.mean(df[df["Pclass"] == 3]["Fare"]))

index_nan_age = list(df["Age"][df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = df["Age"][((df["SibSp"] == df.iloc[i]["SibSp"]) &(df["Parch"] == df.iloc[i]["Parch"])& (df["Pclass"] == df.iloc[i]["Pclass"]))].median()
    age_med = df["Age"].median()
    if not np.isnan(age_pred):
        df["Age"].iloc[i] = age_pred
    else:
        df["Age"].iloc[i] = age_med

X = df.iloc[:,2:12]
y = df.iloc[:,1]

X["Sex"] = [1 if i=="male" else 0  for i in X["Sex"]]
X = pd.get_dummies(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.svm import SVC
svc = SVC(C=525,gamma=0.01)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

filename = 'finalized_model.sav'
pickle.dump(svc, open(filename, 'wb'))

from sklearn import metrics
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))