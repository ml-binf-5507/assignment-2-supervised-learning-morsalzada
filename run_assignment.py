import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler   
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart_disease_uci(1) (1).csv')
print(df.head())

#Data Pre-processing
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['thal'] = df['thal'].map({'normal': 0, 'fixed defect': 1, 'reversible defect': 2})
df = df.drop(columns=['dataset'])
numeric_cols =df.select_dtypes(include='number').columns
df[numeric_cols]= df[numeric_cols].fillna(df[numeric_cols].median())
categorical_cols = df.select_dtypes(exclude='number').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df = pd.get_dummies(df, drop_first=True)
x_reg = df.drop(columns=['chol'])
y_reg = df['chol']
y_clf = df['num'].apply(lambda x: 1 if x > 0 else 0)
x_clf = df.drop(columns=['num', 'chol'])

# Task 1
x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg, y_reg, test_size=0.2, random_state=42)
x_clf_train, x_clf_test, y_clf_train, y_clf_test = train_test_split(x_clf, y_clf, test_size=0.2, random_state=42)
scaler_reg = StandardScaler()
scaler_clf = StandardScaler()
x_reg_train = scaler_reg.fit_transform(x_reg_train)
x_reg_test = scaler_reg.transform(x_reg_test)
x_clf_train = scaler_clf.fit_transform(x_clf_train)
x_clf_test = scaler_clf.transform(x_clf_test)
print("First 5 scaled regression rows:")
print(x_reg_train[:5])
print("First 5 scaled classification rows:")
print(x_clf_train[:5])

# Task 2
