# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 08:49:32 2020

@author: Janss
"""

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py


#%%preprocessing

#path to ikea file
path = "IKEA_SA_Furniture_Web_Scrapings_sss.csv"

#load dataset
ikea_df = pd.read_csv(path)

#delete first column from dataset
del ikea_df['Unnamed: 0']

#convert non-integer values to integers using pd.factorize()
for column_name in ikea_df.columns:
    if not (isinstance(ikea_df[column_name][0], float) or isinstance(ikea_df[column_name][0], int)):
        print(column_name)
        ikea_df[column_name] = pd.factorize(ikea_df[column_name])[0] + 1
    
#ikea_df['category'] = pd.factorize(ikea_df['category'])[0] + 1
#ikea_df['name'] = pd.factorize(ikea_df['name'])[0] + 1
#ikea_df['old_price'] = pd.factorize(ikea_df['old_price'])[0] + 1
#ikea_df['sellable_online'] = pd.factorize(ikea_df['sellable_online'])[0] + 1
#ikea_df['link'] = pd.factorize(ikea_df['link'])[0] + 1
#ikea_df['other_colors'] = pd.factorize(ikea_df['other_colors'])[0] + 1
#ikea_df['short_description'] = pd.factorize(ikea_df['short_description'])[0] + 1
#ikea_df['designer'] = pd.factorize(ikea_df['designer'])[0] + 1


#separate from not converted data
#ikea_data1 = ikea_df.iloc[:,0:5]
#ikea_data2 = ikea_df.iloc[:,7:10]
#ikea_new = [ikea_data1, ikea_data2]
#ikea_data = pd.concat(ikea_new,axis=1, join='inner')

#show correlations
scatter_matrix(ikea_data,alpha=0.5, figsize=(16,16))




X = ikea_data.drop('category', axis=1)
y = ikea_data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#%%prediction

names = ["Nearest Neighbors", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    #SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=20, max_features=5),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


score = []

failed = []

time_needed = []

start_time = 0
end_time = 0
total_time = 0
    
#for each defined model
for mod in classifiers:
    start_time = time.time()
    
    print(mod)
    model = mod
    
    #fit
    print("fit")
    model.fit(X_train, y_train)
    
    #predict
    print("predict")
    y_predict = model.predict(X_test)
    
    #save score
    print("save score")
    score.append(model.score(X_test, y_test))
    
    #calculate wrong predictions
    fail = 0
    print("check for correct predictions")
    for i in range(0,len(y_test)):
        if y_predict[i] != y_test[y_test.index[i]]:
            fail +=1
    print("save number of failed")
    failed.append(fail)
    
    #save time needed
    end_time = time.time()
    total_time = end_time - start_time
    time_needed.append(total_time)

    print("/n")







fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig.tight_layout()

ax1.plot(names, score)
ax1.set(ylabel='Score')
ax1.set_ylim(0,1)

ax2.plot(failed)
ax2.set(ylabel='Wrong predictions')
ax2.set_ylim(0,1000)

ax3.plot(np.log2(time_needed))
ax3.set(ylabel='Time')

ax3.set_xticklabels(names, rotation=90)