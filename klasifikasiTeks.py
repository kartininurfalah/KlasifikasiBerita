#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:55:05 2018

@author: kartini
"""

import xml.etree.ElementTree as ET
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import os

iris = datasets.load_iris()

tree = ET.parse('/media/kartini/Kuliah/KULIAH/Semester 6/Text Mining/dataset tugas/Training set/100850.xml')
root = tree.getroot()


path = '/media/kartini/Kuliah/KULIAH/Semester 6/Text Mining/dataset tugas/Training set'
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    tree = ET.parse(fullname)
    
print(root.tag)
childRoot = []
for child in root:
    childRoot.append(child.tag) 

headline = (root[1].text)
isiBerita = []
for p in (root[3]):
    isiBerita.append(p.text)
    print(isiBerita)

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      (iris.data.shape[0],(iris.target != y_pred).sum()))
#Number of mislabeled points out of a total 150 points : 6

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)