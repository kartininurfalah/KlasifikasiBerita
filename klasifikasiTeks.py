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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stemming = PorterStemmer()
import re
import math
import numpy as np
import pandas as pd

#iris = datasets.load_iris()


PathTraining = '/home/helmisatria/@FALAH/KlasifikasiBerita/Training set'
PathTesting = '/home/helmisatria/@FALAH/KlasifikasiBerita/Testing set'
PathLabel = '/home/helmisatria/@FALAH/KlasifikasiBerita/Label kelas unt training dan testing set'
Class=['YES', 'NO']

# =============================================================================
# get Path
# =============================================================================
def getData(path):
    headline = []
    isiBerita = []
    childRootAll = []
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
    
        childRoot = []
        for child in root:
            childRoot.append(child.tag)
        childRootAll.append(childRoot)
        
        indexHeadline = childRoot.index('headline')
        indexText = childRoot.index('text')
        
        headline.append(root[indexHeadline].text)
        beritaPerP = []
        for p in (root[indexText]):
            beritaPerP.append(p.text)
        isiBerita.append(beritaPerP)
    return headline, isiBerita

# =============================================================================
# Tokenisasi
# =============================================================================
def tokenization(PreProses):
    headlineLower = ''
    deleteStopWordHeadline=''
    HeadlineTokenize=[]
    for index, line in enumerate(PreProses[0]):
        headlineLower = (re.sub(r'[.,\/#!$%\^&\*;:{}=\-_+`~()\"{0-9}]', ' ',line))
        deleteStopWordHeadline = word_tokenize(headlineLower.lower())
    #    headlineUnik = sorted(set(deleteStopWordHeadline),key = str.lower)
        HeadlineTokenize.append(deleteStopWordHeadline)
    IsiBeritaTokenisasiAll = []
    deleteStopWordBerita=''
    for index, line in enumerate(PreProses[1]):
        isiBeritaLower=''
        IsiBeritaTokenisasi = []
        for row, teks in enumerate(line):
            isiBeritaLower =(re.sub(r'[.,\/#!$%\^&\*;:{}=\-_+`~()\"{0-9}]','',teks))
            deleteStopWordBerita = word_tokenize(isiBeritaLower.lower())
    #        beritaUnik = sorted(set(deleteStopWordBerita),key = str.lower)
            IsiBeritaTokenisasi.append(deleteStopWordBerita)
        IsiBeritaTokenisasiAll.append(IsiBeritaTokenisasi)
    return HeadlineTokenize,IsiBeritaTokenisasiAll

# =============================================================================
# Stemming 
# =============================================================================
def stemmingData(dataTokenize):
    StemmingBerita=[]
    StemmingHeadline=[]
    for head,value in enumerate(dataTokenize[0]):
        tempStem = []
        for line,teks in enumerate(value):
            tempStem.append(stemming.stem(teks))
        StemmingHeadline.append(tempStem)
    for indeks1,value1 in enumerate(dataTokenize[1]):
        temp1=[]
        for indeks2, value2 in enumerate(value1):
            temp2=[]
            for indeks3,value3 in enumerate(value2):
                temp2.append(stemming.stem(value3))
            temp1.append(temp2)
        StemmingBerita.append(temp1)
    return StemmingHeadline,StemmingBerita

# =============================================================================
# count for TF
# =============================================================================
def countForTF(Data):
    CountHeadline=[]
    CountBerita=[]
    for i,j in enumerate(Data[0]):
        tempCount=[]
        for m,n in enumerate(j):
            tempCount.append(j.count(n))
        CountHeadline.append(tempCount) 
    for first,val in enumerate(Data[1]):
        tempCount1=[]
        for second,inVal in enumerate(val):
            tempCount2=[]
            for inside,value in enumerate(inVal):
                tempCount2.append(inVal.count(value))
            tempCount1.append(tempCount2)
        CountBerita.append(tempCount1)
    return CountHeadline, CountBerita

# =============================================================================
# get value of TF
# =============================================================================
def getTF(dataCount,dataStemm):
    TfHeadline = []
    for x,y in enumerate(dataCount[0]):
       tempTF=[]
       for indeks,num in enumerate(y):
           tempTF.append(num / len(dataStemm[0]))
       TfHeadline.append(tempTF)
    TfBerita=[]
    for first,val in enumerate(dataCount[1]):
        tempTF1=[]
        for second,inVal in enumerate(val):
            tempTF2=[]
            for inside,value in enumerate(inVal):
                tempTF2.append(value / len(dataStemm[1]))
            tempTF1.append(tempTF2)
        TfBerita.append(tempTF1)
    return TfHeadline, TfBerita

# =============================================================================
# Preproses Data Training
# =============================================================================
PreProses=getData(PathTraining)
AllDataTokenize=tokenization(PreProses)
DataStemming=stemmingData(AllDataTokenize)
DataCount=countForTF(DataStemming)
FindTF=getTF(DataCount, DataStemming)

# =============================================================================
# Preproses Data Testing
# =============================================================================
getDataTesting=getData(PathTesting)
tokenizeTesting=tokenization(getDataTesting)
DataTestStemming=stemmingData(tokenizeTesting)
countDataTest=countForTF(DataTestStemming)
GetTFTesting=getTF(countDataTest,DataTestStemming)

#gnb = GaussianNB()
#y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#print("Number of mislabeled points out of a total %d points : %d"
#      (iris.data.shape[0],(iris.target != y_pred).sum()))
##Number of mislabeled points out of a total 150 points : 6
#
#X = [[0, 0], [1, 1]]
#y = [0, 1]
#clf = svm.SVC()
#clf.fit(X, y)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)