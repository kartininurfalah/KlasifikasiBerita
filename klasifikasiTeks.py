#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:55:05 2018

@author: kartini
"""

import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stemming = PorterStemmer()
import re
import numpy as np
from collections import Counter

#iris = datasets.load_iris()
TRAINCOUNT = 577

PATHTRAINING = '/home/helmisatria/@FALAH/KlasifikasiBerita/Training set'
PATHTESTING = '/home/helmisatria/@FALAH/KlasifikasiBerita/Testing set'
PATHLABEL = '/home/helmisatria/@FALAH/KlasifikasiBerita/Label kelas unt traning dan testing set/Training set.txt'
PATHLABELTEST = '/home/helmisatria/@FALAH/KlasifikasiBerita/Label kelas unt traning dan testing set/Testing set.txt'
CLASS=['YES', 'NO']

# =============================================================================
# get Path
# =============================================================================
def getData(path):
    allDocuments = []
    for filename in os.listdir(path):
        itemIds = ''
        isiBerita = ''
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()
    
        childRoot = []
        for child in root:
            childRoot.append(child.tag)
        
        indexHeadline = childRoot.index('headline')
        indexText = childRoot.index('text')
        
        itemIds = root.attrib.get('itemid')
        isiBerita = root[indexHeadline].text
        
        for p in (root[indexText]):
            isiBerita += p.text
        allDocuments.append([itemIds, isiBerita])
            
#        allDocuments.append(document)
    
    return sorted(allDocuments, key=lambda doc: doc[0])[:TRAINCOUNT]

# =============================================================================
# Tokenisasi
# =============================================================================
def Cleaning(AllDocuments):
    CLEANDOCUMENTS = []
    for i, document in enumerate(AllDocuments):
        cleanContent = []
        for j, content in enumerate(document):
            if j == 0 :
                cleanContent.append(content)
                continue
            stopWords = set(stopwords.words('english'))
            cleanWords = (re.sub(r'[.,\/#!$%\^&\*;:{}=\-_+`~()\'\"{0-9}]', ' ',content.lower()))
            words = word_tokenize(cleanWords)
            wordsFilteredStemmed = []
            
            for w in words:
                if w not in stopWords:
                    wordsFilteredStemmed.append(stemming.stem(w))
            
            cleanContent.append(wordsFilteredStemmed)
            
        CLEANDOCUMENTS.append(cleanContent)
    
    return CLEANDOCUMENTS

# =============================================================================
# CORPUS
# =============================================================================

def CORPUS(Data):
    WORDS = []
    for i, document in enumerate(Data):
        for j, content in enumerate(document):
            if j == 0: continue
            for k, word in enumerate(content):
                WORDS.append(word)
    return WORDS

# =============================================================================
# count for TF
# =============================================================================
def TF(Data, CORPUS):
    RESULT = []
    c = Counter(CORPUS)
    for i, document in enumerate(Data):
        dataTF = []
        dataIDF = []
        for j, word in enumerate(CORPUS):
            countWord = document[1].count(word)
            tf = countWord / len(document[1])
            
            countWordAllDocs = c[word]
            idf = tf * np.log(len(Data) / countWordAllDocs)
            
            dataTF.append(tf)
            dataIDF.append(idf)
        RESULT.append([dataTF, dataIDF])
    return [item[0] for item in RESULT], [item[1] for item in RESULT]
    
# =============================================================================
# DATASET
# =============================================================================

LabelTrain   = np.genfromtxt(PATHLABEL, delimiter=' ', dtype=str)[:, 1][:TRAINCOUNT]
LabelTest    = np.genfromtxt(PATHLABELTEST, delimiter=' ', dtype=str)[:, 1][:TRAINCOUNT]
AllDocuments = getData(PATHTRAINING)
isTesting=False
# =============================================================================
# 
# Testing?
# 
#LabelTrain = LabelTest
#AllDocuments = getData(PATHTESTING)
#isTesting=True
# 
# =============================================================================
# =============================================================================
# Preproses Data Training
# =============================================================================
AllDocumentsTokenized = Cleaning(AllDocuments)
Corpus                = set(CORPUS(AllDocumentsTokenized))
Tf  , TfIdf           = TF(AllDocumentsTokenized, Corpus)

# =============================================================================
# SVM
# =============================================================================

def Accuracy(Data, Validator):
    count = 0
    for i, x in enumerate(Data):
        if (x==Validator[i]): count+= 1
    return count/len(Data)

def SVM():
    ids = [item[0] for item in AllDocumentsTokenized]
    clf = svm.SVC(C=100, gamma=0.1)
    clf.fit(TfIdf, LabelTrain)    
    result = []
    if (isTesting == False):
        result = cross_val_score(clf, TfIdf, LabelTrain, cv=10)
        
        concatenated = np.column_stack((ids, clf.predict(TfIdf)))
    else:
        result = Accuracy(clf.predict(TfIdf), LabelTrain)
        concatenated = np.column_stack((ids, clf.predict(TfIdf)))
        
    return result, concatenated

AccuracySVM, SVMTrain = SVM()

# =============================================================================
# Naive Bayes
# =============================================================================
def NaiveBayes():
    content = [item[1] for item in AllDocumentsTokenized][:TRAINCOUNT]
    concatenatedContent = []
    for i, cont in enumerate(content):
        concatenatedContent.append(' '.join(map(str, cont)))
    vectorizer = CountVectorizer(concatenatedContent)
    Generated = vectorizer.fit_transform(concatenatedContent).toarray()
    
    ids = [item[0] for item in AllDocumentsTokenized]
    
    clf = GaussianNB()
    clf.fit(Generated, LabelTrain)
    
    result = []
    if (isTesting == False):
        result = cross_val_score(clf, Generated, LabelTrain, cv=10)
        
        concatenated = np.column_stack((ids, clf.predict(Generated)))
    else:
        result = Accuracy(clf.predict(Generated), LabelTrain)
        concatenated = np.column_stack((ids, clf.predict(Generated)))
    
    return result, concatenated

AccuracyNaiveBayes, NaiveBayesTrain = NaiveBayes()

#WordImportant=deleteStopWords(corp)
#DataCount=TF(DataStemming)
#DataCorpus=CORPUS(DataStemming)
#DataCount=countForTF(DataStemming)
#FindTF=getTF(DataCount, DataStemming)

#IdfHeadline=[]
#for outer,valInside in enumerate(DataStemming[0]):
#    tempIDF=[]
#    for inside,valHL in enumerate(valInside):
#        print(valInside.count(valHL))
#        tempIDF.append(np.log(len(DataStemming[0]) / (valInside.count(valHL))))
#    IdfHeadline.append(tempIDF)
# =============================================================================
# Preproses Data Testing
# =============================================================================
#getDataTesting=getData(PATHTESTING)
#tokenizeTesting=tokenization(getDataTesting)
#DataTestStemming=stemmingData(tokenizeTesting)
#countDataTest=countForTF(DataTestStemming)
#GetTFTesting=getTF(countDataTest,DataTestStemming)