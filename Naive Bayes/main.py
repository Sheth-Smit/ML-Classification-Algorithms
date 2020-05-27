# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from preprocess import data_preprocess
from NaiveBayes import NaiveBayes

raw_data = open('a1_d3.txt', 'r')
dataset = data_preprocess(raw_data)

split_dataset = np.array_split(dataset, 5)

accuracies = []
f_scores = []

for epoch in range(5):
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for i in range(5):
        if epoch == i:
            X_test = split_dataset[epoch]['Review'].values
            y_test = split_dataset[epoch]['Sentiment'].values
        else:
            X_train.append(pd.DataFrame(split_dataset[i]['Review'].values))
            y_train.append(pd.DataFrame(split_dataset[i]['Sentiment'].values))
    
    X_train = pd.concat(X_train)[0].values
    y_train = pd.concat(y_train)[0].values
    
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    tp = tn = fp = fn = 0
    
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_test[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y_test[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_test[i] == 0 and y_pred[i] == 0:
            tn += 1
    
    accuracy = (tp+tn) / (tp+tn+fp+fn)    
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f_score = (2*precision*recall) / (precision+recall)
    
    accuracies.append(accuracy)
    f_scores.append(f_score)
    
    print("Accuracy after fold ", epoch+1, ": ", accuracy)
    # print("F-score at epoch ", epoch, ": ", f_score)
    
    
print("Accuracy after 5-fold validation: ", round(np.mean(accuracies), 3), " ± ", round(np.std(accuracies), 3))
print("F-score after 5-fold validation: ", round(np.mean(f_scores), 3), " ± ", round(np.std(f_scores), 3))


