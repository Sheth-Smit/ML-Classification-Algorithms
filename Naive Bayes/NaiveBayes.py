# -*- coding: utf-8 -*-

from collections import defaultdict

class NaiveBayes:
    
    def __init__(self):
        
        self.negative_words = defaultdict(int)
        self.positive_words = defaultdict(int)
        prob0 = 0.0
        prob1 = 0.0
        
    
    def fit(self, X_train, y_train):
        
        count0 = 0
        count1 = 0
        
        for i in range(len(X_train)):
            words = X_train[i].split(' ')
            
            if y_train[i] == 0:
                count0 += 1
                for word in words:
                    self.negative_words[word] += 1
            else:
                count1 += 1
                for word in words:
                    self.positive_words[word] += 1
        
        self.prob0 = count0 / (count0 + count1)
        self.prob1 = count1 / (count0 + count1)
        
    def predict(self, X_test):
        
        y_pred = []
        
        for text in X_test:
            prob_class0 = self.prob0
            prob_class1 = self.prob1
            # print(prob_class0, prob_class1)
            
            words = text.split(' ')
            
            for word in words:
                p0 = self.negative_words[word]
                p1 = self.positive_words[word]
                
                
                prob_class0 *= (p0+1)/(p0+p1+2)
                
                prob_class1 *= (p1+1)/(p0+p1+2)
            
            if prob_class0 > prob_class1:
                y_pred.append(0)
            elif prob_class0 < prob_class1:
                y_pred.append(1)
            else:
                if self.prob0 > self.prob1:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
        
        return y_pred
    
