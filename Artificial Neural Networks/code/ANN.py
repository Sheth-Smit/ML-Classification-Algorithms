import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Activations import Activations
from Preprocessor import Preprocessor

class ANN:
    
    def __init__(self, X, y):
        
        self.input = X
        self.y = y
        self.outputs = np.zeros(y.shape)
        self.weights = []
        self.biases = []
        self.units = []
        self.activations = []
        self.layer_output = {}
        self.layer_unactivated = {}
        self.layer_count = 1
        
        self.units.append(self.input.shape[1])
        self.weights.append(np.random.random((self.input.shape[1], 1)))
        self.biases.append(np.random.random())
        self.activations.append('relu')
            
    def add_layer(self, units = 4, activation='sigmoid', initialization='normal'):
        
        if initialization == 'normal':
            weights1 = np.random.randn(self.units[self.layer_count-1], units)
            bias1 = np.random.randn(units)
            weights2 = np.random.randn(units, 1)
            bias2 = np.random.randn()
        else:
            weights1 = np.random.random((self.units[self.layer_count-1], units))
            bias1 = np.random.random(units)
            weights2 = np.random.random((units, 1))
            bias2 = np.random.random()
            
        self.weights[self.layer_count-1] = weights1
        self.biases[self.layer_count-1] = bias1
        self.weights.append(weights2)
        self.biases.append(bias2)
        self.activations.append(activation)
        self.units.append(units)
        self.layer_count += 1
        
    def feed_forward(self, X):
        
        a = z = X
        
        for i in range(self.layer_count):
            self.layer_unactivated[i] = a
            self.layer_output[i] = z
            a = np.dot(self.weights[i].T, z)
            a += self.biases[i]
            z = Activations.activation(a, act_type=self.activations[i])
        
        self.layer_unactivated[self.layer_count] = a
        self.layer_output[self.layer_count] = z
        
        return z
    
    def back_propogation(self, y_act, y_pred):
        deriv_error = {}
        deriv_bias = {}
        
        
        deriv_loss = y_pred-y_act
        for itr in range(self.layer_count-1, -1, -1):
            if itr == self.layer_count-1:
                term1 = np.array(self.layer_output[itr]).reshape([self.layer_output[itr].shape[0], 1])
                term2 = deriv_loss.reshape([1, 1])
                term3 = np.array(Activations.deriv_activation(self.layer_unactivated[itr+1],
                                                              act_type=self.activations[itr])).reshape([
                                                                  self.layer_output[itr+1].shape[0], 1])
                                                                  
                term4 = np.dot(term2, term3)
                                                                  
                deriv_error[itr] = np.dot(term1, term4.T)
                deriv_bias[itr] = term4
            else:
                term1 = np.array(self.layer_output[itr]).reshape([self.layer_output[itr].shape[0], 1])
                term2 = np.dot(deriv_error[itr+1], self.weights[itr+1].T)
                term3 = np.array(Activations.deriv_activation(self.layer_unactivated[itr+1],
                                                              act_type=self.activations[itr]))
                                            
                if self.activations[itr] == 'linear':
                    term3 = 1
                else:
                    term3 = term3.reshape([term3.shape[0], 1])
                
                term4 = np.dot(term2, term3)
                                                                  
                deriv_error[itr] = np.dot(term1, term4.T)
                deriv_bias[itr] = term4

            
        return deriv_error, deriv_bias
        
    def train(self, X_train, y_train, epoch=10, batch_size=300, learning_rate=0.05):
        
        # for wt in self.weights:
        #     print(wt)
        
        loss=0
        correct=0
        total=0
        accuracy=0
        plot_loss = []
        plot_itr = []
        
        for itr in range(epoch):
            
            for batch_itr in range(batch_size):
                index = np.random.randint(0, len(X_train))

                y_pred = self.feed_forward(X_train[index])
                
                deriv_error, deriv_bias = self.back_propogation(y_train[index], y_pred)
                
                for wt_index in range(len(self.weights)):
                    self.weights[wt_index] -= learning_rate*deriv_error[wt_index]
                    self.biases[wt_index] -= learning_rate*deriv_bias[wt_index][0]
                
                if y_pred >= 0.5:
                    y_pred = 1
                else:
                    y_pred = 0
                
                if y_pred == y_train[index]:
                    correct += 1
                total += 1
            
            accuracy = correct/total
            
            if(itr+1) % 100 == 0:
                loss = 0
                for j in range(len(X_train)):
                    loss_y_pred = self.feed_forward(X_train[j])
                    loss += self.binary_crossentropy(y_train[j], loss_y_pred)
                
                plot_loss.append(loss)
                plot_itr.append(itr)
                
                print("Epoch: ", itr+1, "Loss: ", loss)
                print("Training Accuracy: ", accuracy)
    
        # Loss plot
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(plot_itr, plot_loss)
                
    def test(self, X_test):
        y_pred = []
        
        for X in X_test:
            pred = self.feed_forward(X)
            
            if pred >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
                
        return y_pred
    
    def evaluate(self, y_test, y_pred):
        
        tp = tn = fp = fn = 0
        
        for index in range(len(y_test)):
            
            if y_test[index] == 1 and y_pred[index] == 1:
                tp += 1
            elif y_test[index] == 0 and y_pred[index] == 1:
                fp += 1
            elif y_test[index] == 1 and y_pred[index] == 0:
                fn += 1
            else:
                tn += 1
                
        accuracy = (tp+tn) / (tp+tn+fp+fn)    
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f_score = (2*precision*recall) / (precision+recall)
        
        print("Test Accuracy : ", accuracy)
        print("Test F-score: ", f_score)
        
    def binary_crossentropy(self, y_act, y_pred):
        
        loss=0
        
        if y_act == 1:
            loss = -np.log(y_pred)
        else:
            loss = -np.log(1-y_pred)
            
        return loss