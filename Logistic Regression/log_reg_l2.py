import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


def loadCSV(filename):
	'''
	function to load dataset
	'''
	dataset=pd.read_csv(filename)
	return np.array(dataset)


def normalize(X):
	'''
	function to normalize feature matrix, X
	'''
	mins = np.min(X, axis = 0)
	maxs = np.max(X, axis = 0)
	rng = maxs - mins
	norm_X = 1 - ((maxs - X)/rng)
	return norm_X

def standardize(X):
    '''
    function to standardize feature matrix X
    '''
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std_X = (X - mean) / std
    return std_X

def logistic_func(beta, X):
	'''
	logistic(sigmoid) function
	'''
	return 1.0/(1 + np.exp(-np.dot(X, beta.T)))


def log_gradient(beta, X, y,reg_cof):
	'''
	logistic gradient function
	'''
	first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1)
	final_calc = np.dot(first_calc.T, X)
	final_calc += 2*reg_cof*beta
	return final_calc


def cost_func(beta, X, y,reg_cof):
	'''
	cost function, J
	'''
	log_func_v = logistic_func(beta, X)
	y = np.squeeze(y)
	s1 = y*np.log(log_func_v)
	s2 = (1-y)*np.log(1-log_func_v)
	s3 = reg_cof*np.dot(beta,beta.T)
	ans = -s1-s2+s3
	return np.mean(ans)


def grad_desc(X, y, beta, lr=.0005, converge_change=.0001,reg_cof=.1):
	'''
	gradient descent function
	'''
	cost = cost_func(beta, X, y,reg_cof)
	change_cost = 1
	num_iter = 1

	while(change_cost > converge_change):
		old_cost = cost
		beta = beta-(lr*log_gradient(beta, X, y,reg_cof))
		cost = cost_func(beta, X, y,reg_cof)
		change_cost = old_cost-cost
		if num_iter % 100 == 0:
			print("Iterations: ", num_iter, " Cost: ", cost)
		num_iter+=1

	return beta, num_iter


def pred_values(beta, X):
	'''
	function to predict labels
	'''
	pred_prob = logistic_func(beta, X)
	pred_value = np.where(pred_prob >= .5, 1, 0)
	return np.squeeze(pred_value)


def calc_fscore(y_pred,ytest):
	tp=0
	tn=0
	fp=0
	fn=0
	for i in range(0,len(y_pred)):
		if(y_pred[i]==0 and ytest[i]==0):
			tn+=1
		if(y_pred[i]==0 and ytest[i]==1):
			fn+=1
		if(y_pred[i]==1 and ytest[i]==0):
			fp+=1
		if(y_pred[i]==1 and ytest[i]==1):
			tp+=1
	prec=tp/(tp+fp)
	rec=tp/(tp+fn)
	return 2*prec*rec/(prec+rec)


def calc_acc(y_pred,ytest):
	tp=0
	tn=0
	fp=0
	fn=0
	for i in range(0,len(y_pred)):
		if(y_pred[i]==0 and ytest[i]==0):
			tn+=1
		if(y_pred[i]==0 and ytest[i]==1):
			fn+=1
		if(y_pred[i]==1 and ytest[i]==0):
			fp+=1
		if(y_pred[i]==1 and ytest[i]==1):
			tp+=1
	return (tp+tn)/(tp+tn+fp+fn)

def plot_reg(X, y, beta):
	'''
	function to plot decision boundary
	'''
	# labelled observations
	x_0 = X[np.where(y == 0.0)]
	x_1 = X[np.where(y == 1.0)]

	# plotting points with diff color for diff label
	plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
	plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')

	# plotting decision boundary
	x1 = np.arange(0, 1, 0.1)
	x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2]
	plt.plot(x1, x2, c='k', label='reg line')

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend()
	plt.show()



if __name__ == "__main__":
	# load the dataset
	dataset = loadCSV('dataset1.csv')
	print(dataset)


	# Standardizing feature matrix
	X = standardize(dataset[:, :-1])

	# stacking columns wth all ones in feature matrix
	X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
	print(dataset)
	# response vector
	y = dataset[:, -1]
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 5)

	# initial beta values
	beta = np.matrix(np.zeros(xtrain.shape[1]))

	# beta values after running gradient descent
	beta, num_iter = grad_desc(xtrain, ytrain, beta)

	# estimated beta values and number of iterations
	print("Estimated regression coefficients:", beta)
	print("No. of iterations:", num_iter)
	# predicted labels
	y_pred = pred_values(beta, xtest)
	f=calc_fscore(y_pred,ytest)
	print("F-Score: ",f)
	acc=calc_acc(y_pred,ytest)
	print("Accuracy: ",acc)
	# number of correctly predicted labels
	print("Correctly predicted labels:", np.sum(ytest == y_pred))

	# plotting regression line
	plot_reg(xtest, ytest, beta)
