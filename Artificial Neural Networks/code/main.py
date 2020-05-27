import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessor import Preprocessor
from ANN import ANN

dataset = pd.read_csv('../data/housepricedata.csv')
dataset = Preprocessor.normalize(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=41)

ann = ANN(X_train, y_train)
ann.add_layer(units=5, initialization='uniform', activation='sigmoid')
# ann.add_layer(units=5, initialization='uniform', activation='sigmoid')

ann.train(X_train, y_train, epoch=10000, batch_size=20,  learning_rate=0.01)
y_pred = ann.test(X_test)
ann.evaluate(y_test, y_pred)


