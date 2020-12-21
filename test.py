import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from nn import MyNN
from metrics import accuracy_sklearn
from learningrates import staircase_lr
from loss_functions import MeanSquaredError
from activations import Sigmoid, Linear, Tanh, ReLU

# Set a fixed seed for reproducibility.
np.random.seed(3)

# Read the .csv file into a Pandas dataframe, then normalize all of its
# columns with a MinMaxScaler. This will help the network perform better
# on the regression task.
df = pd.read_csv('housing.csv', header=None)
scaler = MinMaxScaler()
fitted_scaler = scaler.fit(df)
df = fitted_scaler.transform(df)

# Transpose the dimensions of the feature matrix X and the target values
# vector Y for compatibility reasons (the implementation of the MyNN
# module requires that the vectors' dimensions be this way).
X = np.array(df[:, :-1]).T
Y = np.array(df[:, -1]).reshape(1, X.shape[1])

# Perform a static, 90%/10% train and test split.
X_train, X_test, y_train, y_test = train_test_split(X.T, Y.T, test_size=.1)

# Since this is a regression example, the metric we are interested in is
# the mean squared error.
metric = MeanSquaredError().compute

# Set up a staircase-like learning rate schedule, specifying different
# values for different ranges of the epoch number. 
lr_values = [(5000, 0.1), (8000, 0.0001), (np.inf, 0.00001)]
# Freeze the first parameter of the `staircase_lr` function so as to
# obtain a function that only takes the epoch number as its input.
staircase_lr_func = partial(staircase_lr, lr_values)

# Get an instance of the MyNN class with the proper parameters and then
# start the training process by calling the fit method.
nnet = MyNN([X.shape[0], 16, 1], output_activation=Sigmoid(),
            hidden_activation=ReLU(), cost_function=MeanSquaredError())
nnet.fit(X_train.T, y_train.T, learning_rate=staircase_lr_func,
         epochs=5000, metric_function=metric)

# Carry out the predictions on the test set so as to evaluate the
# network's performances.
preds = nnet.predict(X_test.T)

# Print the value of the loss function on the test set.
print(f'Test loss: {metric(y_test.T, preds)}')

# Get an idea of the network's performances by comparing the predicted
# target value and the actual value on the first 10 examples in the test
# set (Note: the values are scaled between 0 and 1). 
for i in range(10):
    original_val = y_test.T[:, i]
    predicted_val = preds[:, i]
    
    print(original_val, predicted_val)
