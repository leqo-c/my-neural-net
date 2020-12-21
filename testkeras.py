import pandas as pd
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# Read the .csv file into a Pandas dataframe, then normalize all of its
# columns with a MinMaxScaler. This will help the network perform better
# on the regression task.
df = pd.read_csv('housing.csv', header=None)
scaler = MinMaxScaler()
fitted_scaler = scaler.fit(df)
df = fitted_scaler.transform(df)

# Extract the feature matrix X and the target vector Y from the
# data frame and convert them into numpy arrays.
X = np.array(df[:, :-1])
Y = np.array(df[:, -1])

# Perform a static, 90%/10% train and test split.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1)

# Build the Keras model through using the Sequential interface.
model = Sequential([
    Input((X.shape[1])),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with an optimizer and a loss function.
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, validation_split=.1, verbose=2)

preds = model.predict(X_test)

# Get an idea of the network's performances by comparing the predicted
# target value and the actual value on the first 10 examples in the test
# set (Note: the values are scaled between 0 and 1). 
for i in range(10):
    original_val = y_test[i]
    predicted_val = preds[i, :]
    print(original_val, predicted_val)
