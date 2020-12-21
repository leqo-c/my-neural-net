# my-neural-net
This repository contains a basic implementation of a Neural Network using just Python's basic constructs and the numpy library. Its functionalities include both the creation of a regression and a binary classification model, though more solutions can be implemented by definining the proper activation functions, loss functions and their derivatives (e.g. Softmax for multi-label classification).

To ensure the implementation's correctness, a comparison is made between my implementation and Keras' regarding a simple regression problem on the Boston Housing Dataset (https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).

## Relevant files
Here is a list of the most relevant files and their content:

 - **`nn.py`**: Actual implementation of the Neural Network (forward pass, backward pass, training and prediction functions);
 - **`activations.py`**: Contains the implementation (function and derivative) of all the available activation functions, used in the hidden layers' neurons;
 - **`loss_functions.py`**: Contains the implementation (function and derivative) of all the available loss functions, used in the training phase;
 - **`test.py`**: Run this file to test the Neural Network implementation on the Boston Housing Dataset;
 - **`testkeras.py`**: Run this file to test Keras' implementation of a Neural Network with the same architecture on the Boston Housing Dataset.
