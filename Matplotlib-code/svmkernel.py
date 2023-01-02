
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

from sklearn import svm

# Input data
X = np.array([[0, 0], [1, 1]])  # Training examples
y = np.array([0, 1])  # Class labels

# Define SVM model
model = svm.SVC(kernel='linear')

# Train model on training data
model.fit(X, y)

# Make predictions on new data
x_new = np.array([[2, 2]])
y_pred = model.predict(x_new)

print(f'Prediction for new input vector: {y_pred[0]}')

#This code defines a linear SVM model using the SVC class from the sklearn library, trains the model on the training data (X and y), and then makes a prediction on a new input vector x_new. The prediction is made by evaluating the prediction function using the trained weight vector and bias term, as shown in the equation:

#$$\hat{y} = f(\vec{x}) = \text{sign}(\vec{w} \cdot \vec{x} + b)$$

#You can customize the model by specifying different kernel functions (e.g. polynomial, RBF) and adjusting the value of the hyperparameter C to control the trade-off between maximizing the margin and minimizing the number of misclassified training examples.


