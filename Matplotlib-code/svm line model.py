# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# Get started with interactive Python!
# Supports Python Modules: builtins, math,pandas, scipy 
# matplotlib.pyplot, numpy, operator, processing, pygal, random, 
# re, string, time, turtle, urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp


# Generate some random data for the plot
np.random.seed(0)
X = np.random.randn(200, 2)
y = np.array([1 if x[0] > x[1] else -1 for x in X])

# Train the SVM model using the scikit-learn library
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Get the weights and bias of the model
w = model.coef_[0]
b = model.intercept_[0]

# Create a meshgrid of values for the x and y axis
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Calculate the decision boundary by evaluating the model on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the data points and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.contour(xx, yy, Z.reshape(xx.shape), colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM Decision Boundary')
plt.show()


