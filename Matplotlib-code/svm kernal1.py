import matplotlib.pyplot as plt
import numpy as np

# Generate some random data for the plot
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.random.randint(2, size=100)

# Plot the data points, coloring them by their class label
plt.scatter(X[:, 0], X[:, 1], c=y)

# Add a legend to the plot
plt.legend(['Class 0', 'Class 1'])

# Add the hyperplane to the plot
w = np.array([1, 1])
b = 0
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx = np.linspace(x_min, x_max)
yy = -(w[0] * xx + b) / w[1]
plt.plot(xx, yy, 'k-')

# Add the support vectors to the plot
support_vectors = X[np.abs(1 - y * (np.dot(X, w) + b)) < 1e-5]
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='r', marker='x')

# Add labels and a title to the plot
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Support Vector Machine')

# Show the plot
plt.show()

