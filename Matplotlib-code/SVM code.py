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


def precision(true_positive, false_positive):
  return true_positive / (true_positive + false_positive)

def recall(true_positive, false_negative):
  return true_positive / (true_positive + false_negative)

def f1_score(precision, recall):
  return 2 * (precision * recall) / (precision + recall)

# example values for true positive, false positive, and false negative
true_positive = 50
false_positive = 10
false_negative = 5

precision_val = precision(true_positive, false_positive)
recall_val = recall(true_positive, false_negative)
f1_score_val = f1_score(precision_val, recall_val)

# create the figure
fig, ax = plt.subplots()

# plot the precision, recall, and F1 score values on the y-axis and the x-axis
ax.plot([precision_val, recall_val, f1_score_val], '-', color='Black')

# add labels to the figure
ax.set(xlabel='Metrics', ylabel='Values', title='Performance Metrics for SVM Model')

# add tick marks and labels for the x-axis
plt.xticks([0, 1, 2], ['Precision', 'Recall', 'F1 Score'])

# add a grid to the figure
plt.grid(True)

# add a legend to the figure
plt.legend(['SVM Model'])

# show the figure
plt.show()


