# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:14:30 2018

@author: Rodrigo Azevedo
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

# Thermometer encoder for real vectors
# [0.1, 1] to (for n=10)
# [1 1 0 0 0 0 0 0 0 0
#  1 1 1 1 1 1 1 1 1 1]
def thermometer(real_vec, min_value=-1.0, max_value=1.0, n=10):
    vec = []
    for v in real_vec:
        if v == max_value:
            rang = [1] * n
            vec.extend(rang)
        else:
            t = (max_value - min_value)
            p = v - min_value
            s = int((p / t) * n)
            rang = [1] * (s + 1)
            rang.extend([0] * (n - s - 1))
            vec.extend(rang)
    return np.array(vec)

# One-Hot encoder for real vectors
# [0.1, 1] to (for n=10)
# [0 1 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1]
def one_hot(real_vec, min_value=-1.0, max_value=1.0, n=10):
    vec = []
    for v in real_vec:
        if v == max_value:
            rang = [0] * n
            rang[-1] = 1
            vec.extend(rang)
        else:
            rang = [0] * n
            t = (max_value - min_value)
            p = v - min_value
            s = int((p / t) * n)
            rang[s] = 1
            vec.extend(rang)
    return np.array(vec)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')

def get_confusion_matrix(y_pred, y_test, labels=[]):
    return confusion_matrix(y_test, y_pred, labels=labels)
    
def show_confusion_matrix(cm, labels=[], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Making the Confusion Matrix
    #cm = confusion_matrix(y_test, y_pred, labels=labels)
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=labels, title=title, cmap=cmap)
    plt.show()

def show_retina(X, rows=50, columns=10): 
    pixels = np.array(X, dtype='uint8')
    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((columns, rows))
    # Plot
    plt.imshow(pixels, cmap='gray', interpolation='nearest')
    plt.show()