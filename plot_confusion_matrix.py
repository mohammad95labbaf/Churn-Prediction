# plot_confusion_matrix.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, classifier_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {classifier_name} Classifier')
    plt.show()

def plot_confusion_matrices(classifier_name, accuracy, cm, cr):
    plot_confusion_matrix(cm, classifier_name)
    print(f'Accuracy: {accuracy:.3f}')
    print(cr)
