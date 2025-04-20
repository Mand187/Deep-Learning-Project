import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchsummary import summary


def plotLoss(trainLosses, valLosses):
    plt.figure(figsize=(10, 5))
    plt.plot(trainLosses, label='Training Loss')
    plt.plot(valLosses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotAccuracy(trainAccuracies, valAccuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(trainAccuracies, label='Training Accuracy')
    plt.plot(valAccuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def confusionMatrix(trueLabels, predictedLabels, classNames=None):
    if classNames is None:
        classNames = np.unique(trueLabels)

    cm = confusion_matrix(trueLabels, predictedLabels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classNames, yticklabels=classNames)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def reportFinalMetrics(trainAccuracies, valAccuracies, epochTimes):
    finalTrainAccuracy = trainAccuracies[-1]
    finalValAccuracy = valAccuracies[-1]

    totalTrainingTime = sum(epochTimes)

    print(f"Final Training Accuracy: {finalTrainAccuracy:.2f}%")
    print(f"Final Validation Accuracy: {finalValAccuracy:.2f}%")
    print(f"Total Training Time: {totalTrainingTime:.2f} seconds")

def reportMultiFinalMetrics(trainAccuracies, valAccuracies, epochTimes, modelNames):
    for i, model in enumerate(modelNames):
        finalTrainAccuracy = trainAccuracies[i][-1]
        finalValAccuracy = valAccuracies[i][-1]
        totalTrainingTime = sum(epochTimes[i])

        print(f"{model} Final Training Accuracy: {finalTrainAccuracy:.2f}%")
        print(f"{model} Final Validation Accuracy: {finalValAccuracy:.2f}%")
        print(f"{model} Total Training Time: {totalTrainingTime:.2f} seconds")
        print()

def plotMultiAccuracy(modelNames, trainAccuracies, valAccuracies):
    plt.figure(figsize=(10, 5))
    for i, model in enumerate(modelNames):
        plt.plot(trainAccuracies[i], label=f'{model} Training Accuracy')
        plt.plot(valAccuracies[i], label=f'{model} Validation Accuracy')

    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotMultiLoss(modelNames, trainLosses, valLosses):
    plt.figure(figsize=(10, 5))
    for i, model in enumerate(modelNames):
        plt.plot(trainLosses[i], label=f'{model} Training Loss')
        plt.plot(valLosses[i], label=f'{model} Validation Loss')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def computationalComplexity(model, input_size):
    # Assuming the model is a PyTorch model
    model = summary(model, input_size=input_size)
    print(model)
