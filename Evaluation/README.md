# Evaluation Module Usage

This document provides examples of how to use the evaluation utilities in this project.

## Import

```python
from Evaluation.basicEval import *
```

## Example Usage for Multiple Models

```python
names = []

# Generate a report for multiple models
reportMultiFinalMetrics(trainAccuraciesMulti, validationAccuraciesMulti, epochTimesMulti, names)

# Plot loss and accuracy for multiple models
plotMultiLoss(trainingLosesMulti, validationLossesMulti, names)
plotMultAccuracy(trainAccuraciesMulti, validationAccuraciesMulti, names)
```

## Example Usage for a Single Model

```python
# Generate a report for a single model
reportFinalMetrics(trainAccuracies, valAccuracies, epochTimes)

# Generate a confusion matrix
confusionMatrix(trueLabels, predictedLabels, classNames=None)

# Plot accuracy and loss for a single model
plotAccuracy(trainAccuracies, valAccuracies)
plotLoss(trainLosses, valLosses)
```

## Model Summary

```python
# Calculate the computational complexity of a model
computationalComplexity(model, input_size)
```