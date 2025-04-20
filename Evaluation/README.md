# Import

from Evaluation.basicEval import *

# Example Eval Use:

names = []

reportMultiFinalMetrics(trainAccuraciesMulti, validationAccuraciesMulti, epochTimesMulti, names)

plotMultiLoss(trainingLosesMulti, validationLossesMulti, names)

plotMultAccuracy(trainAccuraciesMulti, validationAccuraciesMulti, names)


# If you dont have multiple models

reportFinalMetrics(trainAccuracies, valAccuracies, epochTimes)

confusionMatrix(trueLabels, predictedLabels, classNames=None)

plotAccuracy(trainAccuracies, valAccuracies)

plotLoss(trainLosses, valLosses)

# Model Summary

computationalComplexity(model, input_size)