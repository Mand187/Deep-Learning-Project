import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import torch


def plotLoss(trainLosses, valLosses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(trainLosses, label='Training Loss')
    plt.plot(valLosses, label='Validation Loss')
    # plt.ylim(0, 5)
    # plt.xlim(0, len(trainLosses))
    plt.title(title)
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

def reportFinalMetrics(trainAccuracies, valAccuracies, trainLosses, valLosses, epochTimes):
    finalTrainAccuracy = trainAccuracies[-1]
    finalValAccuracy = valAccuracies[-1]
    trainLosses = trainLosses[-1]
    valLosses = valLosses[-1]
    totalTrainingTime = sum(epochTimes)

    print(f"Final Training Accuracy: {finalTrainAccuracy:.2f}%")
    print(f"Final Validation Accuracy: {finalValAccuracy:.2f}%")
    print(f"Final Training Loss: {trainLosses:.4f}")
    print(f"Final Validation Loss: {valLosses:.4f}")
    print(f"Total Training Time: {totalTrainingTime:.2f} seconds")

def reportMultiFinalMetrics(trainAccuracies, valAccuracies, trainLosses, valLosses, epochTimes, modelNames):
    for i, model in enumerate(modelNames):
        finalTrainAccuracy = trainAccuracies[i][-1]
        finalValAccuracy = valAccuracies[i][-1]
        trainLosses = trainLosses[i][-1]
        valLosses = valLosses[i][-1]
        totalTrainingTime = sum(epochTimes[i])

        print(f"{model} Final Training Accuracy: {finalTrainAccuracy:.2f}%")
        print(f"{model} Final Validation Accuracy: {finalValAccuracy:.2f}%")
        print(f"{model} Final Training Loss: {trainLosses:.4f}")
        print(f"{model} Final Validation Loss: {valLosses:.4f}")
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

def test_model(model, test_loader, loss_fn, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute batch loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # Weight by batch size

            # Save predictions and actuals
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    predicted = torch.cat(all_preds, dim=0).numpy()
    actual = torch.cat(all_targets, dim=0).numpy()

    # Metrics
    test_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss (MSE): {test_loss:.4f}')

    mask = (actual != -1).astype(float)  # Assuming -1 is the padding value
    
    difference = (predicted - actual) * mask
# ADE
    # displacement_errors_per_point has shape (N, S)
    # It's 0 for time-steps that were fully padded in 'actual'.
    displacement_errors_per_point = np.linalg.norm(difference, axis=-1)
    
    # Create a mask for valid time-steps (points)
    # Assumes a point is padded if its first coordinate is -1
    # valid_points_mask has shape (N, S)
    valid_points_mask = (actual[..., 0] != -1) 
    num_valid_points = np.sum(valid_points_mask)
    
    sum_displacement_errors = np.sum(displacement_errors_per_point)
    
    ade = sum_displacement_errors / num_valid_points if num_valid_points > 0 else 0.0

    # RMSE
    # difference ** 2 will have 0s for padded coordinates
    sum_squared_errors = np.sum(difference ** 2)
    
    # num_valid_coordinates is the total count of non-padded individual coordinate values
    num_valid_coordinates = np.sum(mask) 
    
    mse_val = sum_squared_errors / num_valid_coordinates if num_valid_coordinates > 0 else 0.0
    rmse = np.sqrt(mse_val)

    print(f'Average Displacement Error (ADE): {ade:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    
    predicted_sample = predicted[0, :, 0]
    actual_sample = actual[0, :, 0]
    

    comparison = np.stack((predicted_sample, actual_sample, predicted_sample-actual_sample), axis=1)
    
    print(f"Target shape: {actual.shape}")
    print(f"Predicted shape: {predicted.shape}")
    # Display some predictions
    print("\nPredicted vs Actual (First 5 examples):")
    print(comparison)
    