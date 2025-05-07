# %% [markdown]
# # Testing Notebook

# %% [markdown]
# # Ignore Trainig and Validation Accuracy, those are not objective truth measruements they are more just a guage to see how far the model is from the actual postion of the car
# # Also ignore those metrics for the funciton reportFinalMetrics


import config as cfg
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from Data.data_loading import load_and_preprocess_data, create_tensor_from_dataframe, create_sequences, create_dataloaders 
from Training.train_matt import Trainer
from Training.basicEval import plotLoss, plotAccuracy, reportFinalMetrics, reportMultiFinalMetrics, plotMultiAccuracy, plotMultiLoss, test_model
from NextNet.model_split import FrameTransformer, print_model_info

from Training.customLoss import ADELoss, FDELoss, RMSELoss, PaddedMSELoss

# %%
root_dir = os.getcwd()  # Use current working directory as root
data_dir = os.path.join(root_dir, 'Data')
csv_dir = os.path.join(data_dir, 'csv')
csv_file = os.path.join(csv_dir, 'trimmed_IMG_4097_detections.csv')

print("Data directory: ", data_dir)
print("CSV directory: ", csv_dir)
print("CSV file: ", csv_file)


model_dir = os.path.join(root_dir, 'Model')
save_model_dir = os.path.join(model_dir, 'Saved_Model')
print("Model directory: ", model_dir)
print("Saved model directory: ", save_model_dir)
model_name = 'fde_model_5s.pth'


# %%
ADE = ADELoss()
FDE = FDELoss()
RMSE = RMSELoss()
PADDEDMSE = PaddedMSELoss()


criterion = FDE

# %%
df, transformer_max_ids_per_frame, frame_scaler, feature_scaler = load_and_preprocess_data(csv_folder=csv_dir)

# 2. Create tensor from dataframe
all_data_tensor = create_tensor_from_dataframe(df, transformer_max_ids_per_frame)

# 3. Create input-output sequences
X, Y = create_sequences(all_data_tensor)

# 4. Create dataloaders for training and testing
train_loader, test_loader, train_prefetcher, test_prefetcher = create_dataloaders(X, Y)

# %% [markdown]
# # MSE

# %%
# Adjust the hidden_size and sequence_length to match the input tensor dimensions



# %%

model_file_path = os.path.join(save_model_dir, model_name)  # Add a file name
retrain_model = True
if retrain_model:
    model = FrameTransformer(
        input_feature_size=cfg.NUM_INPUT_FEATURES, 
        num_ids=transformer_max_ids_per_frame, 
        sequence_length=X.size(1),  
        prediction_length=cfg.PREDICTION_LENGTH,
        hidden_size=cfg.HIDDEN_SIZE,  
        num_heads=cfg.NUM_HEADS,
        dropout_rate=cfg.DROPOUT_RATE
    )
    trainScript = Trainer(model, train_prefetcher, test_prefetcher, model_name=model_name[:-4], model_path=model_file_path, device=cfg.DEVICE)  # Load the model from the specified path

    trainScript.earlyStop(enable=True, patience=30, delta=0.01)
    train_losses1, val_losses1, train_accs1, val_accs1, epoch_times1 = trainScript.train(
        num_epochs=cfg.EPOCHS, 
        learningRate=cfg.LEARNING_RATE, 
        criterion=criterion, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    )
else:
    model = torch.load(model_file_path)
    trainScript = Trainer(model, train_prefetcher, test_prefetcher, model_path=os.path.join(save_model_dir, model_name))

    trainScript.earlyStop(enable=True, patience=30, delta=0.01)
    train_losses1, val_losses1, train_accs1, val_accs1, epoch_times1 = trainScript.train(
        num_epochs=int(cfg.EPOCHS*.3), 
        learningRate=cfg.LEARNING_RATE*.1, 
        criterion=PADDEDMSE, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE),
        model_path = model_file_path
    )

test_model(
    model,
    test_loader,
    criterion,
    feature_scaler,
    device=cfg.DEVICE
)

Trainer