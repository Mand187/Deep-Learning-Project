import config as cfg
import csv
import numpy as np
import traceback
import os
import multiprocessing as mp
import torch
from torchtnt.utils.data import CudaDataPrefetcher
from Training.jutils import ColorPrinter, Colors, assert_dir, assert_file
from Data.data_loading_jaskin import load_and_preprocess_data, create_tensor_from_dataframe, create_sequences, create_dataloaders 
from Training.train_matt import Trainer
from NextNet.model_split import FrameTransformer
from Image_Processing.visualize import generate_video_from_predictions

from Training.customLoss import ADELoss, FDELoss, RMSELoss, PaddedMSELoss
printer = ColorPrinter()

def generate_predictions_csv(
    model_dir,
    model_name,
    csv_folder,
    video_file,
    output_csv_name,
    device=None,
):
    model_file = os.path.join(model_dir, model_name, f'{model_name}_epoch_50.pth')
    assert_file(model_file)
    assert_dir(csv_folder)
    assert_file(video_file)
    
    output_csv_path = os.path.join(model_dir, output_csv_name)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = torch.load(model_file, map_location=device, weights_only=False)
    model.eval()
    headers = ['Frame', 'ID', 'X_pred', 'Y_pred', 'X_true', 'Y_true']
    print(f"Exporting predictions to {output_csv_path}...")
    
    df, num_ids = load_and_preprocess_data(csv_folder)
    all_tensors, _ = create_tensor_from_dataframe(df, 20)
    X, Y = create_sequences(all_tensors, prediction_length=30)
    
    # append zeros to ensure X has shape 100, 20, 4
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    x_padding = torch.full((X.shape[0], X.shape[1], 20-X.shape[2], X.shape[3]), -1)
    y_padding = torch.full((Y.shape[0], Y.shape[1], 20-Y.shape[2], Y.shape[3]), -1)
    X = torch.cat((X, x_padding), dim=2)
    Y = torch.cat((Y, y_padding), dim=2)
    print(f"X shape after padding: {X.shape}")
    print(f"Y shape after padding: {Y.shape}")
    
    
    
    
    
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        
        with torch.no_grad():
            # X shape: (100, 18, 5)
            # Y shape: (30, 18, 2)
            
            # X contains (Seq_idx, ID, Features) 
            #    Features = [Frame, X, Y, Width, Height]
            # Y contains [Seq_idx, ID, X, Y]
            
            
            x = X[0]
            x_unsqueezed = x.unsqueeze(0).to(device)
            x = x.cpu().numpy()
            y = Y[0]
            y = y.cpu().numpy()
            print(f"X shape: {x.shape}")
            print(f"Y shape: {y.shape}")
            y_pred = model(x_unsqueezed).squeeze(0).cpu().numpy()
            print(f"Y_pred shape: {y_pred.shape}")
            
            print(f"First ID of First frame of X")
            print(x[0][0])
            print(f"First ID of First frame of Y")
            print(y[0][0])
            print(f"First ID of First frame of Y_pred")
            print(y_pred[0][0])
            
            
            # Write first 100 frames from x only
            # true = pred
            for frame_id, frame in enumerate(x, start=0):
                for v_id, v_id_features in enumerate(frame):
                    print(v_id_features)
                    row = [
                        frame_id,  # Frame
                        v_id,  # ID
                        int(v_id_features[0]),  # X_pred
                        int(v_id_features[1]),  # Y_pred
                        int(v_id_features[0]),  # X_true
                        int(v_id_features[1])   # Y_true
                    ]
                    if np.any(v_id_features < 0):
                        continue
                    csv_writer.writerow(row)
            # Write next 30 frames from y and y_pred
            for seq_idx in range(30):
                y_frame = y[seq_idx]
                y_pred_frame = y_pred[seq_idx]
                for y_id, (y_id_features, y_pred_id_features) in enumerate(zip(y_frame, y_pred_frame)):
                    row = [
                        seq_idx + 100,  # Frame
                        y_id,
                        int(y_pred_id_features[0]),  # X_pred
                        int(y_pred_id_features[1]),  # Y_pred
                        int(y_id_features[0]),  # X_true
                        int(y_id_features[1]),   # Y_true
                    ]
                    if np.any(y_id_features < 0):
                        pass
                        continue
                    csv_writer.writerow(row)
    print(f"Finished exporting predictions to {output_csv_path}")
    return output_csv_path
    
    
if __name__ == '__main__':
    csv_path = generate_predictions_csv(
        model_dir='Model/Saved_Model',
        model_name='rmse_model_1s',
        csv_folder='Data/one_csv',
        video_file='Image_Processing/visualization/merge.mp4',
        output_csv_name='rmse_model_1s_merge_predictions',        
    )
    
    generate_video_from_predictions(
        video_path='Image_Processing/visualization/merge.mp4',
        prediction_path=csv_path,
        output_path='Model/Saved_Model/rmse_model_1s_merge_predictions.mp4',
        IDs_To_Visualize=[9, 11],
        frame_offset=5
    )