import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
from pathlib import Path

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


def compute_accuracy(predictions, targets, threshold=2.0):
    """
    Compute % of predictions within a Euclidean distance threshold of the ground truth.
    predictions, targets: [batch_size, pred_len, num_ids, 2]
    """
    dist = torch.norm(predictions - targets, dim=-1)  # [batch, pred_len, num_ids]
    accurate = (dist < threshold).float()
    return accurate.mean().item() * 100  # percentage


class Trainer:
    def __init__(self, model, trainLoader, testLoader, model_path, device=None):
        self.model = model
        self.model_path = model_path
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTraining on device: {self.device}")
        self.model.to(self.device)

        self.use_early_stopping = False
        self.patience = 10
        self.delta = 0.0

    def earlyStop(self, enable=True, patience=10, delta=0.0):
        self.use_early_stopping = enable
        self.patience = patience
        self.delta = delta

    def train(self, num_epochs=50, learningRate=0.001, criterion=None, optimizer=None):

        if criterion is None:
            criterion = nn.MSELoss()

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learningRate)

        best_val_loss = float('inf')
        patience_counter = 0

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        epoch_times = []
        

        total_start_time = time.time()
        pbar = tqdm(range(1, num_epochs + 1), desc="Training Progress")

        for epoch in pbar:
            epoch_start = time.time()

            # --- Training ---
            self.model.train()
            running_loss, running_acc = 0.0, 0.0

            train_iterator = tqdm(
                self.trainLoader,
                desc=f"Epoch {epoch}/{num_epochs} [Train]",
                leave=False
            )

            for inputs, targets in train_iterator:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc = compute_accuracy(outputs.detach(), targets)
                running_acc += acc

                train_iterator.set_postfix({"batch loss": f"{loss.item():.4f}"})

            train_loss = running_loss / len(self.trainLoader)
            train_acc = running_acc / len(self.trainLoader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # --- Validation ---
            self.model.eval()
            val_loss, val_acc_total = 0.0, 0.0

            val_iterator = tqdm(
                self.testLoader,
                desc=f"Epoch {epoch}/{num_epochs} [Val]",
                leave=False
            )

            with torch.no_grad():
                for inputs, targets in val_iterator:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    acc = compute_accuracy(outputs, targets)
                    val_acc_total += acc

                    val_iterator.set_postfix({"batch loss": f"{loss.item():.4f}"})

            val_loss /= len(self.testLoader)
            val_acc = val_acc_total / len(self.testLoader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Calculate epoch time and append to list
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Update main progress bar
            pbar.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Train Acc": f"{train_acc:.2f}%",
                "Val Loss": f"{val_loss:.4f}",
                "Val Acc": f"{val_acc:.2f}%",
                "Time": f"{epoch_time:.2f}s"
            })

            # --- Early Stopping ---
            if self.use_early_stopping:
                if val_loss < best_val_loss - self.delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(self.model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

        total_time = time.time() - total_start_time
        print(f"\nTraining complete in {total_time:.2f} seconds, or {total_time / 60:.2f} minutes")
        if epoch < num_epochs:
            print(f"Training stopped early at epoch {epoch} due to early stopping criteria.")
        else:
            print(f"Total epochs run: {epoch}")
        print(f"Average time per epoch: {(total_time / epoch):.2f} seconds")
        print(f"Inference time per batch: {(total_time / epoch / len(self.trainLoader)):.2f} seconds")
        print(f"Final Training Loss: {train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}")
        print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
        print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")

        return train_losses, val_losses, train_accs, val_accs, epoch_times

    def save_model(self, path):
        """Save the model to a file"""
        torch.save(self.model, path)
    
    def export_predictions_to_csv(self, data_loader=None, csv_path='predictions.csv', include_targets=True, 
                                batch_limit=None, sample_offset=0):
        """
        Export model predictions to a CSV file in the format: frame, id, X, Y.
        
        Args:
            data_loader (DataLoader, optional): DataLoader to use for predictions. 
                                              If None, uses the test loader. Default: None
            csv_path (str): Path where the CSV will be saved. Default: 'predictions.csv'
            include_targets (bool): Whether to include ground truth targets in the CSV. Default: True
            batch_limit (int, optional): Limit the number of batches to process. Default: None
            sample_offset (int): Offset for the sample_idx (useful for continuing from previous exports). Default: 0
        
        Returns:
            str: Path to the saved CSV file
        """
        if data_loader is None:
            data_loader = self.testLoader
        
        self.model.eval()
        
        # Create directory if it doesn't exist
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            Path(csv_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine the shape of outputs
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                sample_output = self.model(inputs)
                output_shape = sample_output.shape
                break
        
        # Set up headers for frame, id, X, Y format
        headers = ["frame", "id", "X", "Y"]
        if include_targets:
            headers = ["frame", "id", "pred_X", "pred_Y", "target_X", "target_Y"]
            
        print(f"Exporting predictions to {csv_path}...")
        
        # For tracking progress
        total_rows = 0
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            
            with torch.no_grad():
                sample_idx = sample_offset
                for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Exporting predictions")):
                    if batch_limit and batch_idx >= batch_limit:
                        break
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    # Check output shape and process accordingly
                    if len(output_shape) == 4:  # [batch, pred_len, num_ids, 2]
                        batch_size, pred_len, num_ids, _ = outputs.shape
                        
                        # Process each sample in the batch
                        for i in range(batch_size):
                            curr_output = outputs[i].cpu().numpy()  # [pred_len, num_ids, 2]
                            curr_target = targets[i].cpu().numpy() if include_targets else None
                            
                            # For each timestep (frame)
                            for t in range(pred_len):
                                # For each identity
                                for id_idx in range(num_ids):
                                    x_pred, y_pred = curr_output[t, id_idx]
                                    
                                    if include_targets:
                                        x_target, y_target = curr_target[t, id_idx]
                                        csv_writer.writerow([
                                            sample_idx + t,  # frame 
                                            id_idx,          # id
                                            f"{x_pred:.6f}", # pred_X
                                            f"{y_pred:.6f}", # pred_Y
                                            f"{x_target:.6f}", # target_X
                                            f"{y_target:.6f}"  # target_Y
                                        ])
                                    else:
                                        csv_writer.writerow([
                                            sample_idx + t,  # frame
                                            id_idx,          # id
                                            f"{x_pred:.6f}", # X
                                            f"{y_pred:.6f}"  # Y
                                        ])
                                    
                                    total_rows += 1
                    else:
                        # For other output formats, print warning and export as is
                        print(f"Warning: Output shape {output_shape} doesn't match expected [batch, pred_len, num_ids, 2]")
                        print("Exporting in basic format instead")
                        
                        for i in range(outputs.shape[0]):
                            # Assume simple output is a single (x,y) coordinate pair
                            if len(outputs[i].shape) == 1 and outputs[i].shape[0] == 2:
                                x_pred, y_pred = outputs[i].cpu().numpy()
                                
                                if include_targets:
                                    x_target, y_target = targets[i].cpu().numpy()
                                    csv_writer.writerow([
                                        sample_idx,       # frame
                                        0,                # id
                                        f"{x_pred:.6f}",  # pred_X
                                        f"{y_pred:.6f}",  # pred_Y
                                        f"{x_target:.6f}", # target_X
                                        f"{y_target:.6f}"  # target_Y
                                    ])
                                else:
                                    csv_writer.writerow([
                                        sample_idx,       # frame
                                        0,                # id
                                        f"{x_pred:.6f}",  # X
                                        f"{y_pred:.6f}"   # Y
                                    ])
                            else:
                                # For unsupported formats, just export the raw data
                                print(f"Warning: Unsupported output shape for sample: {outputs[i].shape}")
                                continue
                            
                            total_rows += 1
                    
                    # Increment sample index for next batch
                    sample_idx += 1
        
        print(f"Successfully exported {total_rows} rows to {csv_path}")
        return csv_path
    
    def compute_and_export_errors(self, data_loader=None, csv_path='prediction_errors.csv', 
                                 batch_limit=None, error_metrics=None):
        """
        Compute prediction errors and export to CSV with frame, id, error format.
        
        Args:
            data_loader (DataLoader, optional): DataLoader to use. If None, uses test loader.
            csv_path (str): Path where the CSV will be saved. Default: 'prediction_errors.csv'
            batch_limit (int, optional): Limit the number of batches to process. Default: None
            error_metrics (list, optional): List of error metrics to compute. 
                                           Default: ['mse', 'mae', 'euclidean']
        
        Returns:
            tuple: (Path to the saved CSV file, dictionary with overall errors)
        """
        if data_loader is None:
            data_loader = self.testLoader
            
        if error_metrics is None:
            error_metrics = ['mse', 'mae', 'euclidean']
            
        self.model.eval()
        
        # Create directory if it doesn't exist
        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            Path(csv_dir).mkdir(parents=True, exist_ok=True)
            
        # Determine the shape of outputs to create proper headers
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                sample_output = self.model(inputs)
                output_shape = sample_output.shape
                break
                
        # Create headers in the frame, id, error format
        headers = ["frame", "id"]
        for metric in error_metrics:
            headers.append(metric)
                
        print(f"Computing and exporting errors to {csv_path}...")
        
        # For computing overall statistics
        overall_stats = {metric: [] for metric in error_metrics}
        total_rows = 0
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            
            with torch.no_grad():
                sample_idx = 0
                for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Computing errors")):
                    if batch_limit and batch_idx >= batch_limit:
                        break
                        
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    # Check output shape and process accordingly
                    if len(output_shape) == 4:  # [batch, pred_len, num_ids, 2]
                        batch_size, pred_len, num_ids, _ = outputs.shape
                        
                        # Process each sample in the batch
                        for i in range(batch_size):
                            pred = outputs[i]  # [pred_len, num_ids, 2]
                            target = targets[i]  # [pred_len, num_ids, 2]
                            
                            # For each timestep (frame)
                            for t in range(pred_len):
                                # For each identity
                                for id_idx in range(num_ids):
                                    pred_point = pred[t, id_idx]  # [2]
                                    target_point = target[t, id_idx]  # [2]
                                    
                                    # Calculate errors for this specific point
                                    row = [sample_idx + t, id_idx]
                                    
                                    for metric in error_metrics:
                                        if metric == 'mse':
                                            error = torch.mean((pred_point - target_point) ** 2).item()
                                        elif metric == 'mae':
                                            error = torch.mean(torch.abs(pred_point - target_point)).item()
                                        elif metric == 'euclidean':
                                            error = torch.norm(pred_point - target_point).item()
                                        
                                        row.append(f"{error:.6f}")
                                        overall_stats[metric].append(error)
                                    
                                    csv_writer.writerow(row)
                                    total_rows += 1
                    else:
                        # For other output formats, print warning and export as is
                        print(f"Warning: Output shape {output_shape} doesn't match expected [batch, pred_len, num_ids, 2]")
                        print("Exporting in basic format instead")
                        
                        for i in range(outputs.shape[0]):
                            if len(outputs[i].shape) == 1 and outputs[i].shape[0] == 2:
                                pred_point = outputs[i]
                                target_point = targets[i]
                                
                                row = [sample_idx, 0]  # frame, id
                                
                                for metric in error_metrics:
                                    if metric == 'mse':
                                        error = torch.mean((pred_point - target_point) ** 2).item()
                                    elif metric == 'mae':
                                        error = torch.mean(torch.abs(pred_point - target_point)).item()
                                    elif metric == 'euclidean':
                                        error = torch.norm(pred_point - target_point).item()
                                    
                                    row.append(f"{error:.6f}")
                                    overall_stats[metric].append(error)
                                
                                csv_writer.writerow(row)
                                total_rows += 1
                            else:
                                print(f"Warning: Unsupported output shape for sample: {outputs[i].shape}")
                                continue
                    
                    # Increment sample index for next batch
                    sample_idx += 1
        
        # Compute overall statistics
        stats_summary = {}
        for metric, values in overall_stats.items():
            if values:
                stats_summary[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values)
                }
        
        print(f"Successfully exported {total_rows} rows to {csv_path}")
        print("\nOverall Error Statistics:")
        for metric, stats in stats_summary.items():
            print(f"  {metric.upper()}: mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
                  f"min={stats['min']:.4f}, max={stats['max']:.4f}, std={stats['std']:.4f}")
        
        return csv_path, stats_summary