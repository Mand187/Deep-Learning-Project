import time
import csv

import wandb.wandb_run
import config as cfg
from NextNet.model_split import FrameTransformer
import wandb
import torch
import torch.profiler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import concurrent.futures
# import gradscaling and mixed precision
from torch.amp import GradScaler, autocast
from Training.jutils import Colors, ColorPrinter
import json



class Trainer:
    def __init__(
        self,
        model,
        trainLoader,
        testLoader,
        save_path,
        model_name,
        plot_path='.',
        device=None,
        wandb_run: wandb.wandb_run.Run =None,
    ):
        self.wandb_run = wandb_run
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        device_id = self.device.index
        
        colors = [
            Colors.BLUE,
            Colors.GREEN,
            Colors.MAGENTA,
            Colors.ORANGE
        ]
        self.printer = ColorPrinter(color=colors[device_id % len(colors)])
        
        num_params = sum(p.numel() for p in model.parameters())
        self.printer.print(f"{self.model_name}: Model has {num_params:,} parameters")
        self.wandb_run.config.update({"params": num_params})
        self.wandb_run.config.update({"device": str(self.device)})
        
        
        self.best_model_file_path = os.path.join(self.model_pickle_dir, f"{self.model_name}.pth")
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.model: FrameTransformer = model
        self.save_path = save_path
        self.plot_path = plot_path
        self.model_name = model_name
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.printer.print(f"{self.model_name}: \nTraining on device: {self.device}")
        self.model.to(self.device)
        self.wandb_dir = os.path.join(self.save_path, 'wandb')
        os.makedirs(self.wandb_dir, exist_ok=True)
        

        self.use_early_stopping = False
        self.patience = 10
        self.delta = 0.0
        
        self.model_artifact = wandb.Artifact(self.model_name, type="model")
        self.model_artifact.add_file(self.best_model_file_path)
        
        
        self.model_top_dir = os.path.join(self.save_path, self.model_name)
        os.makedirs(self.model_top_dir, exist_ok=True)
        self.model_pickle_dir = os.path.join(self.model_top_dir, 'pickles')
        os.makedirs(self.model_pickle_dir, exist_ok=True)
        
        self.history_file_path = os.path.join(self.model_top_dir, 'history.json')
        
        self.printer.print(f"{self.model_name}: Model Dir: {self.model_top_dir}")
        self.train_losses = []
        self.val_losses = []
        self.val_common_losses = []
        self.train_common_losses = []

    def earlyStop(self, enable=True, patience=10, delta=0.0):
        self.use_early_stopping = enable
        self.patience = patience
        self.delta = delta
            

    def train(self, num_epochs=50, learningRate=0.001, criterion=None, optimizer=None, common_loss_fn=None):
        self.printer.print(f"{self.model_name}: Training for {num_epochs} epochs with learning rate {learningRate}")

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learningRate)

        best_val_loss = float('inf')
        patience_counter = 0

        self.train_losses, self.val_losses = [], []
        self.epoch_times = []
        
        scaler = GradScaler(device=self.device)

        total_start_time = time.time()

        with self.wandb_run:
            for epoch in range(1, num_epochs + 1):
                epoch_start = time.time()
                train_loss = 0.0  
                val_loss = 0.0      
                train_common_loss = 0.0
                common_loss = 0.0

                # --- Training ---
                self.model.train()

                running_batch_time = 0.0
                
                
                for inputs, targets in self.trainLoader:
                    batch_start_time = time.time()  
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(inputs)
                        train_batch_loss = criterion(outputs, targets)
                        train_common_loss += common_loss_fn(outputs, targets)

                    # Ensure loss is a scalar

                    scaler.scale(train_batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += train_batch_loss.item()
                    running_batch_time += (time.time() - batch_start_time)
                avg_train_batch_time = running_batch_time / len(self.trainLoader.data_iterable)
                train_loss = train_loss / len(self.trainLoader.data_iterable)
                self.wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/common_loss": train_common_loss,
                        "train/avg_batch_time": avg_train_batch_time,
                    }
                )
                self.train_losses.append(train_loss)
                self.train_common_losses.append(train_common_loss)

                # --- Validation ---
                self.model.eval()
                
                running_val_batch_time = 0.0
                with torch.no_grad():
                    for inputs, targets in self.testLoader:
                        batch_start_time = time.time()
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        with autocast(device_type=self.device.type, dtype=torch.float16):
                            outputs = self.model(inputs)
                            #outputs = outputs.view_as(targets)
                            val_loss += criterion(outputs, targets).item()
                            common_loss += common_loss_fn(outputs, targets).item()
                        running_val_batch_time += (time.time() - batch_start_time)
                avg_val_batch_time = running_val_batch_time / len(self.testLoader.data_iterable)
                val_loss /= len(self.testLoader.data_iterable)
                common_loss /= len(self.testLoader.data_iterable)
                self.wandb_run.log(
                    {
                        "val/loss": val_loss,
                        "val/common_loss": common_loss,
                        "val/avg_batch_time": avg_val_batch_time,
                    }
                )

                self.val_losses.append(val_loss)
                self.val_common_losses.append(common_loss)

                # Calculate epoch time and append to list
                epoch_time = time.time() - epoch_start
                self.epoch_times.append(epoch_time)
                self.wandb_run.log(
                    {
                        "epoch_time": epoch_time,
                    }
                )

                # Update main progress bar
                self.printer.print(f"\r{self.model_name}: Epoch {epoch}/{num_epochs} - Common Loss: {common_loss:.4f} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s", end='')
                    
                self.save_history()
                # --- Early Stopping ---
                if self.use_early_stopping:
                    if val_loss < best_val_loss - self.delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.printer.print(f"{self.model_name}: Best Loss at epoch {epoch}: {best_val_loss:.4f}")
                        self.pool.submit(self.save_model)
                        self.wandb_run.log_artifact(self.model_artifact)
                        #self.pool.submit(self.push_history, epoch)
                        
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            self.printer.print(f"\n{self.model_name}: Early stopping at epoch {epoch}")
                            break

        total_time = time.time() - total_start_time
        self.printer.print(f"\n{self.model_name}: Training complete in {total_time:.2f} seconds, or {total_time / 60:.2f} minutes")
        if epoch < num_epochs:
            self.printer.print(f"{self.model_name}: Training stopped early at epoch {epoch} due to early stopping criteria.")
        else:
            self.printer.print(f"{self.model_name}: Total epochs run: {epoch}")
        self.printer.print(f"{self.model_name}: Average time per epoch: {(total_time / epoch):.2f} seconds")
        self.printer.print(f"{self.model_name}: Inference time per batch: {(total_time / epoch / len(self.trainLoader.data_iterable)):.2f} seconds")
        self.printer.print(f"{self.model_name}: Final Training Loss: {self.train_losses[-1]:.4f}")
        self.printer.print(f"{self.model_name}: Final Validation Loss: {self.val_losses[-1]:.4f}")


        
        self.pool.shutdown(wait=True)
        self.plot_losses()
        
        
        return self.train_losses, self.val_losses, self.epoch_times



    def save_model(self, *args, **kwargs):
        """Save the model to a file"""
        torch.save(self.model, self.best_model_file_path, *args, **kwargs)

    def save_history(self):
        try:
            self.printer.print(f"{self.model_name}: Attempting to save history to {self.history_file_path}")
            history = {
                'common_losses': self.val_common_losses,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'epoch_times': self.epoch_times
            }
        
            with open(self.history_file_path, 'w') as f:
                json.dump(history, f, indent=4)
            self.printer.print(f"{self.model_name}: Successfully saved history to {self.history_file_path}")
        except Exception as e:
            self.printer.print(f"{self.model_name}: ERROR saving history to {self.history_file_path}: {e}", color=Colors.RED)
            # import traceback # Uncomment for detailed traceback
            # self.printer.print(traceback.format_exc()) # Uncomment for detailed traceback

    def push_history(self, epoch):
        os.system(f"git add {self.history_file_path} >> '{self.model_top_dir}/gitlog.txt'")
        os.system(f"git commit -m 'Updated history for {self.model_name} at epoch {epoch}' >> '{self.model_top_dir}/gitlog.txt'")
        os.system(f"git push origin main >> '{self.model_top_dir}/gitlog.txt'")
        
    def plot_losses(self):
        plot_filepath = os.path.join(self.plot_path, f"{self.model_name}_losses.png")
        self.printer.print(f"{self.model_name}: Plotting losses to {plot_filepath}")
        self.printer.print(f"{self.model_name}: Saving losses to {self.model_name}_losses.png")
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'Training and Validation Loss For {self.model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.tight_layout()
        plt.savefig(plot_filepath)
        plt.close()
        self.printer.print(f"{self.model_name}: Plot saved to {plot_filepath}")
        plt.close()
        
        
    def test_model(self, losses):
        if self.best_model_file_path is not None:
            del self.model
            self.model = torch.load(self.best_model_file_path).to(self.device)
        
        self.model.eval()
        total_losses = torch.zeros(len(losses))
        with torch.no_grad():
            for inputs, targets in self.testLoader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                for i, loss_fn in enumerate(losses):
                    loss = loss_fn(outputs, targets)
                    total_losses[i] += loss.item()


        total_loss = total_losses / len(self.testLoader.data_iterable)
        self.printer.print(f"{self.model_name}: Test Losses: {total_loss}")
        return total_loss
