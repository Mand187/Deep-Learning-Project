import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import concurrent.futures
from Training.jutils import Colors, ColorPrinter



class Trainer:
    def __init__(self, model, trainLoader, testLoader, save_path, model_name, plot_path='.', device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_id = self.device.index
        
        colors = [
            Colors.BLUE,
            Colors.GREEN,
            Colors.MAGENTA,
            Colors.ORANGE
        ]
        self.printer = ColorPrinter(color=colors[device_id % len(colors)])
        
        
        self.best_model_file_path = None
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.model = model
        self.save_path = save_path
        self.plot_path = plot_path
        self.model_name = model_name
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.printer.print(f"{self.model_name}: \nTraining on device: {self.device}")
        self.model.to(self.device)
        
        
        

        self.use_early_stopping = False
        self.patience = 10
        self.delta = 0.0
        
        self.model_path = os.path.join(self.save_path, self.model_name)
        os.makedirs(self.model_path, exist_ok=True)
        self.printer.print(f"{self.model_name}: Model path: {self.model_path}")
        self.train_losses = []
        self.val_losses = []

    def earlyStop(self, enable=True, patience=10, delta=0.0):
        self.use_early_stopping = enable
        self.patience = patience
        self.delta = delta

    def train(self, num_epochs=50, learningRate=0.001, criterion=None, optimizer=None):
        self.printer.print(f"{self.model_name}: Training for {num_epochs} epochs with learning rate {learningRate}")
        if criterion is None:
            criterion = nn.MSELoss(reduction='none')

            def masked_loss(outputs, targets, mask):
                loss = criterion(outputs, targets)
                loss = loss * mask.unsqueeze(-1)  # Apply mask to the loss
                return loss.mean()

            def create_mask(targets, padding_value=0):
                return (targets != padding_value).float()

            # Wrap the criterion to include masking
            def criterion(outputs, targets):
                mask = create_mask(targets)
                return masked_loss(outputs, targets, mask)

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learningRate)

        best_val_loss = float('inf')
        patience_counter = 0

        self.train_losses, self.val_losses = [], []
        epoch_times = []

        total_start_time = time.time()


        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0  # Initialize for the current epoch
            val_loss = 0.0      # Initialize for the current epoch

            # --- Training ---
            self.model.train()

            for inputs, targets in self.trainLoader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Ensure loss is a scalar
                if loss.ndim > 0: # If loss is not a scalar
                    loss = loss.mean() # Reduce to scalar by taking the mean

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.trainLoader.data_iterable)
            self.train_losses.append(train_loss)

            # --- Validation ---
            self.model.eval()
            

            with torch.no_grad():
                for inputs, targets in self.testLoader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    outputs = outputs.view_as(targets)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()



            val_loss /= len(self.testLoader.data_iterable)

            self.val_losses.append(val_loss)

            # Calculate epoch time and append to list
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Update main progress bar
            self.printer.print(f"\r{self.model_name}: Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s")

            # --- Early Stopping ---
            if self.use_early_stopping:
                if val_loss < best_val_loss - self.delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(epoch)
                    self.printer.print(f"{self.model_name}: Best Loss at epoch {epoch}: {best_val_loss:.4f}")
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


        
        self.save_model(epoch)
        self.plot_losses()
        
        self.pool.shutdown(wait=True)
        
        return self.train_losses, self.val_losses, epoch_times



    def save_model(self, epoch, *args, **kwargs):
        """Save the model to a file"""
        self.best_model_file_path = os.path.join(self.model_path, f"{self.model_name}_epoch_{epoch}.pth")
        self.printer.print(f"{self.model_name}: Saving model to {self.best_model_file_path}")
        self.pool.submit(
            torch.save,
            self.model,
            self.best_model_file_path,
            *args,
            **kwargs
        )
        #torch.save(self.model, self.best_model_file_path)
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
        #plt.close()
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

