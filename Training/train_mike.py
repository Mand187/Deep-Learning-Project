"""
Training Module for Vehicle Position Prediction Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import signal
import matplotlib.pyplot as plt

def linearOffset(input, offset, target):
    """
    Calculate a linear offset with constraints
    
    Args:
        input: Input value
        offset: Default offset
        target: Target value
        
    Returns:
        Calculated offset value
    """
    # max() ensures offset is always positive or 0
    # min() returns the smaller offset between target - input and default offset
    return max(0, min(offset, target - input))

class Trainer:
    """
    Trainer class for vehicle position prediction model
    """
    def __init__(self,
                 model,
                 train_prefetcher,
                 test_prefetcher,
                 train_loader,
                 test_loader,
                 device,
                 loss_fn=None,
                 ):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            train_prefetcher: Training data prefetcher
            test_prefetcher: Testing data prefetcher  
            train_loader: Training data loader (for dataset size)
            test_loader: Testing data loader (for dataset size)
            device: Device to train on
        """
        self.model = model
        self.train_prefetcher = train_prefetcher
        self.test_prefetcher = test_prefetcher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Training parameters
        self.loss_function = loss_fn
        self.optimizer = optim.Adam(
            params=model.parameters(),
            lr=0.001,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-5
        )
        
        # Training metrics
        self.avg_train_batch_loss_per_epoch = []
        self.avg_test_batch_loss_per_epoch = []
        self.train_accuracy_per_epoch = []
        self.test_accuracy_per_epoch = []
        self.best_test_accuracy = 0
        
        # Signal handling for interruption
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle interrupt signal (Ctrl+C)"""
        self.interrupted = True
        print("Interrupt received. Flag set...")
    
    def train(self, epochs=50, save_checkpoints=False, minimum_test_accuracy=0):
        """
        Train the model
        
        Args:
            epochs: Number of epochs to train for (-1 for infinite)
            save_checkpoints: Whether to save model checkpoints
            minimum_test_accuracy: Minimum test accuracy to stop training
            
        Returns:
            model: Trained model
        """
        epoch_iterator = 0
        train_start_time = time.time()
        
        while not self.interrupted and (
            (epoch_iterator < epochs or epochs == -1) or 
            train_epoch_accuracy < test_epoch_accuracy + linearOffset(input=test_epoch_accuracy, offset=3, target=99) or 
            self.best_test_accuracy < minimum_test_accuracy
        ):
            epoch_start_time = time.time()
            self.model.train()
            
            num_correct_in_epoch = 0
            total_train_loss_in_epoch = 0
            for X_train_batch, Y_train_batch in self.train_prefetcher:
                X_train_batch = X_train_batch.to(self.device, non_blocking=True)
                Y_train_batch = Y_train_batch.to(self.device, non_blocking=True)
                
                Y_train_pred_logits = self.model(X_train_batch)
                
                train_batch_loss = self.loss_function(Y_train_pred_logits, Y_train_batch.type(torch.int64))
                
                self.optimizer.zero_grad()
                train_batch_loss.backward()
                self.optimizer.step()
                
                num_correct_in_epoch += torch.eq(Y_train_pred_logits.argmax(dim=1), Y_train_batch).sum().item()
                total_train_loss_in_epoch += train_batch_loss
            
            self.model.eval()
            
            with torch.inference_mode():
                train_epoch_average_batch_loss = total_train_loss_in_epoch / len(self.train_loader)
                self.avg_train_batch_loss_per_epoch.append(train_epoch_average_batch_loss)
                
                train_epoch_accuracy = num_correct_in_epoch / len(self.train_loader.dataset) * 100
                self.train_accuracy_per_epoch.append(train_epoch_accuracy)
                
                num_correct_in_epoch = 0
                total_test_loss_in_epoch = 0
                for X_test_batch, Y_test_batch in self.test_prefetcher:
                    X_test_batch = X_test_batch.to(self.device, non_blocking=True)
                    Y_test_batch = Y_test_batch.to(self.device, non_blocking=True)
                
                    Y_test_pred_logits = self.model(X_test_batch)
                
                    test_batch_loss = self.loss_function(Y_test_pred_logits, Y_test_batch.type(torch.int64))
            
                    num_correct_in_epoch += torch.eq(Y_test_pred_logits.argmax(dim=1), Y_test_batch).sum().item()
                    
                    total_test_loss_in_epoch += test_batch_loss
                
                test_epoch_average_batch_loss = total_test_loss_in_epoch / len(self.test_loader)
                self.avg_test_batch_loss_per_epoch.append(test_epoch_average_batch_loss)
                
                test_epoch_accuracy = num_correct_in_epoch / len(self.test_loader.dataset) * 100
                self.test_accuracy_per_epoch.append(test_epoch_accuracy)
            
                epoch_time = time.time() - epoch_start_time
                est_remaining_time = (epochs - epoch_iterator - 1) * epoch_time / 60
                print(f"epoch: {epoch_iterator} \t| train loss: {train_epoch_average_batch_loss:.5f}, train accuracy: {train_epoch_accuracy:.2f}% \t| test loss: {test_epoch_average_batch_loss:.5f}, test accuracy: {test_epoch_accuracy:.2f}% \t| TTG: {int(est_remaining_time):02}:{int((est_remaining_time - int(est_remaining_time))*60):02}")
                
                new_best_model = test_epoch_accuracy > minimum_test_accuracy and test_epoch_accuracy > self.best_test_accuracy
                if new_best_model: 
                    self.best_test_accuracy = test_epoch_accuracy
                    print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑ NEW BEST MODEL ↑↑↑↑↑↑↑↑↑↑↑↑↑")
                    
                if save_checkpoints and new_best_model: 
                    torch.save(self.model.state_dict(), 'Saved_Models/best_model.pth')
                    print(f"↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ SAVED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
                
                epoch_iterator += 1
        
        total_train_time = (time.time() - train_start_time) / 60
        average_epoch_time = total_train_time / epoch_iterator

        print(f"Total Training Time: {int(total_train_time):02}:{int((total_train_time - int(total_train_time))*60):02}")
        print(f"Average Epoch Time: {int(average_epoch_time):02}:{int((average_epoch_time - int(average_epoch_time))*60):02}")
        
        return self.model
    
    def plot_metrics(self):
        """Plot training and testing metrics"""
        with torch.inference_mode():
            avg_train_batch_loss_per_epoch = torch.tensor(self.avg_train_batch_loss_per_epoch).cpu()
            avg_test_batch_loss_per_epoch = torch.tensor(self.avg_test_batch_loss_per_epoch).cpu()
            
            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

            # First subplot
            axs[0].scatter(
                x=range(len(avg_train_batch_loss_per_epoch)), 
                y=avg_train_batch_loss_per_epoch, 
                label="Training Loss"
            )
            axs[0].scatter(
                x=range(len(avg_test_batch_loss_per_epoch)), 
                y=avg_test_batch_loss_per_epoch, 
                label="Test / Validation Loss"
            )
            axs[0].set_title('Loss Per Epoch')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')
            axs[0].legend()
            axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Second subplot
            axs[1].scatter(
                x=range(len(self.train_accuracy_per_epoch)), 
                y=self.train_accuracy_per_epoch, 
                label="Training Accuracy"
            )
            axs[1].scatter(
                x=range(len(self.test_accuracy_per_epoch)), 
                y=self.test_accuracy_per_epoch, 
                label="Test / Validation Accuracy"
            )
            axs[1].set_title('Accuracy Per Epoch')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Accuracy %')
            axs[1].legend()
            axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Adjust layout and display the plot
            plt.tight_layout()  # Avoid overlap between subplots
            plt.show()
    
    def save_model(self, path):
        """Save the model to a file"""
        torch.save(self.model, path)