import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

class Trainer:
    def __init__(self, model, trainLoader, testLoader, device=None):
        self.model = model
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

    def train(self, num_epochs=50, learningRate=0.001, criterion = None, optimizer=None):
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learningRate)

        best_val_loss = float('inf')
        patience_counter = 0

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        true_labels_all, pred_labels_all = [], []
        epoch_times = []  # Track time per epoch

        total_start_time = time.time()
        pbar = tqdm(range(1, num_epochs + 1), desc="Training Progress")

        for epoch in pbar:
            epoch_start = time.time()

            # --- Training ---
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            # Create a progress bar for the training batches
            train_iterator = tqdm(
                self.trainLoader, 
                desc=f"Epoch {epoch}/{num_epochs} [Train]",
                leave=False
            )
            
            for inputs, labels in train_iterator:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                if isinstance(outputs, dict) and 'logits' in outputs:
                    outputs = outputs['logits']
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                
                # Update training batch progress bar
                train_iterator.set_postfix({"batch loss": f"{loss.item():.4f}"})

            train_loss = running_loss / len(self.trainLoader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # --- Validation ---
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            true_labels, pred_labels = [], []

            # Create a progress bar for the validation batches
            val_iterator = tqdm(
                self.testLoader, 
                desc=f"Epoch {epoch}/{num_epochs} [Val]",
                leave=False
            )
            
            with torch.no_grad():
                for inputs, labels in val_iterator:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)

                    true_labels.extend(labels.cpu().numpy())
                    pred_labels.extend(preds.cpu().numpy())
                    
                    # Update validation batch progress bar
                    val_iterator.set_postfix({"batch loss": f"{loss.item():.4f}"})

            val_loss /= len(self.testLoader)
            val_acc = 100. * correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            true_labels_all = true_labels
            pred_labels_all = pred_labels

            # Calculate epoch time and append to list
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # Update main epoch progress bar
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

        return train_losses, val_losses, train_accs, val_accs, epoch_times, true_labels_all, pred_labels_all