import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


def compute_accuracy(predictions, targets, threshold=0.1):
    """
    Compute % of predictions within a Euclidean distance threshold of the ground truth.
    predictions, targets: [batch_size, pred_len, num_ids, 2]
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")
    dist = torch.norm(predictions - targets, dim=-1)  # [batch, pred_len, num_ids]
    accurate = (dist < threshold).float()
    return accurate.mean().item() * 100  # percentage


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

    def train(self, num_epochs=50, learningRate=0.001, criterion=None, optimizer=None, model_path=None):

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

            train_loss = running_loss / len(self.trainLoader.data_iterable)
            train_acc = running_acc / len(self.trainLoader.data_iterable)
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
                    outputs = outputs.view_as(targets)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    acc = compute_accuracy(outputs, targets)
                    val_acc_total += acc

                    val_iterator.set_postfix({"batch loss": f"{loss.item():.4f}"})

            val_loss /= len(self.testLoader.data_iterable)
            val_acc = val_acc_total / len(self.testLoader.data_iterable)
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
        print(f"Inference time per batch: {(total_time / epoch / len(self.trainLoader.data_iterable)):.2f} seconds")
        print(f"Final Training Loss: {train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {val_losses[-1]:.4f}")
        print(f"Final Training Accuracy: {train_accs[-1]:.2f}%")
        print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
        
        torch.save(self.model, model_path)

        return train_losses, val_losses, train_accs, val_accs, epoch_times

    def save_model(self, model, path):
        """Save the model to a file"""
        torch.save(model, path)

