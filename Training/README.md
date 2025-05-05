# Training Module Usage Guide

## Imports

```python
from Training.train import *
from Training.customLoss import *
```

## Example Usage

### Initialize Trainer
Create a `Trainer` instance with your model, training data loader, and test data loader:
```python
trainer = Trainer(model, trainLoader, testLoader)
```

### Enable Early Stopping (Optional)
Enable early stopping with specified parameters:
```python
trainer.earlyStop(True, patience=20, delta=0.5)
```

### Train the Model
Train the model and retrieve training metrics:
```python
train_losses, val_losses, train_acc, val_accs, times_ResNet, y_true, y_pred = trainer.train(
    num_epochs=NUM_EPOCH, 
    learningRate=LEARNING_RATE
)
```

### Custom Criterion or Optimizer (Optional)
Use a custom loss function or optimizer during training:
```python
trainer.train(
    num_epochs=NUM_EPOCH, 
    learningRate=LEARNING_RATE, 
    criterion=CIRTERION, 
    optimizer=OPTIMIZER
)
```

### Default Criterion and Optimizer
If no custom criterion or optimizer is provided, the following defaults are used:
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(self.model.parameters(), lr=learningRate)
```