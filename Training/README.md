# Imports

from Training.train import *
from Training.customLoss import *

# Example Train Use:

# Call with model and Train Loader and testLoader:
trainer = Trainer(model, trainLoader, testLoader)

# If you want early stopping :
trainer.earlyStop(True, patience=20, delta=0.5)

# Train Function:
train_losses, val_losses, train_acc, val_accs, times_ResNet, y_true, y_pred = 
trainer.train(num_epochs=NUM_EPOCH, learningRate=LEARNING_RATE)

# If you have a different cirterion or optimzer:

trainer.train(num_epochs=NUM_EPOCH, learningRate=LEARNING_RATE, criterion=CIRTERION, optimizer = OPTIMIZER)

# If no optimizer or criterion is defined then will default to:

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(self.model.parameters(), lr=learningRate)