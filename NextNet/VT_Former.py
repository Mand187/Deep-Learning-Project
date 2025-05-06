# %%

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
os.listdir('/content/drive/MyDrive/')

# %%
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


path="/content/drive/MyDrive/Dataset/Dataset.csv"

try:
    df = pd.read_csv(path, nrows = 3000)
    print("Data loaded successfully!")

    # Step 5: Display the first 5 rows of the DataFrame
    print(df.head())

    print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

except FileNotFoundError:
    print("File not found. Please check the path and try again.")

# %%
#Prepare the nodes
node_features = df[['Local X', 'Local Y', 'Vehicle Velocity', 'Vehicle Accl']]
node_features = (node_features - node_features.mean()) / node_features.std()
node_features_tensor = torch.tensor(node_features.values, dtype=torch.float)

# %%
print("\nFirst 5 Node Features:")
print(node_features_tensor[:5])  # Change 5 to any number to see more/less nodes

# %%
# Step 3: Create edges based on Preceding/Following Vehicle relationships and a proximity radius
def create_edges_with_proximity(df, radius):
    edge_index = []

    # Iterate through the dataframe to establish edges between vehicles
    for i, row in df.iterrows():
        preceding_vehicle = row['Preceeding Vehicle']
        following_vehicle = row['Following Vehicle']

        # Helper function to check distance and create edge
        def add_edge_if_within_radius(i, vehicle_id):
            if vehicle_id != 0:
                vehicle_index = df[df['Vehicle ID'] == vehicle_id].index
                if len(vehicle_index) > 0:
                    vehicle_index = vehicle_index[0]

                    # Calculate the Euclidean distance between the vehicles
                    distance = np.sqrt((row['Local X'] - df.loc[vehicle_index]['Local X'])**2 +
                                       (row['Local Y'] - df.loc[vehicle_index]['Local Y'])**2)

                    # If within the radius, create bidirectional edges
                    if distance <= radius:
                        edge_index.append([i, vehicle_index])
                        edge_index.append([vehicle_index, i])  # Add both directions for an undirected graph

        # Check and add edges for the preceding vehicle
        add_edge_if_within_radius(i, preceding_vehicle)

        # Check and add edges for the following vehicle
        add_edge_if_within_radius(i, following_vehicle)

    # Convert edge_index to tensor and transpose it to (2, num_edges)
    return torch.tensor(edge_index, dtype=torch.long).t()

# Define a proximity radius (e.g., 100 feet)
proximity_radius = 100

# Step 4: Create edge index (edges) based on Preceding/Following Vehicle and proximity
edge_index = create_edges_with_proximity(df, proximity_radius)

# Step 5: Create a PyTorch Geometric Data object
data = Data(x=node_features_tensor, edge_index=edge_index)

# Step 6: Check the Data object
print(data)


# %%
# Define the GATNet Module
class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=0.2)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=0.2)
        self.gat4 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Match output_dim with Transformer input_dim

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = F.elu(self.gat3(x, edge_index))
        x = F.elu(self.gat4(x, edge_index))
        x = self.fc(x)  # Final output dimension should match Transformer’s input_dim
        return x

# %%
# Step 4: Model Hyperparameters
input_dim = 4  # Number of node features (Local X, Local Y, Vehicle Velocity, Vehicle Accl)
hidden_dim = 32  # Number of hidden units in GAT layers
output_dim = 2  # Predicting 2D coordinates (x, y)
num_heads = 4   # Number of attention heads

# %%
# Instantiate the model
model = GATNet(input_dim, hidden_dim, output_dim, num_heads)

# %%
# Step 5: Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()  # Using Mean Squared Error for regression (predicting coordinates)

# %%
# Step: Add the next positions (Next X, Next Y) as the target labels
df['Next X'] = df.groupby('Vehicle ID')['Local X'].shift(-1)
df['Next Y'] = df.groupby('Vehicle ID')['Local Y'].shift(-1)

# Remove NaN values (last row of each vehicle will not have a next position)
df = df.dropna(subset=['Next X', 'Next Y'])

# Standardize node features (if not done previously)
scaler_features = StandardScaler()
node_features = scaler_features.fit_transform(df[['Local X', 'Local Y', 'Vehicle Velocity', 'Vehicle Accl']].values)
node_features_tensor = torch.tensor(node_features, dtype=torch.float)

# Standardize target values for Next X and Next Y
scaler_target = StandardScaler()
target = scaler_target.fit_transform(df[['Next X', 'Next Y']].values)
target_tensor = torch.tensor(target, dtype=torch.float)

# Create the PyTorch Geometric Data object with the standardized targets
data = Data(x=node_features_tensor, edge_index=edge_index, y=target_tensor)

# %%
# Split the node features and targets into training and testing sets
train_mask, test_mask = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

# --- Reindex edge_index for training data ---
# 1. Filter edges based on train_mask (instead of indexing directly)
train_edge_index = edge_index[:, np.isin(edge_index[0], train_mask) & np.isin(edge_index[1], train_mask)]

# 2. Get unique node indices in the filtered training set
train_nodes = np.unique(train_edge_index)

# 3. Create a mapping from original node indices to new indices in the training set
node_map = {node: i for i, node in enumerate(train_nodes)}

# 4. Reindex train_edge_index using the mapping
train_edge_index = torch.tensor([[node_map[n1], node_map[n2]]
                                  for n1, n2 in train_edge_index.t().tolist()], dtype=torch.long).t()

# --- Repeat the same process for test data ---
# 1. Filter edges based on test_mask
test_edge_index = edge_index[:, np.isin(edge_index[0], test_mask) & np.isin(edge_index[1], test_mask)]

# 2. Get unique node indices in the filtered test set
test_nodes = np.unique(test_edge_index)

# 3. Create a mapping from original node indices to new indices in the test set
node_map = {node: i for i, node in enumerate(test_nodes)}

# 4. Reindex test_edge_index using the mapping
test_edge_index = torch.tensor([[node_map[n1], node_map[n2]]
                                 for n1, n2 in test_edge_index.t().tolist()], dtype=torch.long).t()


train_data = Data(x=node_features_tensor[train_mask],
                  edge_index=train_edge_index,  # Use reindexed edge_index
                  y=target_tensor[train_mask])

test_data = Data(x=node_features_tensor[test_mask],
                 edge_index=test_edge_index,   # Use reindexed edge_index
                 y=target_tensor[test_mask])

# %%
# Step 6: Define a training loop
def train_model(model, data, epochs=100):
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients

        # Forward pass through the model
        out = model(data)

        # Assuming the ground truth for trajectory is in the Data object (use actual labels in practice)
        target = data.y  # Placeholder for ground truth labels

        # Compute the loss
        loss = loss_fn(out, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print loss at every epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')



# %%
def test_model(model, test_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for faster testing
        out = model(test_data)  # Make predictions on the test data

        # Compute the test loss (MSE)
        test_loss = loss_fn(out, test_data.y)
        print(f'Test Loss (MSE): {test_loss.item()}')

        # Convert predictions and actual values to numpy arrays
        predicted = out.numpy()
        actual = test_data.y.numpy()

        # Calculate ADE (Average Displacement Error)
        # ADE is the mean Euclidean distance between each predicted and actual trajectory point
        displacement_errors = np.linalg.norm(predicted - actual, axis=1)
        ade = np.mean(displacement_errors)
        print(f'Average Displacement Error (ADE): {ade}')

        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        # Display predicted vs actual for the first 5 examples
        print("\nPredicted vs Actual (First 5 examples):")
        print("Predicted:", predicted[:5])
        print("Actual:", actual[:5])

# %%
# First, train the model
train_model(model, train_data, epochs=100)

# After training, evaluate the model on the test set
test_model(model, test_data)


# %% [markdown]
# *Transformer Module*

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# Define the DecoderOnlyTransformer Module
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, seq_len, input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim):
        super(DecoderOnlyTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)  # Embedding layer to match GATNet output to embed_dim
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))  # Positional encoding

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])

        # Linear layer to map the final output to desired output_dim (2 for (x, y))
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # Embed GATNet output and add positional encoding
        embed_dim=32
        seq_len=10
        device = x.device
        #self.embedding = nn.Linear(input_dim, embed_dim)
        # Change here: seq_len should be num_nodes
        #self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.embedding = nn.Linear(embed_dim, embed_dim)

        batch_size = x.shape[0]  # Get batch size from input
        self.positional_encoding = nn.Parameter(torch.randn(1, self.embed_dim, self.embed_dim), requires_grad=True).to(device)
        positional_encoding = self.positional_encoding.repeat(batch_size, 1, 1)  # Repeat for batch size
        #self.positional_encoding = nn.Parameter(torch.randn(1, x.shape[1], self.embed_dim)).to(x.device)

        #x = self.embedding(x) + self.positional_encoding
        x = self.embedding(x) + positional_encoding[:, :x.shape[1], :]

        # Pass through each decoder layer
        for layer in self.decoder_layers:
            x = layer(x, x)  # Both tgt and memory are x for decoder-only

        # Project to the output dimension (e.g., 2 for (x, y) positions)
        output = self.fc_out(x)
        return output

# Sample Execution Code

# Instantiate the GAT and Transformer models
gat_model = GATNet(input_dim=4, hidden_dim=32, output_dim=32, num_heads=4)  # output_dim = 32 to match Transformer’s embed_dim
transformer_model = DecoderOnlyTransformer(seq_len=10, input_dim=32, embed_dim=32, num_heads=4, ff_dim=128, num_layers=8, output_dim=2)

# Sample data for GATNet
from torch_geometric.data import Data
input_dim = 4
num_nodes = 32
x = torch.rand((num_nodes, input_dim))  # Random node features
edge_index = torch.randint(0, num_nodes, (2, 150))  # Random edges for the graph
data = Data(x=x, edge_index=edge_index)

# Run GATNet to obtain embeddings
gat_output = gat_model(data)  # Shape: (num_nodes, 32) - (32 is the embed_dim for Transformer)

# Reshape GATNet output to match Transformer input
gat_output = gat_output.unsqueeze(0)  # Add batch dimension, shape: (1, num_nodes, 32)

# Run Transformer on GATNet output
output = transformer_model(gat_output)
print("Output shape:", output.shape)  # Expected shape: (1, num_nodes, 2)


# %% [markdown]
# *Load and Preprocess the Dataset*

# %%
'''
import torch
import torch.nn as nn

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads),
            num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.positional_encoding = self.generate_positional_encoding(hidden_dim, 5000)  # Increased max_len


    def generate_positional_encoding(self, d_model, max_len):
        """
        Generate positional encoding for Transformer.

        Args:
            d_model: Dimension of the model.
            max_len: Maximum sequence length.

        Returns:
            Positional encoding tensor of shape (max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        return pe

    def forward(self, x):
        """
        Forward pass of the DecoderOnlyTransformer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        # Reshape before linear layer
        x = x.reshape(-1, self.embedding.in_features)

        x = self.embedding(x) + self.positional_encoding[:, :x.shape[0], :]

        # Reshape for TransformerDecoder
        batch_size = x.shape[0] // num_nodes  # Assuming num_nodes is a global variable
        x = x.view(num_nodes, batch_size, -1)  # (sequence_length, batch_size, embedding_dim)

        # TransformerDecoder expects (sequence_length, batch_size, embedding_dim)
        x = self.transformer(x)

        # Reshape back to (batch_size * num_nodes, output_dim)
        x = x.view(batch_size * num_nodes, -1)
        x = self.fc_out(x)
        return x
'''

# %%
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data_path = path
data = pd.read_csv(path)

# Take a random sample of 3000 rows
sampled_data = data.sample(n=3000, random_state=42)  # random_state for reproducibility

sampled_data['Next X'] = sampled_data.groupby('Vehicle ID')['Local X'].shift(-1)
sampled_data['Next Y'] = sampled_data.groupby('Vehicle ID')['Local Y'].shift(-1)
sampled_data = sampled_data.dropna(subset=['Next X', 'Next Y'])

# Preprocess data
input_columns = ['Local X', 'Local Y', 'Vehicle Velocity', 'Vehicle Accl']
target_columns = ['Next X','Next Y']

all_columns = input_columns + target_columns

scaler = MinMaxScaler()
sampled_data[all_columns] = scaler.fit_transform(sampled_data[all_columns])
'''
sampled_data[input_columns] = scaler.fit_transform(sampled_data[input_columns])
sampled_data[target_columns] = scaler.transform(sampled_data[target_columns])
'''

# Convert to torch tensors
X = torch.tensor(sampled_data[input_columns].values, dtype=torch.float32) #input
y = torch.tensor(sampled_data[target_columns].values, dtype=torch.float32) #output

# Reshape data to fit (samples, sequence length, input/output dim)
seq_len = 10
num_samples = X.shape[0] // seq_len
X = X[:num_samples * seq_len].view(num_samples, seq_len, -1)  # Shape: (samples, seq_len, input_dim)
y = y[:num_samples * seq_len].view(num_samples, seq_len, -1)  # Shape: (samples, seq_len, output_dim)

# Create DataLoader
batch_size = 8
train_data = TensorDataset(X, y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# %%
# Assuming X contains normalized inputs and y contains normalized targets

# Input data statistics
print("Input Data Statistics:")
print("Min:", X.min().item(), "Max:", X.max().item())
print("Mean:", X.mean().item(), "Std Dev:", X.std().item())

# Target data statistics
print("\nTarget Data Statistics:")
print("Min:", y.min().item(), "Max:", y.max().item())
print("Mean:", y.mean().item(), "Std Dev:", y.std().item())


# %% [markdown]
# *split the data form training,testing and validation*

# %%
'''
# Define the split sizes (e.g., 60% train, 20% validation, 20% test)
train_size = int(0.6 * len(train_data))
val_size = int(0.2 * len(train_data))
test_size = len(train_data) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(train_data, [train_size, val_size, test_size])

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
'''

# %% [markdown]
# *Training Loop*

# %%
for data_list in train_loader:
    print([type(item) for item in data_list])  # Check types in data_list
    break  # Check the first batch only

# %%
from torch_geometric.data import Data
num_edges=150
# Example tensors for a single graph sample
x = torch.rand((num_nodes, input_dim))  # Node features tensor
edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge indices tensor
y = torch.rand((num_nodes, output_dim))  # Target values tensor

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y)


# %%
'''
# Assume `data_list` contains multiple `Data` objects, one for each graph
data_list = [Data(x=torch.rand((num_nodes, input_dim)),
                  edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                  y=torch.rand((num_nodes, output_dim))) for _ in range(num_samples)]
'''

# %%
'''
from torch_geometric.data import DataLoader

# Create a DataLoader with batching enabled

train_size = int(0.6 * len(train_data))
val_size = int(0.2 * len(train_data))
test_size = len(train_data) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(train_data, [train_size, val_size, test_size])

# Create DataLoaders for each set
train_loader = DataLoader(data_list, batch_size=32, shuffle=True)
val_loader = DataLoader(data_list, batch_size=32, shuffle=False)
'''

# %%
from torch_geometric.data import Data, DataLoader,Batch

# Example data creation
num_nodes = 32
input_dim = 4
output_dim = 2
num_edges =150

# Generate some example graphs
data_list = [
    Data(
        x=torch.rand((num_nodes, input_dim)),           # Node features
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),  # Edge indices
        y=torch.rand((num_nodes, output_dim))            # Target values
    )
    for _ in range(100)  # Assume 100 samples
]

# Use this list of Data objects to create DataLoaders
train_loader = DataLoader(data_list, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(data_list, batch_size=32, shuffle=False, drop_last=True)

'''
train_loader = DataLoader(data_list, batch_size=32, shuffle=True)
val_loader = DataLoader(data_list, batch_size=32, shuffle=False)
'''


# %% [markdown]
# '*DEBUGGING*

# %%
'''
for data_list in train_loader:
    # Verify that `data_list` is a list of `Data` objects
    print(f"Type of data_list: {type(data_list)}")
    print(f"Type of first item in data_list: {type(data_list[0])}")

    # Try to combine into a Batch and check the result
    try:
        data = Batch.from_data_list(data_list).to(device)
        print(f"Batched data type: {type(data)}")
    except Exception as e:
        print(f"Error during batching: {e}")
'''

# %%
'''
for data_list in train_loader:
    # Check the type of the batch and its contents
    print(f"Batch type: {type(data_list)}")          # Should print <class 'list'>
    print(f"First item type in batch: {type(data_list[0])}")  # Should print <class 'torch_geometric.data.data.Data'>

    # If any unexpected types appear, this is where they might be introduced.
    break  # Check only the first batch to reduce output
'''

# %%
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for data in train_loader:  # `data` is already a batched Data object
    data = data.to(device)  # Move the entire batch to the device

    # Run the inputs through GATNet
    gat_output = gat_model(data)  # `data` contains `x`, `edge_index`, etc.

'''



# %% [markdown]
# *TRAINING LOOP*

# %%
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch

# Define models, optimizer, etc.
optimizer = optim.Adam(list(gat_model.parameters()) + list(transformer_model.parameters()), lr=0.0001)

# Define the loss function
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
for data in train_loader:  # `data` is already a batched Data object
    data = data.to(device)  # Move the batch to the device

    # Step 1: Run the inputs through GATNet
    gat_output = gat_model(data)  # Expects `data.x` and `data.edge_index`

    # Step 2: Reshape or process `gat_output` as needed
    # Assuming you need to reshape it for the Transformer input
    batch_size = data.num_graphs
    sequence_length = gat_output.size(0) // batch_size
    gat_output = gat_output.view(batch_size, sequence_length, -1)

    # Step 3: Run through Transformer
    outputs = transformer_model(gat_output)

    # Step 4: Compute loss
    targets = data.y.to(device)  # Assuming `data.y` is the target
    targets = targets.view(outputs.shape)  # Reshape targets to match outputs if needed
    loss = criterion(outputs, targets)

    # Backpropagation
    loss.backward()
    optimizer.step()
    '''


# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

# Define optimizer and loss function
optimizer = optim.Adam(list(gat_model.parameters()) + list(transformer_model.parameters()), lr=0.0001)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Send models to the device
gat_model.to(device)
transformer_model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Set models to training mode
    print(f"Epoch [{epoch + 1}/{num_epochs}]")   # Should output <class 'torch_geometric.data.Data'>

    #Combine list of Data objects into a single Batch object
    #data = Batch.from_data_list(data_list).to(device)
    #data = data_list.to(device)
    gat_model.train()
    transformer_model.train()

    running_loss = 0.0
    for data in train_loader:  # Iterate through batches (DataBatch objects) directly
        # Move data to the device
        data = data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Step 1: Run the inputs through GATNet
        gat_output = gat_model(data)  # Expects `data.x` and `data.edge_index` in GATNet

        # Check the shape of gat_output
        print(f"GAT output shape: {gat_output.shape}")

        # Step 2: Reshape GAT output if needed
        # Ensure gat_output has shape [batch_size, seq_len (num_nodes), embed_dim]
        if len(gat_output.shape) == 2:  # If it’s [num_nodes * batch_size, embed_dim]
            batch_size = data.num_graphs # Get number of graphs in the batch #changed to data.num_graphs
            num_nodes = data.x.shape[0] // batch_size  # Compute nodes per graph
            gat_output = gat_output.view(batch_size, num_nodes, -1)
        # Check the reshaped shape
        print(f"Reshaped GAT output shape: {gat_output.shape}")

        # Step 3: Run the GAT output through Transformer directly
        outputs = transformer_model(gat_output)

        # Step 4: Compute loss between model output and targets
        targets = data.y.to(device)  # Assuming targets are stored as `data.y`

        # Check shapes of outputs and targets
        print(f"Outputs shape: {outputs.shape}")
        print(f"Targets shape: {targets.shape}")

        targets = targets.view(outputs.shape)  # Reshape targets to match outputs shape

        # Ensure targets have the same shape as outputs
        if targets.shape != outputs.shape:
            raise ValueError(f"Targets shape {targets.shape} does not match Outputs shape {outputs.shape}")

        loss = criterion(outputs, targets)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    # Print the average loss per epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')
    # Validation phase
    gat_model.eval()
    transformer_model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradients for validation
        for data_list in val_loader:  # Expect each batch to be a list of Data objects
            # Combine list of Data objects into a single Batch object
            #data = Batch.from_data_list(data_list).to(device)
            data = data_list.to(device)

            # Run inputs through GAT and then Transformer
            gat_output = gat_model(data)

            # Reshape if needed
            if len(gat_output.shape) == 2:
                batch_size = data.num_graphs
                num_nodes = data.x.shape[0] // batch_size
                gat_output = gat_output.view(batch_size, num_nodes, -1)

            # Get predictions from Transformer
            targets = data.y.to(device)  # Assuming targets are stored as `data.y`
            outputs = transformer_model(gat_output)


# %%
'''
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()  # Loss function for regression

# Training loop
num_epochs = 50  # Adjust number of epochs based on model performance
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)  # Average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed during validation
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
'''

# %% [markdown]
# *Testing*

# %%
import torch
import numpy as np
import torch.nn as nn

# Set the models to evaluation mode
gat_model.eval()
transformer_model.eval()

# Placeholder to accumulate test loss
test_loss = 0.0
loss_fn = nn.MSELoss()  # Define the loss function (MSE for regression tasks)

# Initialize lists to store predicted and actual values
predicted_values = []
actual_values = []

# Evaluation loop
with torch.no_grad():  # Disable gradient computation for faster testing
    for data in val_loader:  # Assuming `val_loader` yields `Data` objects from PyTorch Geometric
        data = data.to(device)

        # Forward pass: Get predictions through GATNet and Transformer
        gat_output = gat_model(data)  # Process input through GATNet
        batch_size = data.num_graphs
        seq_len = gat_output.size(0) // batch_size
        gat_output = gat_output.view(batch_size, seq_len, -1)  # Reshape as needed
        outputs = transformer_model(gat_output)  # Get predictions from Transformer

        # Get targets
        targets = data.y.to(device)
        targets = targets.view(outputs.shape)  # Reshape targets to match outputs

        # Compute test loss
        loss = loss_fn(outputs, targets)
        test_loss += loss.item()

        # Store predictions and actual targets (convert tensors to numpy arrays)
        predicted_values.append(outputs.cpu().numpy())  # Move to CPU and convert to numpy
        actual_values.append(targets.cpu().numpy())  # Move to CPU and convert to numpy

# Calculate the average test loss
avg_test_loss = test_loss / len(val_loader)
print(f"Test Loss (MSE): {avg_test_loss:.4f}")

# Concatenate all predicted and actual values for further metrics
predicted_values = np.concatenate(predicted_values, axis=0)  # Shape: (num_samples, seq_len, output_dim)
actual_values = np.concatenate(actual_values, axis=0)  # Shape: (num_samples, seq_len, output_dim)

# Calculate ADE (Average Displacement Error)
displacement_errors = np.linalg.norm(predicted_values - actual_values, axis=-1)  # Euclidean distance per sample
ade = np.mean(displacement_errors)
print(f"Average Displacement Error (ADE): {ade:.4f}")

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((predicted_values - actual_values) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Display the first 5 predictions vs actuals for quick inspection
print("\nPredicted vs Actual (First 5 examples):")
for i in range(5):
    print(f"Example {i+1}:")
    print("Predicted:", predicted_values[i])
    print("Actual:", actual_values[i])
    print()



