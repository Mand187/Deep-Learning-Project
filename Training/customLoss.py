import torch
import torch.nn as nn

# Lane-aware loss function
def lane_loss(predicted, lane_positions):
    distance_to_lane = 0
    for pred_point in predicted:
        min_distance = float('inf')
        for lane in lane_positions:
            A, B, C = lane  # Line equation coefficients
            x, y = pred_point
            # Calculate the perpendicular distance to the line Ax + By + C = 0
            distance = abs(A * x + B * y + C) / torch.sqrt(A ** 2 + B ** 2)
            min_distance = min(min_distance, distance)
        distance_to_lane += min_distance
    
    return distance_to_lane / len(predicted)

class PaddedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(PaddedMSELoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
    def forward(self, predictions, targets):
        """
        Calculate the Mean Squared Error (MSE) loss, ignoring padded values.
        """
        # Check if predictions and targets have the same shape
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")
        
        # Create a mask for non-padded values
        mask = (targets != -1).float()
        
        # Calculate MSE for non-padded values
        mse = self.mse(predictions, targets) * mask
        mse = mse.sum(dim=-1)  # Sum over the last dimension (x,y coordinates)
        mse = mse.sum(dim=1)

class ADELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ADELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Calculate the Average Displacement Error.
        """
        # Calculate Euclidean distance (L2 norm) across the last dimension (x,y coordinates)
        
        mask = (targets != -1).float()
        
        euclidean_distance = torch.norm(predictions - targets, p=2, dim=-1)
        # Apply mask to ignore padded values
        euclidean_distance = euclidean_distance * mask
        # Average over sequence length
        ade = euclidean_distance.mean(dim=1)  # average over sequence length (dim=1)
        
        
        # Apply reduction
        if self.reduction == 'mean':
            return ade.mean()
        elif self.reduction == 'sum':
            return ade.sum()
        else:  # 'none'
            return ade


class FDELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FDELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        # Get the final positions (last timestep)
        final_predictions = predictions[:, -1]  # (batch_size, num_agents, 2) or (batch_size, 2)
        final_targets = targets[:, -1]  # (batch_size, num_agents, 2) or (batch_size, 2)
        
        # Calculate Euclidean distance (L2 norm)
        fde = torch.norm(final_predictions - final_targets, p=2, dim=-1)
        
        mask = (targets != -1).float()
        # Apply mask to ignore padded values
        fde = fde * mask[:, -1]
        
        # Apply reduction
        if self.reduction == 'mean':
            return fde.mean()
        elif self.reduction == 'sum':
            return fde.sum()
        else:  # 'none'
            return fde


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets):
        # Calculate squared error for each element
        squared_error = self.mse(predictions, targets)
        
        # For trajectory data, we often want to calculate RMSE across coordinate dimensions
        if predictions.dim() >= 3:
            # Sum over coordinate dimensions (last dim)
            squared_error = squared_error.sum(dim=-1)
            
            # Mean over sequence length
            if predictions.dim() >= 4:  # If we have multiple agents
                # (batch, seq_len, num_agents) -> (batch, num_agents)
                mse = squared_error.mean(dim=1)
            else:  # No agent dimension
                # (batch, seq_len) -> (batch)
                mse = squared_error.mean(dim=1)
        else:
            # For non-trajectory data, handle differently
            if self.reduction == 'none':
                return torch.sqrt(squared_error)
            mse = squared_error.mean(dim=tuple(range(1, squared_error.dim())))
        
        # Take square root
        rmse = torch.sqrt(mse)
        
        # Apply reduction
        if self.reduction == 'mean':
            return rmse.mean()
        elif self.reduction == 'sum':
            return rmse.sum()
        else:  # 'none'
            return rmse