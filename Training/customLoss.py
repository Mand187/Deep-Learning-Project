import torch
import torch.nn as nn
import sys
import os

# * MODULE IMPORTS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import PADDING_TOKEN

# Lane-aware loss function
def lane_loss(predicted, lane_positions, mask=None):
    distance_to_lane = 0
    valid_count = 0
    for i, pred_point in enumerate(predicted):
        if mask is not None and not mask[i]:
            continue
        min_distance = float('inf')
        for lane in lane_positions:
            A, B, C = lane  # Line equation coefficients
            x, y = pred_point
            # Calculate the perpendicular distance to the line Ax + By + C = 0
            distance = abs(A * x + B * y + C) / torch.sqrt(A ** 2 + B ** 2)
            min_distance = min(min_distance, distance)
        distance_to_lane += min_distance
        valid_count += 1
    
    return distance_to_lane / valid_count if valid_count > 0 else 0

class ADELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ADELoss, self).__init__()
        self.reduction = reduction
    def forward(self, predictions, targets):
        """
        Calculate the Average Displacement Error (ADE), ignoring padded values.
        predictions: (batch_size, sequence_length, num_features) e.g., (B, T, 2)
        targets: (batch_size, sequence_length, num_features) e.g., (B, T, 2)
        """
        # Check if predictions and targets have the same shape
        if predictions.shape != targets.shape:
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            raise ValueError("Predictions and targets must have the same shape.")

        # Calculate Euclidean distance (L2 norm) across the last dimension (x,y coordinates)
        # euclidean_distance will have shape (batch_size, sequence_length)
        euclidean_distance = torch.linalg.vector_norm(predictions - targets, dim=-1)

        # Create a mask for valid (non-padded) target values.
        # A target point (e.g., x,y coordinates) is considered valid if all its features are not -1.
        # valid_mask will be a boolean tensor of shape (batch_size, sequence_length).
        # It's True for non-padded time steps, False for padded ones.
        valid_mask = (targets != PADDING_TOKEN).all(dim=-1)
        
        # Apply the mask to the distances.
        # For padded entries (where mask is False), their contribution to ADE will be 0.
        # masked_euclidean_distance will have shape (batch_size, sequence_length)
        masked_euclidean_distance = euclidean_distance * valid_mask.float()
        
        # Sum the distances over the sequence length for each sample.
        # sum_distances_per_sample will have shape (batch_size)
        sum_distances_per_sample = masked_euclidean_distance.sum(dim=1)
        
        # Count the number of valid (non-padded) time steps for each sample.
        # num_valid_timesteps_per_sample will have shape (batch_size)
        num_valid_timesteps_per_sample = valid_mask.sum(dim=1).float()
        
        # Calculate ADE for each sample.
        # If num_valid_timesteps_per_sample is 0 for a sample, ade_per_sample for that sample will be 0.
        # clamp(min=1.0) prevents division by zero, resulting in 0.0 / 1.0 = 0.0 for fully padded sequences.
        ade_per_sample = sum_distances_per_sample / num_valid_timesteps_per_sample.clamp(min=1.0)
        
        # Apply reduction
        if self.reduction == 'mean':
            # Calculate the mean ADE over samples that have at least one valid timestep.
            # If all samples are fully padded, num_valid_samples will be 0, and the mean will be 0.
            num_valid_samples = (num_valid_timesteps_per_sample > 0).sum().float().clamp(min=1.0)
            return ade_per_sample.sum() / num_valid_samples
        elif self.reduction == 'sum':
            return ade_per_sample.sum()
        else:  # 'none'
            return ade_per_sample
    


class FDELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FDELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        
        if predictions.shape != targets.shape:
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            raise ValueError("Predictions and targets must have the same shape.")
        
        # predictions: (batch_size, sequence_length, num_features) e.g., (B, T, 2)
        # targets: (batch_size, sequence_length, num_features) e.g., (B, T, 2)

        # Get the final predicted positions and target positions
        # These will have shape (batch_size, num_features)
        final_predictions = predictions[:, -1]
        final_targets = targets[:, -1]

        # Calculate the Euclidean distance (L2 norm) between final predicted and target positions
        # This will result in a tensor of shape (batch_size)
        euclidean_distance_final_points = torch.linalg.vector_norm(final_predictions - final_targets, dim=-1)

        # Create a mask for valid (non-padded) final target positions.
        # A final target point (e.g., x,y coordinates) is considered valid if all its features are not -1.
        # valid_mask_final_targets will be a boolean tensor of shape (batch_size).
        # It's True for samples where the final target is not padded, False otherwise.
        valid_mask_final_targets = (final_targets != PADDING_TOKEN).all(dim=-1)
        
        # Apply the mask to the distances.
        # For padded entries (where mask is False), their contribution to FDE will be 0.
        # fde_per_sample will have shape (batch_size)
        fde_per_sample = euclidean_distance_final_points * valid_mask_final_targets.float()

        # Apply reduction
        if self.reduction == 'mean':
            # Sum of FDE for valid samples divided by the number of valid samples.
            num_valid_samples = valid_mask_final_targets.sum()
            # If num_valid_samples is 0, fde_per_sample.sum() will also be 0.
            # clamp(min=1.0) prevents division by zero, resulting in 0.0 / 1.0 = 0.0.
            return fde_per_sample.sum() / num_valid_samples.clamp(min=1.0)
        elif self.reduction == 'sum':
            return fde_per_sample.sum()
        else:  # 'none'
            return fde_per_sample

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMSELoss, self).__init__()
        self.mse_loss = PaddedMSELoss(reduction=reduction)
        
    def forward(self, predictions, targets):
        # Compute MSE loss and take the square root
        mse = self.mse_loss(predictions, targets)
        return torch.sqrt(mse)
        
class PaddedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)
    def forward(self, predictions, targets):
        """
        Calculate the Mean Squared Error (MSE) loss, ignoring padded values.
        
        predictions.shape: [batch_size, prediction_length, num_ids, 2]
        """
        
        # Check if predictions and targets have the same shape
        if predictions.shape != targets.shape:
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            raise ValueError("Predictions and targets must have the same shape.")
        
        # Create a mask for non-padded values
        mask = (targets != PADDING_TOKEN).float()
        
        # Calculate MSE for non-padded values
        mse = self.mse(predictions, targets) * mask
        mse = mse.sum(dim=-1)  # Sum over the last dimension (x,y coordinates)
        mse = mse.sum(dim=1)

        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:  # 'none'
            return mse
