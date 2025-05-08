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
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Calculate the Average Displacement Error (ADE), ignoring padded values.
        
        predictions & targets.shape: [batch_size, prediction_length, num_ids, 2]
        
        return.shape: 1 if reduction = 'mean' or 'sum' else predictions & targets.shape
        """
        # Check if predictions and targets have the same shape
        if predictions.shape != targets.shape:
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            raise ValueError("Predictions and targets must have the same shape.")

        # Calculate Euclidean distance (MSE norm) across the last dimension (x,y coordinates)
        # IE: √[(pred_x - tgt_x)^2 + (pred_y - tgt_y)^2]
        euclidean_distance = torch.linalg.vector_norm(predictions - targets, dim=-1) # [batch_size, prediction_length, num_ids]

        # Create a mask for valid (non-padded) target values.
        # A target point (e.g., x,y coordinates) is considered valid if all its features are not -1.
        # It's True for non-padded time steps, False for padded ones.
        valid_mask = (targets != PADDING_TOKEN).all(dim=-1) # [batch_size, prediction_length, num_ids]
        
        # Apply the mask to the distances.
        # For padded entries (where mask is False), their contribution to ADE will be 0.
        masked_euclidean_distance = euclidean_distance * valid_mask.float() # [batch_size, prediction_length, num_ids]
        
        # Sum the distances over the sequence length for each sample.
        sum_distances_per_sample = masked_euclidean_distance.sum(dim=1) # [batch_size, num_ids]
        
        # Count the number of valid (non-padded) time steps for each sample.
        num_valid_timesteps_per_sample = valid_mask.sum(dim=1).float() # [batch_size, num_ids]
        
        
        # Calculate ADE for each sample.
        # If num_valid_timesteps_per_sample is 0 for a sample, ade_per_sample for that sample will be 0.
        # clamp(min=1.0) prevents division by zero, resulting in 0.0 / 1.0 = 0.0 for fully padded sequences.
        ade_per_sample = sum_distances_per_sample / num_valid_timesteps_per_sample.clamp(min=1.0) # [batch_size, num_ids]
        
        # Apply reduction
        if self.reduction == 'mean':
            # Calculate the mean ADE over samples that have at least one valid timestep.
            # If all samples are fully padded, num_valid_samples will be 0, and the mean will be 0.
            num_valid_samples = (num_valid_timesteps_per_sample > 0).sum().float()
            return ade_per_sample.sum() / num_valid_samples.clamp(min=1.0)
        
        elif self.reduction == 'sum':
            return ade_per_sample.sum()
        
        else:  # 'none'
            return ade_per_sample
    


class FDELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Calculate Final Displacement Error (FDE) loss, ignoring padded values.
        
        predictions & targets.shape: [batch_size, prediction_length, num_ids, 2]
        
        return.shape: 1 if reduction = 'mean' or 'sum' else predictions & targets.shape
        """
        
        if predictions.shape != targets.shape:
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            raise ValueError("Predictions and targets must have the same shape.")
        
        # Get the final predicted positions and target positions
        final_predictions = predictions[:, -1] # [batch_size, num_ids, 2]
        final_targets = targets[:, -1] # [batch_size, num_ids, 2]

        # Calculate the Euclidean distance (L2 norm) between final predicted and target positions
        euclidean_distance_final_points = torch.linalg.vector_norm(final_predictions - final_targets, dim=-1) # [batch_size, num_ids]

        # Create a mask for valid (non-padded) final target positions.
        # A final target point (e.g., x,y coordinates) is considered valid if all its features are not -1.
        # valid_mask_final_targets will be a boolean tensor of shape (batch_size).
        # It's True for samples where the final target is not padded, False otherwise.
        valid_mask_final_targets = (final_targets != PADDING_TOKEN).all(dim=-1) # [batch_size, num_ids]
        
        # Apply the mask to the distances.
        # For padded entries (where mask is False), their contribution to FDE will be 0.
        # fde_per_sample will have shape (batch_size)
        fde_per_sample = euclidean_distance_final_points * valid_mask_final_targets.float() # [batch_size, num_ids]

        # Apply reduction
        if self.reduction == 'mean':
            # Sum of FDE for valid samples divided by the number of valid samples.
            # If num_valid_samples is 0, fde_per_sample.sum() will also be 0.
            # clamp(min=1.0) prevents division by zero, resulting in 0.0 / 1.0 = 0.0.
            num_valid_samples = valid_mask_final_targets.sum()
            return fde_per_sample.sum() / num_valid_samples.clamp(min=1.0)
        
        elif self.reduction == 'sum':
            return fde_per_sample.sum()
        
        else:  # 'none'
            return fde_per_sample

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.mse_loss = PaddedMSELoss(reduction=reduction)
        self.eps = eps
        
    def forward(self, predictions, targets):
        """
        Calculate Square root of the Mean Squared Error (MSE) loss, ignoring padded values.
        
        predictions & targets.shape: [batch_size, prediction_length, num_ids, 2]
        
        return.shape: 1 if reduction = 'mean' or 'sum' else predictions & targets.shape
        """
        
        # Compute MSE loss and take the square root
        mse = self.mse_loss(predictions, targets)
        # Add epsilon for numerical stability before sqrt
        return torch.sqrt(mse + self.eps)
        
class PaddedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)
        
    def forward(self, predictions, targets):
        """
        Calculate the Mean Squared Error (MSE) loss, ignoring padded values.
        
        predictions & targets.shape: [batch_size, prediction_length, num_ids, 2]
        
        return.shape: 1 if reduction = 'mean' or 'sum' else predictions & targets.shape
        """
        
        # Check if predictions and targets have the same shape
        if predictions.shape != targets.shape:
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            raise ValueError("Predictions and targets must have the same shape.")
        
        # Create a mask for non-padded values
        mask = (targets != PADDING_TOKEN).all(dim=-1, keepdim=True).float() # Ensure mask is broadcastable
        
        # Calculate element-wise squared error
        squared_error = (predictions - targets)**2
        
        # Apply mask to the squared error
        masked_squared_error = squared_error * mask
        
        if self.reduction == 'mean':
            # Sum masked errors and divide by the count of non-padded elements
            # The number of elements to average over is the sum of the mask (where mask is 1 for valid, 0 for padded)
            # We need to sum over all dimensions of the mask that correspond to the error tensor.
            # Since squared_error is [B, P, N, F], and mask is [B, P, N, 1] (after keepdim=True),
            # the number of valid elements is mask.sum().
            # Ensure we don't divide by zero if all elements are padded.
            num_valid_elements = mask.sum().clamp(min=1.0)
            return masked_squared_error.sum() / num_valid_elements
        elif self.reduction == 'sum':
            return masked_squared_error.sum()
        else: # 'none'
            # If reduction is 'none', we should return the masked squared error per element.
            # However, nn.MSELoss(reduction='none') would return element-wise SE.
            # To be consistent, if the user wants 'none', they likely expect the masked SE.
            return masked_squared_error