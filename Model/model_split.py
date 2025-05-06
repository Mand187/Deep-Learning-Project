import torch
import torch.nn as nn
import math
from torchprofile import profile_macs

from config import *

debug = False
class FrameTransformer(nn.Module):
    def __init__(self, input_feature_size=NUM_INPUT_FEATURES, num_ids=None, sequence_length=SEQUENCE_LENGTH, 
                 prediction_length=PREDICTION_LENGTH, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, 
                 dropout_rate=DROPOUT_RATE):
        super().__init__()
        if debug:
            print(f"[FrameTransformer __init__] hidden_size={hidden_size}, num_ids={num_ids}, num_heads={num_heads}") # Debug print
        if num_ids is None:
            raise ValueError("num_ids must be provided to FrameTransformer")
        frame_attention_embed_dim = hidden_size * num_ids
        if debug:
            print(f"[FrameTransformer __init__] frame_attention embed_dim calculated as: {frame_attention_embed_dim}") # Debug print

        self.prediction_length = prediction_length
        
        # Sinusoidal positional encoding for frames - creates a unique positional encoding for each frame in the sequence
        # that helps the model understand the order/temporal relationships between frames. Uses alternating sin/cos waves
        # of different frequencies to encode position information.
        positions = torch.arange(sequence_length).unsqueeze(1)
        feature_frequency = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        
        positional_encoder = torch.zeros(1, sequence_length, hidden_size)
        
        positional_encoder[0, :, 0::2] = torch.sin(positions * feature_frequency)
        positional_encoder[0, :, 1::2] = torch.cos(positions * feature_frequency)
        self.register_buffer('frame_pos_encoder', positional_encoder)  # register as buffer so it moves with model to GPU
        
        # Input feature projection
        self.input_proj = nn.Linear(input_feature_size, hidden_size)
        
        # Multihead attention across IDs in a frame
        self.id_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Multihead attention across frames
        self.frame_attention = nn.MultiheadAttention(
            embed_dim=frame_attention_embed_dim, # Use calculated dim
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Temporal convolution to map sequence length to prediction length
        # Stack of 1D convolutional layers over the temporal dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=sequence_length // 2, out_channels=prediction_length, kernel_size=1)
        )

        # Output feature projection
        self.output_proj = nn.Linear(hidden_size, 2)
        
        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(frame_attention_embed_dim) # Use calculated dim
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        x.shape: [batch_size, sequence_length, num_ids, input_feature_size]
        
        return.shape: [batch_size, prediction_length, num_ids, 2 (X, Y)]
        """
        batch_size, seq_len, num_ids_from_input, input_feat_dim = x.shape
        # Get hidden_size from the layer to ensure consistency
        hidden_size_from_layer = self.input_proj.out_features
        if debug:
            print(f"[FrameTransformer forward] Input x shape: {x.shape}") # Debug print
            print(f"[FrameTransformer forward] num_ids_from_input={num_ids_from_input}, hidden_size_from_layer={hidden_size_from_layer}") # Debug print
        
        # Project input features to HIDDEN_SIZE
        x = self.input_proj(x)  # [batch, seq, num_ids, hidden_size]
        
        # Add frame positional encoding
        # Ensure positional encoder matches hidden_size_from_layer if they could differ
        if x.size(-1) != self.frame_pos_encoder.size(-1):
             print(f"WARNING: hidden_size mismatch between input_proj ({x.size(-1)}) and pos_encoder ({self.frame_pos_encoder.size(-1)})", flush=True)
             # Handle mismatch or raise error if necessary - For now, just warn
        # Add positional encoding - Ensure broadcasting is correct
        x = x + self.frame_pos_encoder.unsqueeze(2).expand(-1, -1, num_ids_from_input, -1)
        
        # Reshape for ID attention (treat each frame independently)
        x_id = x.reshape(batch_size * seq_len, num_ids_from_input, -1) # [batch * seq, num_ids, hidden_size]
        
        # Self attention across IDs with residual
        id_attn_out, _ = self.id_attention(x_id, x_id, x_id)
        id_attn_out = self.dropout(id_attn_out)
        id_attn_out = self.norm1(x_id + id_attn_out)
        
        # Reshape back for frame attention
        x_frame = id_attn_out.reshape(batch_size, seq_len, num_ids_from_input, -1)
        # Calculate expected dimension explicitly
        expected_frame_dim = num_ids_from_input * hidden_size_from_layer
        x_frame = x_frame.reshape(batch_size, seq_len, expected_frame_dim) # Use explicit calculation
        if debug:
            print(f"[FrameTransformer forward] x_frame shape before frame_attention: {x_frame.shape}", flush=True) # Debug print
            print(f"[FrameTransformer forward] Expected frame_attention embed_dim: {self.frame_attention.embed_dim}", flush=True) # Debug print
        
        # Self attention across frames with residual
        frame_attn_out, _ = self.frame_attention(x_frame, x_frame, x_frame)
        frame_attn_out = self.dropout(frame_attn_out)
        frame_attn_out = self.norm2(x_frame + frame_attn_out)
        
        # Reshape for temporal convolution
        # Use hidden_size_from_layer for consistency
        output = frame_attn_out.reshape(batch_size, seq_len, num_ids_from_input, hidden_size_from_layer)
        output = output.permute(0, 2, 1, 3) # [batch, num_ids, seq_len, hidden_size]
        output = output.reshape(batch_size * num_ids_from_input, seq_len, hidden_size_from_layer) # [batch*num_ids, seq_len, hidden_size]
        
        # Apply temporal convolution
        # Input shape: [batch*num_ids, 100, 64] (N, C_in=100, L_in=64)
        # Conv1d(in_channels=100, ...) expects this shape
        output = self.temporal_conv(output) # Output shape: [batch*num_ids, 30, 64] (N, C_out=30, L_out=64)
        
        # Reshape back and project to output feature size
        # Need shape: [batch, pred_len, num_ids, 2]
        # Current shape: [batch*num_ids, 30, 64] (N, C_out, L_out)
        # Reshape to [batch, num_ids, 30, 64]
        output = output.reshape(batch_size, num_ids_from_input, self.prediction_length, hidden_size_from_layer)
        output = output.permute(0, 2, 1, 3) # [batch, pred_len, num_ids, hidden_size]
        output = self.output_proj(output)  # [batch, pred_len, num_ids, 2]
        
        return output


def print_model_info(model, sample_X):
    """Print model parameter count and computational complexity"""
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total Num Params in loaded model: {total_params:,}")
    
    # Calculate MACs (Multiply-Accumulate Operations)
    macs = profile_macs(model, (sample_X, ))
    print(f"Computational complexity: {macs:,} MACs")
    print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")