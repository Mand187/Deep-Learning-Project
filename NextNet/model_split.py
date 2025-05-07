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
        # self.temporal_conv = nn.Sequential(
        #     nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=sequence_length, out_channels=sequence_length // 2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=sequence_length // 2, out_channels=prediction_length, kernel_size=1)
        # )
        self.temporal_conv = nn.Conv1d(
            in_channels=sequence_length,
            out_channels=prediction_length,
            kernel_size=1
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
        batch_size, seq_len, num_ids, input_feat_dim = x.shape
        
        # Project input features to HIDDEN_SIZE
        x = self.input_proj(x)  # [batch_size, sequence_length, num_ids, HIDDEN_SIZE]
        
        # Add frame positional encoding
        x = x + self.frame_pos_encoder.unsqueeze(2)
        
        # Reshape for ID attention (treat each frame independently)
        x_id = x.reshape(batch_size * seq_len, num_ids, -1) # [batch * seq, num_ids, HIDDEN_SIZE]
        
        # Self attention across IDs with residual
        id_attn_out, _ = self.id_attention(x_id, x_id, x_id)
        id_attn_out = self.dropout(id_attn_out)
        id_attn_out = self.norm1(x_id + id_attn_out)
        
        # Reshape back for frame attention
        x_frame = id_attn_out.reshape(batch_size, seq_len, num_ids, -1)
        x_frame = x_frame.reshape(batch_size, seq_len, -1) # [batch_size, sequence_length, num_ids * HIDDEN_SIZE]
        
        # Self attention across frames with residual
        frame_attn_out, _ = self.frame_attention(x_frame, x_frame, x_frame)
        frame_attn_out = self.dropout(frame_attn_out)
        frame_attn_out = self.norm2(x_frame + frame_attn_out)
        
        # Reshape for temporal convolution
        output = frame_attn_out.reshape(batch_size, seq_len, num_ids, -1) # [batch_size, sequence_length, num_ids, HIDDEN_SIZE]
        output = output.permute(0, 2, 1, 3) # [batch_size, num_ids, sequence_length, HIDDEN_SIZE]
        output = output.reshape(batch_size * num_ids, seq_len, -1) # [batch_size*num_ids, sequence_length, HIDDEN_SIZE]
        
        # Apply temporal convolution
        output = self.temporal_conv(output) # [batch_size*num_ids, prediction_length, HIDDEN_SIZE]
        
        # Reshape back and project to input feature size
        output = output.reshape(batch_size, num_ids, self.prediction_length, -1) # [batch_size, num_ids, prediction_length, HIDDEN_SIZE]
        output = output.permute(0, 2, 1, 3) # [batch_size, prediction_length, num_ids, HIDDEN_SIZE]
        output = self.output_proj(output)  # [batch_size, prediction_length, num_ids, 2]
        
        return output


def print_model_info(model, sample_X):
    """Print model parameter count and computational complexity"""
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total Num Params in loaded model: {total_params:,}")
    
    # Calculate MACs (Multiply-Accumulate Operations)
    macs = profile_macs(model, (sample_X, ))
    print(f"Computational complexity: {macs:,} MACs")
    print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)")