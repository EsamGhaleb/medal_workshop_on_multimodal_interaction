import torch

import torch.nn as nn

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
   def __init__(self, model_dim, dropout=0.1, max_len=5000):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)

      # Create a long enough P matrix
      pe = torch.zeros(max_len, model_dim)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(1)
      self.register_buffer('pe', pe)

   def forward(self, x):
      # x shape is (seq_length, batch_size, model_dim)
      x = x + self.pe[:x.size(0)]
      return self.dropout(x)


class TransformerEncoder(nn.Module):
   def __init__(self, input_dim, model_dim, num_layers, num_heads, dim_feedforward=2048, dropout=0.1):
      super(TransformerEncoder, self).__init__()
      self.input_linear = nn.Linear(input_dim, model_dim)
      encoder_layer = nn.TransformerEncoderLayer(
         d_model=model_dim,
         nhead=num_heads,
         dim_feedforward=dim_feedforward,
         dropout=dropout
      )
      # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
      self.output_linear = nn.Linear(model_dim, model_dim)
      
   def forward(self, src, src_mask=None):
      # src shape: (seq_length, batch_size, input_dim)
      src = self.input_linear(src)
      # Apply positional encoding
      # src shape: (seq_length, batch_size, model_dim)
      # encoded = self.encoder(src, mask=src_mask)
      return self.output_linear(src)
   
def generate_square_subsequent_mask(sz):
   # Generates a square mask for the sequence. The masked positions are filled with -inf.
   mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
   return mask

def main():
   # Example usage of TransformerEncoder.
   
   # Define dimensions and sequence parameters
   seq_length = 10
   batch_size = 32
   input_dim = 16
   model_dim = 32
   num_layers = 2
   num_heads = 4
   
   # Create a random input tensor.
   # Transformer expects input of shape: (seq_length, batch_size, input_dim)
   src = torch.randn(seq_length, batch_size, input_dim)
   
   # Optional: Generate a mask for auto-regressive behavior
   mask = generate_square_subsequent_mask(seq_length)
   
   # Initialize the TransformerEncoder
   transformer_encoder = TransformerEncoder(input_dim, model_dim, num_layers, num_heads)
   
   # Forward pass through the encoder
   output = transformer_encoder(src, src_mask=mask)
   
   print("Output shape:", output.shape)
   
if __name__ == '__main__':
   main()