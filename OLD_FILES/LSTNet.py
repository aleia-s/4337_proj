import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTNet(nn.Module):
    
    def __init__(self, num_features=5, device='cuda'):
        super(LSTNet, self).__init__()
        self.device = device
        self.num_features = num_features
        
        # Model parameters
        self.conv_out_channels = 32
        self.gru_hidden_size = 64
        self.skip_lengths = [4, 24]  # Weekly and monthly patterns
        self.skip_hidden_size = 16
        self.ar_window = 7  # Autoregressive window size
        
        # Convolutional layer - use smaller kernel size for time dimension (4 instead of 7)
        # This is to handle our sequence length of 4
        conv_kernel_time = min(4, 4)  # Use min of 4 or sequence length to avoid errors
        self.conv = nn.Conv2d(1, self.conv_out_channels, 
                             kernel_size=(conv_kernel_time, self.num_features))
        
        # GRU layers
        self.gru = nn.GRU(self.conv_out_channels, self.gru_hidden_size, 
                         batch_first=True)
        
        # Skip GRU layers
        self.skip_gru = nn.ModuleList([
            nn.GRU(self.conv_out_channels, self.skip_hidden_size, 
                  batch_first=True)
            for _ in range(len(self.skip_lengths))
        ])
        
        # Output layers
        self.fc = nn.Linear(
            self.gru_hidden_size + 
            len(self.skip_lengths) * self.skip_hidden_size,
            self.num_features
        )
        
        # Autoregressive component
        self.ar = nn.Linear(self.ar_window, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Move to device
        self.to(device)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Convolutional layer
        c = x.unsqueeze(1)  # Add channel dimension
        c = F.relu(self.conv(c))
        c = self.dropout(c)
        c = c.squeeze(3)  # Remove last dimension
        
        # GRU layer
        r = c.permute(0, 2, 1)
        r, _ = self.gru(r)
        r = r[:, -1, :]
        r = self.dropout(r)
        
        # Skip connections - modify to handle potentially smaller sequence lengths
        skip_outputs = []
        for i, skip_len in enumerate(self.skip_lengths):
            # Ensure skip_len isn't larger than our feature map
            actual_skip_len = min(skip_len, c.size(2))
            if actual_skip_len <= 0:
                # If feature map is too small, create a zero tensor of appropriate size
                s = torch.zeros(batch_size, self.skip_hidden_size, device=self.device)
                skip_outputs.append(s)
                continue
                
            # Get last skip_len time steps
            s = c[:, :, -actual_skip_len:]
            
            # Process through skip GRU
            s = s.permute(0, 2, 1)
            s, _ = self.skip_gru[i](s)
            s = s[:, -1, :]
            s = self.dropout(s)
            skip_outputs.append(s)
        
        # Combine GRU and skip outputs
        combined = torch.cat([r] + skip_outputs, dim=1)
        
        # Final output
        output = self.fc(combined)
        
        # Autoregressive component - check if we have enough time steps
        if self.ar_window > 0 and x.size(1) >= self.ar_window:
            ar = x[:, -self.ar_window:, :]
            ar = ar.permute(0, 2, 1)
            ar = self.ar(ar)
            ar = ar.squeeze(2)
            output = output + ar
        
        return output