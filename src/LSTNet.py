import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG, ACTIVE_DATASET

class LSTNet(nn.Module):
    def __init__(self, num_features: int, device: torch.device, output_size=None):
        super().__init__()
        self.device = device
        cfg = MODEL_CONFIG

        self.num_features = num_features
        self.output_size = output_size if output_size is not None else (num_features if ACTIVE_DATASET == "unemployment" else 1)
        self.conv_out_channels = cfg['conv_out_channels']
        self.gru_hidden_size = cfg['gru_hidden_size']
        self.skip_lengths = cfg['skip_lengths']
        self.skip_hidden_size = cfg['skip_hidden_size']
        self.ar_window = cfg['ar_window']

        # 1) Convolution: (batch, 1, seq_len, num_features) -> ...
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_out_channels,
            kernel_size=(7, self.num_features)
        )

        # 2) Main GRU on conv output
        self.gru = nn.GRU(
            input_size=self.conv_out_channels,
            hidden_size=self.gru_hidden_size,
            batch_first=True
        )

        # 3) Skip‐GRUs
        self.skip_gru = nn.ModuleList([
            nn.GRU(self.conv_out_channels, self.skip_hidden_size, batch_first=True)
            for _ in self.skip_lengths
        ])

        # 4) Final fully‐connected
        total_gru_dim = self.gru_hidden_size + len(self.skip_lengths) * self.skip_hidden_size
        self.fc = nn.Linear(total_gru_dim, self.output_size)

        # 5) Autoregressive (linear) on last ar_window timesteps
        self.ar = nn.Linear(self.ar_window, 1)

        # 6) Dropout
        self.dropout = nn.Dropout(cfg['dropout'])

        # Move to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_features)
        batch_size = x.size(0)

        # --- conv ---
        c = x.unsqueeze(1)              # (batch, 1, seq_len, num_features)
        c = F.relu(self.conv(c))        # (batch, conv_out, L', 1)
        c = self.dropout(c)
        c = c.squeeze(-1)               # (batch, conv_out, L')

        # --- main GRU ---
        r, _ = self.gru(c.permute(0,2,1))   # (batch, L', conv_out) -> ...
        r = r[:, -1, :]                     # (batch, gru_hidden)
        r = self.dropout(r)

        # --- skip GRUs ---
        skip_outs = []
        for i, skip_len in enumerate(self.skip_lengths):
            s = c[:, :, -skip_len:]         # last skip_len steps
            s, _ = self.skip_gru[i](s.permute(0,2,1))
            s = s[:, -1, :]
            skip_outs.append(self.dropout(s))

        # --- combine ---
        combined = torch.cat([r] + skip_outs, dim=1)  # (batch, total_gru_dim)
        output = self.fc(combined)                    # (batch, output_size)

        # --- autoregressive term ---
        if self.ar_window > 0 and ACTIVE_DATASET == "unemployment":
            ar = x[:, -self.ar_window:, :]            # (batch, ar_window, num_features)
            ar = self.ar(ar.permute(0,2,1))           # (batch, num_features, 1)
            ar = ar.squeeze(-1)                       # (batch, num_features)
            output = output + ar

        return output
