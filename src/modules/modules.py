import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def one_param(m):
    "Get the first parameter of the model."
    return next(iter(m.parameters()))


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Update the model parameters with exponential moving average.

        Parameters:
            - ma_model (nn.Module): Model with the moving average parameters.
            - current_model (nn.Module): Current model with the original parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Update the average using exponential moving average.

        Parameters:
            - old (torch.Tensor): Old average.
            - new (torch.Tensor): New value.

        Returns:
            - torch.Tensor: Updated average.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Perform a step of exponential moving average.

        Parameters:
            - ema_model (nn.Module): Model with exponential moving average.
            - model (nn.Module): Current model.
            - step_start_ema (int): Start EMA after this number of steps.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Reset the parameters of the EMA model to match the current model.

        Parameters:
            - ema_model (nn.Module): Model with exponential moving average.
            - model (nn.Module): Current model.
        """
        ema_model.load_state_dict(model.state_dict())

 #=============================================================================
# Redesigned UNet for Sensor Data (Conditional Version)
# =============================================================================
class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=128, num_classes=None, device="cuda:0"):
        """
        A simplified UNet optimized for sensor data.
        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            time_dim (int): Dimension for time embedding.
            num_classes (int, optional): Number of classes for conditional embedding.
            device (str): Device identifier.
        """
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        base_channels = 32  # Lower number of channels for sensor data

        # Input convolution: Adjust kernel to treat the data as (seq_size, 1)
        self.inc = DoubleConv(c_in, base_channels, kernel_size=(3,1), padding=(1,0))
        # Two downsampling blocks using self-attention + 2x1 max pooling
        self.down1 = DownSensor(base_channels, base_channels * 2)
        self.down2 = DownSensor(base_channels * 2, base_channels * 4)

        # Bottleneck block
        self.bot = DoubleConv(base_channels * 4, base_channels * 4, kernel_size=(3,1), padding=(1,0))

        # Two upsampling blocks with skip connections
        self.up1 = UpSensor(base_channels * 4, base_channels * 2)
        self.up2 = UpSensor(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, c_out, kernel_size=1)

        # Optional label embedding for conditional generation
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        self.emb_layer = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),  # 128 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(time_dim * 2, time_dim)     # 256 -> 128
        )


    def pos_encoding(self, t, channels):
        """
        Computes a sinusoidal positional encoding for the time steps.
        Args:
            t (torch.Tensor): Timestep tensor.
            channels (int): Number of channels for encoding.
        Returns:
            torch.Tensor: Positional encoding.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        # Embed time step t and add label embedding if provided
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t = t + self.label_emb(y)

        # Contracting path
        x1 = self.inc(x)             # [B, 32, seq_size, 1]
        x2 = self.down1(x1, t)         # [B, 64, seq_size/2, 1]
        x3 = self.down2(x2, t)         # [B, 128, seq_size/4, 1]

        x_bot = self.bot(x3)

        # Expansive path with skip connections
        x = self.up1(x_bot, x2, t)
        x = self.up2(x, x1, t)
        out = self.outc(x)
        return out

# =============================================================================
# Sensor-specific Downsampling and Upsampling Blocks
# =============================================================================
class DownSensor(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Downsampling block for sensor data:
         - Applies self-attention first.
         - Then performs 2x1 max pooling.
         - Finally applies two convolution layers.
        """
        super().__init__()
        self.sa = SelfAttention(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
    def forward(self, x, t):
        x = self.sa(x)
        x = self.pool(x)
        x = self.conv(x)
        return x

class UpSensor(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upsampling block for sensor data:
         - Upsamples the feature map.
         - Concatenates the skip connection.
         - Applies self-attention followed by convolutions.
         
        Note: 'in_channels' is the number of channels from the upsampled input.
              In our U-Net, the skip connection has in_channels // 2 channels.
              So, after concatenation, the total channels become in_channels + (in_channels // 2).
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(2, 1), mode='nearest')
        skip_channels = in_channels // 2  # assuming the skip connection has half the channels
        total_channels = in_channels + skip_channels  # total channels after concatenation
        self.sa = SelfAttention(total_channels)
        self.conv = DoubleConv(total_channels, out_channels, kernel_size=(3,1), padding=(1,0))
    
    def forward(self, x, skip, t):
        x = self.upsample(x)
        # Ensure that spatial dimensions match (crop or pad skip if necessary)
        if x.size(-2) != skip.size(-2):
            diff = skip.size(-2) - x.size(-2)
            skip = skip[:, :, diff//2: diff//2 + x.size(-2), :]
        x = torch.cat([x, skip], dim=1)
        x = self.sa(x)
        x = self.conv(x)
        return x
# =============================================================================
# Utility Modules: DoubleConv and SelfAttention
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Two sequential convolutional layers with ReLU activations.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        """
        A simple self-attention mechanism for 2D feature maps.
        Args:
            in_channels (int): Number of input channels.
            heads (int): Number of attention heads.
        """
        super().__init__()
        self.heads = heads
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        # Reshape and split channels for multi-head attention
        q = self.query(x).view(b, self.heads, c // self.heads, h * w)
        k = self.key(x).view(b, self.heads, c // self.heads, h * w)
        v = self.value(x).view(b, self.heads, c // self.heads, h * w)
        attn = self.softmax(torch.matmul(q.transpose(-2, -1), k) / np.sqrt(c // self.heads))
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        return out


