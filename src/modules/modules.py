
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SelfAttention(nn.Module):
    """
    Self-Attention module using Multihead Attention, Layer Normalization, and Feedforward layers.
    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """Forward pass for Self - Attention adjusted for 1D signal data.
        Input x shape: (batch, channels, width)
        """
        # For 1D signals, we only have width, no height.
        batch_size, channels, width = x.shape

        # Reshape to (batch, width, channels) for attention mechanism
        x = x.permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)

        # Apply feedforward and add residual connection
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        # Permute back to original shape: (batch, channels, width)
        return attention_value.permute(0, 2, 1)

class DoubleConv(nn.Module):
"""
    DoubleConvolution module with optional residual connection.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, conv_type="1d"):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels


        if conv_type == "1d":
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, mid_channels),
                nn.GELU(),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, out_channels),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, mid_channels),
                nn.GELU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, out_channels),
            )


    def forward(self, x):
        """
        Forward pass of the DoubleConv module.

        Parameters:
            - x (torch.Tensor): Input tensor.

        Returns:
            - torch.Tensor: Output tensor after double convolution.
        """

        if x.dim() == 4 and x.size(-1) == 1:
            x = x.squeeze(-1)  # Remove the last dimension

        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Down-sampling block with max-pooling, DoubleConv blocks, and an embedding layer.
    """
    def __init__(self, in_channels, out_channels, emb_dim=256, conv_type="1d"):
        super().__init__()
        if conv_type == "1d":
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool1d(2),
                DoubleConv(in_channels, in_channels, residual=True, conv_type=conv_type),
                DoubleConv(in_channels, out_channels, conv_type=conv_type),
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, in_channels, residual=True, conv_type=conv_type),
                DoubleConv(in_channels, out_channels, conv_type=conv_type),
            )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        """
        Forward pass of the Down-sampling block.

        Parameters:
            - x (torch.Tensor): Input tensor.
            - t (torch.Tensor): Time information tensor.

        Returns:
            - torch.Tensor: Output tensor after down-sampling and embedding.
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1]) if len(x.size()) == 3 else self.emb_layer(t)[:, :,
                                                                                                 None, None].repeat(1,
                                                                                                                    1,
                                                                                                                    x.shape[
                                                                                                                        -2],
                                                                                                                    x.shape[
                                                                                                                        -1])
        return x + emb


class Up(nn.Module):
    """
    Up-sampling block with upsampling, concatenation, DoubleConv blocks, and an embedding layer.
    """
    def __init__(self, in_channels, out_channels, emb_dim=256, conv_type="1d"):
        super().__init__()
        if conv_type == "1d":
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, conv_type=conv_type),
            DoubleConv(in_channels, out_channels, in_channels // 2, conv_type=conv_type),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        """
        Forward pass of the Up-sampling block.

        Parameters:
            - x (torch.Tensor): Input tensor.
            - skip_x (torch.Tensor): Skip connection tensor.
            - t (torch.Tensor): Time information tensor.

        Returns:
            - torch.Tensor: Output tensor after up-sampling and embedding.
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1]) if len(x.size()) == 3 else self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    """
    U-Net architecture with optional self-attention and conditional embedding.
    """
   

    def __init__(self, c_in, c_out, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64, conv_type="1d")
        self.down1 = Down(64, 128, conv_type="1d")
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256, conv_type="1d")
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, conv_type="1d")
        self.sa3 = SelfAttention(256)
        
        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256, conv_type="1d")
            self.bot3 = DoubleConv(256, 256, conv_type="1d")
        else:
            self.bot1 = DoubleConv(256, 512, conv_type="1d")
            self.bot2 = DoubleConv(512, 512, conv_type="1d")
            self.bot3 = DoubleConv(512, 256, conv_type="1d")

        self.up1 = Up(512, 128, conv_type="1d")
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64, conv_type="1d")
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64, conv_type="1d")
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Generate positional encoding based on time information.

        Parameters:
            - t (torch.Tensor): Time information tensor.
            - channels (int): Number of channels.

        Returns:
            - torch.Tensor: Positonal encoding tensor.
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        """
        Forward pass of the U-Net architecture.

        Parameters:
            - x (torch.Tensor): Input tensor.
            - t (torch.Tensor): Time information tensor.

        Returns:
            - torch.Tensor: Output tensor after U-Net processing.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)

        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

    def forward(self, x, t):
        """
        Forward pass of the U-Net architecture with positional encoding.

        Parameters:
            - x (torch.Tensor): Input tensor.
            - t (torch.Tensor): Time information tensor.

        Returns:
            - torch.Tensor: Output tensor after U-Net processing.
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    """
    Conditional U-Net architecture with label embedding.
    """
    def __init__(self, c_in, c_out, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        """
        Forward pass of the conditional U-Net architecture.

        Parameters:
            - x (torch.Tensor): Input tensor.
            - t (torch.Tensor): Time information tensor.
            - y (torch.Tensor): Class label tensor.

        Returns:
            - torch.Tensor: Output tensor after conditional U-Net processing.
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)
