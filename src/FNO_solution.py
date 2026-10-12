"""Fourier Neural Operator (FNO) for the 1D backwater problem.

The FNO treats the whole grid of coordinates as one signal and predicts
the water height on that grid.
"""

import torch
import torch.fft
import torch.nn as nn


class FNO1dLayer(nn.Module):
    """One 1D Fourier layer.

    Combines a spectral path (for each channel, a learned linear map
    applied to the lowest Fourier modes of that channel) with a pointwise
    path (a 1x1 convolution) and returns their sum.

    Note: in practice (e.g. Li et al., 2021), the spectral path also mixes
    the channels in Fourier space. Here, for simplicity, each channel is
    transformed independently and channels are only mixed by the 1x1
    convolution.

    Args:
        n_channels: number of input and output channels.
        n_modes: number of low-frequency Fourier modes kept by the spectral
            path; higher modes are discarded.
    """

    def __init__(self, n_channels, n_modes):
        super().__init__()
        self.n_channels = n_channels
        self.n_modes = n_modes

        self.pointwise_conv = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.spectral_weights = nn.Parameter(
            torch.randn(n_channels, n_modes, n_modes, dtype=torch.cfloat)
            / max(n_modes, 1)
        )  # R in course slides

    def forward(self, input_tensor):
        """Apply the layer.

        Args:
            input_tensor: tensor of shape ``[N, n_channels]`` or
                ``[batch, N, n_channels]``.

        Returns:
            Tensor of shape ``[batch, N, n_channels]``.
        """
        x = input_tensor

        # Notebook inputs have shape [N, n_channels], so we add a batch dimension.
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.transpose(1, 2)
        x_fourier = torch.fft.rfft(x, dim=-1)
        number_of_kept_modes = min(self.n_modes, x_fourier.shape[-1])

        transformed_fourier = torch.zeros_like(x_fourier)
        for channel_index in range(self.n_channels):
            # Keep the low modes of this channel and apply its matrix R.
            kept_modes = x_fourier[:, channel_index, :number_of_kept_modes]
            mode_weights = self.spectral_weights[
                channel_index, :number_of_kept_modes, :number_of_kept_modes
            ]
            transformed_fourier[:, channel_index, :number_of_kept_modes] = (
                kept_modes @ mode_weights
            )

        x_back_to_space = torch.fft.irfft(
            transformed_fourier,
            n=x.shape[-1],
            dim=-1,
        )
        pointwise_output = self.pointwise_conv(x)

        return (x_back_to_space + pointwise_output).transpose(1, 2)


class FNO(nn.Module):
    """Fourier Neural Operator mapping a 1D grid of coordinates to ``h``.

    A pointwise encoder lifts the coordinate to ``n_channels`` channels,
    two ``FNO1dLayer`` blocks process the signal, and a two-layer pointwise
    decoder projects back to one channel.

    Args:
        device: torch device stored on the model (used by the losses).
        n_modes_layer_1: Fourier modes kept in the first layer.
        n_modes_layer_2: Fourier modes kept in the second layer.
        n_channels: width of the hidden representation.
        seed: if given, seeds ``torch`` before initializing the weights.
    """

    def __init__(
        self,
        device,
        n_modes_layer_1=10,
        n_modes_layer_2=10,
        n_channels=5,
        seed=None,
    ):
        super().__init__()
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        self.input_encoder = nn.Conv1d(1, n_channels, kernel_size=1)
        self.fno_layer_1 = FNO1dLayer(n_channels, n_modes_layer_1)
        self.fno_layer_2 = FNO1dLayer(n_channels, n_modes_layer_2)
        self.activation = nn.GELU()
        self.output_decoder_1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.output_decoder_2 = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, normalized_coordinates):
        """Predict the water height on a grid.

        Args:
            normalized_coordinates: coordinates in ``[0, 1]`` of shape
                ``[N, 1]`` (or ``[batch, N, 1]``).

        Returns:
            Predicted heights of shape ``[N, 1]`` (batch dimension squeezed).
        """
        x = normalized_coordinates

        # Notebook inputs have shape [N, 1], so we add a batch dimension.
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Step 1: encode the scalar coordinate into several channels.
        x = x.transpose(1, 2)
        x = self.input_encoder(x)
        x = x.transpose(1, 2)

        # Step 2: apply the first Fourier layer.
        x = self.fno_layer_1(x)
        x = self.activation(x)

        # Step 3: apply the second Fourier layer.
        x = self.fno_layer_2(x)
        x = self.activation(x)

        # Step 4: decode back to one scalar output.
        x = x.transpose(1, 2)
        x = self.output_decoder_1(x)
        x = self.activation(x)
        x = self.output_decoder_2(x)
        x = x.transpose(1, 2)

        return x.squeeze(0)
