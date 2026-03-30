import torch
import torch.fft
import torch.nn as nn


class FNO1dLayer(nn.Module):
    def __init__(self, n_channels, n_modes):
        super().__init__()
        self.n_channels = n_channels
        self.n_modes = n_modes

        self.pointwise_conv = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.spectral_weights = nn.Parameter(
            torch.randn(n_channels, n_channels, n_modes, dtype=torch.cfloat)
            / (n_channels * max(n_modes, 1))
        )

    def forward(self, input_tensor):
        x = input_tensor

        # Notebook inputs have shape [N, n_channels], so we add a batch dimension.
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.transpose(1, 2)
        x_fourier = torch.fft.rfft(x, dim=-1)
        number_of_kept_modes = min(self.n_modes, x_fourier.shape[-1])

        transformed_fourier = torch.zeros_like(x_fourier)
        for mode_index in range(number_of_kept_modes):
            input_mode = x_fourier[:, :, mode_index]
            mode_weights = self.spectral_weights[:, :, mode_index]
            transformed_fourier[:, :, mode_index] = input_mode @ mode_weights

        x_back_to_space = torch.fft.irfft(
            transformed_fourier,
            n=x.shape[-1],
            dim=-1,
        )
        pointwise_output = self.pointwise_conv(x)

        return (x_back_to_space + pointwise_output).transpose(1, 2)


class FNO(nn.Module):
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
