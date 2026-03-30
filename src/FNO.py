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

        # Notebook inputs have shape [N, 1], so we add a batch dimension.
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.transpose(1, 2)
        # TODO apply torch.fft.rfft to obtain x_fourier
        number_of_kept_modes = min(self.n_modes, x_fourier.shape[-1])

        transformed_fourier = torch.zeros_like(x_fourier)
        for mode_index in range(number_of_kept_modes):
            # TODO: populate transformed_fourier for each mode_index using self.spectral_weights

        # TODO: apply torch.fft.irfft to obtain x_back_to_space from transformed_fourier
        # TODO: get pointwise_output using the conv layer to prepare the residual connection 

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

        # TODO define self.input_encoder using a conv layer
        # TODO define two FNO layers, self.fno_layer_1 and self.fno_layer_2
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

        # TODO: apply the first and second FNO layers

        # Step 4: decode back to one scalar output.
        x = x.transpose(1, 2)
        x = self.output_decoder_1(x)
        x = self.activation(x)
        x = self.output_decoder_2(x)
        x = x.transpose(1, 2)

        return x.squeeze(0)
