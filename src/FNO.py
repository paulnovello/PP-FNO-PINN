import torch
import torch.fft
import torch.nn as nn

class FNO1dLayer(nn.Module):
    def __init__(self, n_enc, n_modes):
        super().__init__()
        self.n_enc = n_enc
        self.n_modes = n_modes

        self.W = nn.Conv1d(n_enc, n_enc, kernel_size=1)
        self.R = nn.Parameter(
            torch.randn(n_enc, n_enc, n_modes, dtype=torch.cfloat)
            / (n_enc * max(n_modes, 1))
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2)

        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.n_modes, x_ft.shape[-1])

        x_ft_transformed = torch.zeros_like(x_ft)
        x_ft_transformed[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :modes],
            self.R[:, :, :modes],
        )

        x_ifft = torch.fft.irfft(x_ft_transformed, n=x.shape[-1], dim=-1)
        out = x_ifft + self.W(x)
        return out.transpose(1, 2)


class FNO(nn.Module):
    def __init__(
        self,
        device,
        layers,
        n_enc=16,
        seed=None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device

        self.encoder = nn.Conv1d(1, n_enc, kernel_size=1)
        self.fno_layers = nn.ModuleList(
            [FNO1dLayer(n_enc, n_modes) for n_modes in layers]
        )
        self.activation = nn.GELU()
        self.decoder = nn.Sequential(
            nn.Conv1d(n_enc, n_enc, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n_enc, 1, kernel_size=1),
        )

    def forward(self, input_tensor):
        x = input_tensor
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)

        for fno_layer in self.fno_layers:
            x = self.activation(fno_layer(x))

        x = x.transpose(1, 2)
        out = self.decoder(x)
        out = out.transpose(1, 2)
        return out.squeeze(0)
