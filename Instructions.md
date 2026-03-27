# Surrogate Modelling with Fourier Neural Operators and PINNs

In this Programming Practical, you will:

- Implement a Fourier Neural Operator
- Test it on the approximation of Backwater ODE's solution (see Subject_previous_pp.pdf)
- Train an MLP on the Backwater PDE (Implicit Neural Representation)
- Test the effect of using a Physics Informed loss (implemented in the previous Programming Practical) with the MLP
- Compare all these approaches when learning Backwater ODE's solution

## Fourier Neural Operator

Here is a cheatsheet of FNO in 1D from the course:

![SciML INSA P1.png](attachment:ea228280-85fa-4dfc-9b23-cbbb1aa1bd10:SciML_INSA_P1.png)

![SciML INSA P1.png](attachment:9fe4ff6c-4b43-41ce-a364-55bb1bc7334c:SciML_INSA_P1.png)

We will be implementing a model `FNO` which calls for `FNO1dLayers`. The FNO will be used in the Backwater problem, take the vector of spatial coordinates as input, and output the water height. 

 In the file `Class_FNO.py`, complete the class `FNO1dLayer`:

l.27 -> 31

```python
# Linear transform on lowest Fourier modes
# Shape: [???, ???, ???] for channel-wise mode transform
self.R = nn.Parameter(
    torch.randn(???, ???, ???, dtype=torch.cfloat) * 0.1
)

```

l.33 -> 64 (`forward` method):

```python
def forward(self, x):
    # x: [batch, k, n_enc] (or [k, n_enc])
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, k, n_enc]
    x = x.transpose(1, 2)  # [batch, n_enc, k]

    # FFT along spatial dim (k)
    x_ft = torch.fft.rfft(x, dim=-1)  # [batch, n_enc, modes_ft]
    modes = min(self.n_modes, x_ft.shape[-1])
    # Apply learnable weights channel-wise to lowest n_modes Fourier modes
    x_ft_transformed = x_ft.clone()

    # For each channel, apply mode transformation
    for c in range(self.n_enc):
        x_ft_transformed[:, c, ???] = torch.matmul(
            x_ft[:, c : c + 1, ???],  # [batch, 1, ???]
            self.R[c, ???, ???],  # [???, ???]
        ).squeeze(
            1
        )  # [batch, modes]

    # Zero out higher modes
    if modes < x_ft.shape[-1]:
        x_ft_transformed[..., ???] = 0

    # Inverse FFT
    x_ifft = torch.fft.irfft(x_ft_transformed, n=x.shape[-1], dim=-1)

    # Residual connection: W(x) + F-1(R(F(x)))
    out = ???
    out = out.transpose(1, 2)  # [batch, k, n_enc]
    return out

```

And the `forward` method of the class `FNO`, l.128 -> 159

```python
def forward(self, input_tensor):
    """
    input_tensor: [K, 1] or [B, K, 1]
    returns: [B, K, 1]
    """
    # Normalize
    input_tensor = (input_tensor - self.variable_min) / (
        self.variable_max - self.variable_min
    )

    K, _ = input_tensor.shape

    x_in = input_tensor  # [K, 1]
    if x_in.dim() == 2:
        x_in = x_in.unsqueeze(0)  # [1, K, 1]

    # Lift
    x = x_in.transpose(1, 2)  # [1, 1, K]

    # Lift to n_enc channels
    x = ???  # [1, n_enc, K]
    x = x.transpose(1, 2)  # [1, K, n_enc]

    # Apply FNO layers + activations
    for fno_layer in self.fno_layers:
        x = ???  # [1, K, n_enc]
    x = x.transpose(1, 2)  # [1, n_enc, K]

    # Project back to 1 channel
    out = ???  # [1, 1, K]
    out = out.transpose(1, 2)  # [1, K, 1]
    return out.squeeze(0)  # [K, 1]

```

Now go to the notebook `main_FNO.pynb` and test your implementation. You should see on the plots that the training is very messy, with lots of spikes between the observation points. Any idea on how to solve the problem?

## MLP and PINNs

In this section, you will look at `main_PINN.ipynb`, which is very similar to that of the previous programming practical. Try to change the argument `use_pinn` of the function `train_model` and to play with `N_obs` from `ObservationPoints` to see what happens.

## Study on N_obs

Now that everything is running, complete the scripts `n_obs_study_FNO.py` and `n_obs_study_PINN.py` to run a study on the effect of `N_obs` on the error (you will find a value range of `[10, 20, 40, 60, 80, 100]` in the file). The code for training the model is pasted and you have to create the loops and save `final_rmse` appropriately (for instance as `.npz` or `.pkl` files).

Once it is done, create a script to read the results and plot on the same graph against n_obs:

- The error of the FNO
- The error of the MLP with PINN loss
- The error of the MLP without PINN loss

Do you have any observation?

## Bonus: FNO that takes $Ks$ as input

The current FNO $\tilde{h}$ takes the vector of spatial coordinates as input $X = (x_1,….,x_{N_{obs}})$ and approximates $(h(x_1),….,h(x_{N_{obs}}))$, where $h$ is computed with $K_s$ fixed.

How would you do to build a FNO that takes $K_s \in \mathbb{R}^p$ as input (discretized as $(K_s(x_1),….,K_s(x_{N_{obs}}))$ ), and output an approximation of $(h(x_1, K_s),….,h(x_{N_{obs}}, K_s))$ ($h$ now depends on $K_s$.