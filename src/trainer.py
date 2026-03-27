import time
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from losses import (
    LossNormalizationState,
    LossWeights,
    observation_loss,
    physics_informed_losses,
)


@dataclass(frozen=True)
class TrainingResult:
    total_loss: float
    residual_loss: float
    observation_loss: float
    boundary_loss: float
    steps: int
    evaluations: int


@dataclass
class _TrainingState:
    start_time: float
    print_every: int
    steps: int = 0
    evaluations: int = 0
    last_total: torch.Tensor | None = None
    last_residual: torch.Tensor | None = None
    last_observation: torch.Tensor | None = None
    last_boundary: torch.Tensor | None = None
    last_grad_norm: float = 0.0


def _grad_norm(parameters):
    gradients = [
        parameter.grad.detach().flatten()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return torch.norm(torch.cat(gradients), p=2).item()


def _zero_tensor(device):
    return torch.tensor(0.0, device=device)


class Trainer:
    def __init__(self, optimizer_name="adamw", max_grad_norm=1e3):
        self.optimizer_name = optimizer_name
        self.max_grad_norm = max_grad_norm

    def fit(
        self,
        model,
        col,
        obs,
        ref_solution,
        *,
        n_iter=300,
        lr=1e-2,
        display_freq=(20, 100),
    ):
        del ref_solution

        state = self._initialize_state(display_freq)

        if self.optimizer_name == "lbfgs":
            self._fit_with_lbfgs(model, col, obs, n_iter, state)
        else:
            self._fit_with_adamw(model, col, obs, n_iter, lr, state)

        return self._build_result(model.device, state)

    def _initialize_state(self, display_freq):
        return _TrainingState(
            start_time=time.time(),
            print_every=max(display_freq[0], 1),
        )

    def _fit_with_adamw(self, model, col, obs, n_iter, lr, state):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for step in range(1, n_iter + 1):
            optimizer.zero_grad()
            loss = observation_loss(model, obs, col)
            loss.backward()

            parameters = list(model.parameters())
            nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.max_grad_norm,
                norm_type=2.0,
            )

            optimizer.step()

            state.steps = step
            state.evaluations += 1
            state.last_grad_norm = _grad_norm(parameters)
            self._store_observation_state(model.device, loss, state)
            self._maybe_print_observation_progress(step, state)

    def _fit_with_lbfgs(self, model, col, obs, n_iter, state):
        parameters = list(model.parameters())
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1,
            max_iter=n_iter - 1,
            max_eval=10 * n_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=-1,
            tolerance_change=-1,
        )

        grads = partial(
            self._compute_observation_gradients,
            optimizer=optimizer,
            model=model,
            col=col,
            obs=obs,
            parameters=parameters,
            state=state,
        )
        optimizer.step(grads)
        state.steps = 1

    def _compute_observation_gradients(
        self,
        *,
        optimizer,
        model,
        col,
        obs,
        parameters,
        state,
    ):
        optimizer.zero_grad()
        loss = observation_loss(model, obs, col)
        loss.backward()

        nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.max_grad_norm,
            norm_type=2.0,
        )

        state.evaluations += 1
        state.last_grad_norm = _grad_norm(parameters)
        self._store_observation_state(model.device, loss, state)
        self._maybe_print_observation_progress(state.evaluations, state)
        return loss

    def _store_observation_state(self, device, loss, state):
        state.last_total = loss.detach().clone()
        state.last_residual = _zero_tensor(device)
        state.last_observation = loss.detach().clone()
        state.last_boundary = _zero_tensor(device)

    def _maybe_print_observation_progress(self, step, state):
        if step % state.print_every != 0:
            return

        print("#" * 50)
        print(f"Processing iteration {step}")
        print("-" * 25)
        print(f"J_obs       = {state.last_observation.item():.2e}")
        print(f"||grad(J)|| = {state.last_grad_norm:.2e}")
        print(f"time        = {time.time() - state.start_time:.2f} s")

    def _build_result(self, device, state):
        total = state.last_total if state.last_total is not None else _zero_tensor(device)
        residual = (
            state.last_residual
            if state.last_residual is not None
            else _zero_tensor(device)
        )
        observation = (
            state.last_observation
            if state.last_observation is not None
            else _zero_tensor(device)
        )
        boundary = (
            state.last_boundary
            if state.last_boundary is not None
            else _zero_tensor(device)
        )

        return TrainingResult(
            total_loss=total.item(),
            residual_loss=residual.item(),
            observation_loss=observation.item(),
            boundary_loss=boundary.item(),
            steps=state.steps,
            evaluations=state.evaluations,
        )


class PITrainer:
    PRETRAINING_WEIGHTS = LossWeights(residual=0.0, observation=1.0, boundary=1.0)
    FULL_WEIGHTS = LossWeights(residual=1.0, observation=1.0, boundary=1.0)

    def __init__(self, train_k=False, max_grad_norm=1e3):
        self.train_k = train_k
        self.max_grad_norm = max_grad_norm

    def fit(
        self,
        model,
        col,
        obs,
        ref_solution,
        *,
        pre_train_iter=100,
        alter_steps=4,
        alter_freq=(40, 10),
        normalize_losses=True,
        display_freq=(20, 100),
    ):
        del ref_solution

        state = self._initialize_state(display_freq)
        normalization_state = self._initialize_normalization_state(normalize_losses)

        # Step 1: pretrain the neural network on observations and boundary condition only.
        self._run_lbfgs_phase(
            model,
            col,
            obs,
            list(model.network_parameters()),
            pre_train_iter,
            self.PRETRAINING_WEIGHTS,
            normalize_losses,
            normalization_state,
            state,
        )

        for _ in range(alter_steps):
            # Step 2a: if requested, update the physical parameter k with the full PI loss.
            if self.train_k:
                self._run_lbfgs_phase(
                    model,
                    col,
                    obs,
                    [model.k],
                    alter_freq[1],
                    self.FULL_WEIGHTS,
                    normalize_losses,
                    normalization_state,
                    state,
                )

            # Step 2b: update the neural network weights with the full PI loss.
            self._run_lbfgs_phase(
                model,
                col,
                obs,
                list(model.network_parameters()),
                alter_freq[0],
                self.FULL_WEIGHTS,
                normalize_losses,
                normalization_state,
                state,
            )

        return self._build_result(model.device, state)

    def _initialize_state(self, display_freq):
        return _TrainingState(
            start_time=time.time(),
            print_every=max(display_freq[0], 1),
        )

    def _initialize_normalization_state(self, normalize_losses):
        if not normalize_losses:
            return None
        return LossNormalizationState()

    def _run_lbfgs_phase(
        self,
        model,
        col,
        obs,
        parameters,
        max_iter,
        weights,
        normalize_losses,
        normalization_state,
        state,
    ):
        parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        if not parameters or max_iter <= 0:
            return

        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1,
            max_iter=max_iter - 1,
            max_eval=10 * max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=-1,
            tolerance_change=-1,
        )
        grads = partial(
            self._compute_physics_informed_gradients,
            optimizer=optimizer,
            model=model,
            col=col,
            obs=obs,
            parameters=parameters,
            weights=weights,
            normalize_losses=normalize_losses,
            normalization_state=normalization_state,
            state=state,
        )
        optimizer.step(grads)
        state.steps += 1

    def _compute_physics_informed_gradients(
        self,
        *,
        optimizer,
        model,
        col,
        obs,
        parameters,
        weights,
        normalize_losses,
        normalization_state,
        state,
    ):
        optimizer.zero_grad()

        losses = physics_informed_losses(
            model,
            col,
            obs,
            weights=weights,
            normalize=normalize_losses,
            state=normalization_state,
        )
        total = losses.total
        total.backward()

        nn.utils.clip_grad_norm_(
            parameters,
            max_norm=self.max_grad_norm,
            norm_type=2.0,
        )

        if self.train_k:
            model.clamp_parameters()

        state.evaluations += 1
        state.last_grad_norm = _grad_norm(parameters)
        self._store_physics_informed_state(losses, total, state)
        self._maybe_print_physics_informed_progress(model, state)
        return total

    def _store_physics_informed_state(self, losses, total, state):
        state.last_total = total.detach().clone()
        state.last_residual = losses.residual.detach().clone()
        state.last_observation = losses.observation.detach().clone()
        state.last_boundary = losses.boundary.detach().clone()

    def _maybe_print_physics_informed_progress(self, model, state):
        if state.evaluations % state.print_every != 0:
            return

        print("#" * 50)
        print(f"Processing evaluation {state.evaluations}")
        print("-" * 25)
        print(
            "J           = "
            f"{state.last_total.item():.2e} "
            f"(residual : {state.last_residual.item():.2e}, "
            f"obs : {state.last_observation.item():.2e}, "
            f"BC : {state.last_boundary.item():.2e})"
        )
        print(f"||grad(J)|| = {state.last_grad_norm:.2e}")
        print(f"parameter    = {model.parameter_values().detach().clone()}")
        print(f"time         = {time.time() - state.start_time:.2f} s")

    def _build_result(self, device, state):
        total = state.last_total if state.last_total is not None else _zero_tensor(device)
        residual = (
            state.last_residual
            if state.last_residual is not None
            else _zero_tensor(device)
        )
        observation = (
            state.last_observation
            if state.last_observation is not None
            else _zero_tensor(device)
        )
        boundary = (
            state.last_boundary
            if state.last_boundary is not None
            else _zero_tensor(device)
        )

        return TrainingResult(
            total_loss=total.item(),
            residual_loss=residual.item(),
            observation_loss=observation.item(),
            boundary_loss=boundary.item(),
            steps=state.steps,
            evaluations=state.evaluations,
        )
