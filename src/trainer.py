import time

import torch
import torch.nn as nn

from losses import observation_loss, physics_informed_loss


def _display_frequencies(display_freq):
    if isinstance(display_freq, int):
        return max(display_freq, 1), 0

    if len(display_freq) == 1:
        return max(display_freq[0], 1), 0

    return max(display_freq[0], 1), max(display_freq[1], 0)


def _gradient_norm(parameters):
    gradient_list = []
    for parameter in parameters:
        if parameter.grad is not None:
            gradient_list.append(parameter.grad.detach().flatten())

    if not gradient_list:
        return 0.0

    return torch.norm(torch.cat(gradient_list), p=2).item()


def _plot_training_state(
    model,
    collocation_points,
    observation_points,
    ref_solution,
):
    if ref_solution is None:
        return

    try:
        from IPython.display import display as ipy_display
    except ImportError:
        return

    import matplotlib.pyplot as plt

    import display as display_module

    figure = display_module.display_results(
        model,
        collocation_points,
        ref_solution,
        observation_points,
        plot_col=True,
        show=False,
    )
    ipy_display(figure)
    plt.close(figure)


def _maybe_plot_training_state(
    before_evaluations,
    after_evaluations,
    plot_every,
    model,
    collocation_points,
    observation_points,
    ref_solution,
):
    if plot_every <= 0:
        return

    if after_evaluations // plot_every <= before_evaluations // plot_every:
        return

    _plot_training_state(
        model,
        collocation_points,
        observation_points,
        ref_solution,
    )


def _build_lbfgs(
    parameters,
    *,
    use_line_search=True,
):
    return torch.optim.LBFGS(
        parameters,
        lr=1,
        max_iter=1,
        max_eval=1,
        line_search_fn="strong_wolfe" if use_line_search else None,
        tolerance_grad=-1,
        tolerance_change=-1,
    )


def _run_lbfgs_with_evaluation_budget(
    optimizer,
    closure,
    evaluation_budget,
    *,
    evaluation_counter,
    on_step_completed=None,
):
    evaluation_budget = max(int(evaluation_budget), 0)
    if evaluation_budget <= 0:
        return 0

    start_evaluations = evaluation_counter()
    base_line_search = optimizer.param_groups[0]["line_search_fn"]
    number_of_steps = 0

    while evaluation_counter() - start_evaluations < evaluation_budget:
        before_evaluations = evaluation_counter()
        remaining_evaluations = evaluation_budget - (
            before_evaluations - start_evaluations
        )
        parameter_group = optimizer.param_groups[0]
        parameter_group["max_iter"] = 1
        parameter_group["max_eval"] = max(remaining_evaluations, 1)
        parameter_group["line_search_fn"] = (
            base_line_search if remaining_evaluations > 1 else None
        )

        optimizer.step(closure)

        after_evaluations = evaluation_counter()
        if after_evaluations <= before_evaluations:
            break

        number_of_steps += 1
        if on_step_completed is not None:
            on_step_completed(before_evaluations, after_evaluations)

    optimizer.param_groups[0]["line_search_fn"] = base_line_search
    return number_of_steps


class Trainer:
    def __init__(self, optimizer_name="adamw", max_grad_norm=1e3):
        self.optimizer_name = optimizer_name
        self.max_grad_norm = max_grad_norm

    def fit(
        self,
        model,
        collocation_points,
        observation_points,
        ref_solution,
        *,
        n_iter=300,
        lr=1e-2,
        display_freq=(20, 100),
    ):
        start_time = time.time()
        print_every, plot_every = _display_frequencies(display_freq)

        number_of_steps = 0
        number_of_evaluations = 0
        last_total_loss = torch.tensor(0.0, device=model.device)
        last_observation_loss = torch.tensor(0.0, device=model.device)
        last_gradient_norm = 0.0

        if self.optimizer_name == "lbfgs":
            parameters = list(model.parameters())
            optimizer = _build_lbfgs(parameters)

            # LBFGS asks for a function that recomputes the loss and gradients.
            def grads():
                nonlocal number_of_evaluations
                nonlocal last_total_loss
                nonlocal last_observation_loss
                nonlocal last_gradient_norm

                optimizer.zero_grad()
                total_loss = observation_loss(
                    model,
                    observation_points,
                    collocation_points,
                )
                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=self.max_grad_norm,
                    norm_type=2.0,
                )

                number_of_evaluations += 1
                last_total_loss = total_loss.detach().clone()
                last_observation_loss = total_loss.detach().clone()
                last_gradient_norm = _gradient_norm(parameters)

                if number_of_evaluations % print_every == 0:
                    print("#" * 50)
                    print(f"Processing evaluation {number_of_evaluations}")
                    print("-" * 25)
                    print(f"J_obs       = {last_observation_loss.item():.2e}")
                    print(f"||grad(J)|| = {last_gradient_norm:.2e}")
                    print(f"time        = {time.time() - start_time:.2f} s")
                return total_loss

            def on_step_completed(before_evaluations, after_evaluations):
                _maybe_plot_training_state(
                    before_evaluations,
                    after_evaluations,
                    plot_every,
                    model,
                    collocation_points,
                    observation_points,
                    ref_solution,
                )

            number_of_steps = _run_lbfgs_with_evaluation_budget(
                optimizer,
                grads,
                n_iter,
                evaluation_counter=lambda: number_of_evaluations,
                on_step_completed=on_step_completed,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            for step in range(1, n_iter + 1):
                optimizer.zero_grad()
                total_loss = observation_loss(
                    model,
                    observation_points,
                    collocation_points,
                )
                total_loss.backward()

                parameters = list(model.parameters())
                nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=self.max_grad_norm,
                    norm_type=2.0,
                )

                optimizer.step()

                number_of_steps = step
                number_of_evaluations += 1
                last_total_loss = total_loss.detach().clone()
                last_observation_loss = total_loss.detach().clone()
                last_gradient_norm = _gradient_norm(parameters)

                if step % print_every == 0:
                    print("#" * 50)
                    print(f"Processing iteration {step}")
                    print("-" * 25)
                    print(f"J_obs       = {last_observation_loss.item():.2e}")
                    print(f"||grad(J)|| = {last_gradient_norm:.2e}")
                    print(f"time        = {time.time() - start_time:.2f} s")
                if plot_every > 0 and step % plot_every == 0:
                    _plot_training_state(
                        model,
                        collocation_points,
                        observation_points,
                        ref_solution,
                    )

        return {
            "total_loss": float(last_total_loss.item()),
            "residual_loss": 0.0,
            "observation_loss": float(last_observation_loss.item()),
            "boundary_loss": 0.0,
            "steps": number_of_steps,
            "evaluations": number_of_evaluations,
        }


class PITrainer:
    def __init__(self, train_k=False, max_grad_norm=1e3):
        self.train_k = train_k
        self.max_grad_norm = max_grad_norm

    def fit(
        self,
        model,
        collocation_points,
        observation_points,
        ref_solution,
        *,
        pre_train_iter=100,
        alter_steps=4,
        alter_freq=(40, 10),
        scale_losses=True,
        display_freq=(20, 100),
    ):
        print_every, plot_every = _display_frequencies(display_freq)

        training_state = {
            "start_time": time.time(),
            "print_every": print_every,
            "plot_every": plot_every,
            "number_of_steps": 0,
            "number_of_evaluations": 0,
            "last_total_loss": torch.tensor(0.0, device=model.device),
            "last_residual_loss": torch.tensor(0.0, device=model.device),
            "last_observation_loss": torch.tensor(0.0, device=model.device),
            "last_boundary_loss": torch.tensor(0.0, device=model.device),
            "last_gradient_norm": 0.0,
        }

        if scale_losses:
            loss_scales = {
                "residual": None,
                "observation": None,
                "boundary": None,
            }
        else:
            loss_scales = None

        # Step 1: pretrain the neural network on observations and boundary only.
        self._run_lbfgs_phase(
            model,
            collocation_points,
            observation_points,
            list(model.network_parameters()),
            pre_train_iter,
            lambda_residual=0.0,
            lambda_observation=1.0,
            lambda_boundary=1.0,
            scale_losses=scale_losses,
            loss_scales=loss_scales,
            ref_solution=ref_solution,
            training_state=training_state,
        )

        # Step 2: alternate between the parameter update and the network update.
        for _ in range(alter_steps):
            if self.train_k:
                self._run_lbfgs_phase(
                    model,
                    collocation_points,
                    observation_points,
                    [model.k],
                    alter_freq[1],
                    lambda_residual=1.0,
                    lambda_observation=1.0,
                    lambda_boundary=1.0,
                    scale_losses=scale_losses,
                    loss_scales=loss_scales,
                    ref_solution=ref_solution,
                    training_state=training_state,
                )

            self._run_lbfgs_phase(
                model,
                collocation_points,
                observation_points,
                list(model.network_parameters()),
                alter_freq[0],
                lambda_residual=1.0,
                lambda_observation=1.0,
                lambda_boundary=1.0,
                scale_losses=scale_losses,
                loss_scales=loss_scales,
                ref_solution=ref_solution,
                training_state=training_state,
            )

        return {
            "total_loss": float(training_state["last_total_loss"].item()),
            "residual_loss": float(training_state["last_residual_loss"].item()),
            "observation_loss": float(
                training_state["last_observation_loss"].item()
            ),
            "boundary_loss": float(training_state["last_boundary_loss"].item()),
            "steps": training_state["number_of_steps"],
            "evaluations": training_state["number_of_evaluations"],
        }

    def _run_lbfgs_phase(
        self,
        model,
        collocation_points,
        observation_points,
        parameters,
        max_iter,
        *,
        lambda_residual,
        lambda_observation,
        lambda_boundary,
        scale_losses,
        loss_scales,
        ref_solution,
        training_state,
    ):
        trainable_parameters = []
        for parameter in parameters:
            if parameter.requires_grad:
                trainable_parameters.append(parameter)

        if not trainable_parameters or max_iter <= 0:
            return

        optimizer = _build_lbfgs(
            trainable_parameters,
            use_line_search=True,
        )

        # LBFGS asks for a function that recomputes the loss and gradients.
        def grads():
            optimizer.zero_grad()
            loss_values = physics_informed_loss(
                model,
                collocation_points,
                observation_points,
                lambda_residual=lambda_residual,
                lambda_observation=lambda_observation,
                lambda_boundary=lambda_boundary,
                scale_losses=scale_losses,
                loss_scales=loss_scales,
            )
            total_loss = loss_values["total_loss"]
            total_loss.backward()

            nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=self.max_grad_norm,
                norm_type=2.0,
            )

            if self.train_k:
                model.clamp_parameters()

            training_state["number_of_evaluations"] += 1
            training_state["last_total_loss"] = total_loss.detach().clone()
            training_state["last_residual_loss"] = loss_values[
                "residual_loss"
            ].detach().clone()
            training_state["last_observation_loss"] = loss_values[
                "observation_loss"
            ].detach().clone()
            training_state["last_boundary_loss"] = loss_values[
                "boundary_loss"
            ].detach().clone()
            training_state["last_gradient_norm"] = _gradient_norm(
                trainable_parameters
            )

            if (
                training_state["number_of_evaluations"]
                % training_state["print_every"]
                == 0
            ):
                print("#" * 50)
                print(
                    "Processing evaluation "
                    f"{training_state['number_of_evaluations']}"
                )
                print("-" * 25)
                print(
                    "J           = "
                    f"{training_state['last_total_loss'].item():.2e} "
                    f"(residual : {training_state['last_residual_loss'].item():.2e}, "
                    f"obs : {training_state['last_observation_loss'].item():.2e}, "
                    f"BC : {training_state['last_boundary_loss'].item():.2e})"
                )
                print(
                    "||grad(J)|| = "
                    f"{training_state['last_gradient_norm']:.2e}"
                )
                print(
                    "parameter    = "
                    f"{model.parameter_values().detach().clone()}"
                )
                print(
                    "time         = "
                    f"{time.time() - training_state['start_time']:.2f} s"
                )
            return total_loss

        def on_step_completed(before_evaluations, after_evaluations):
            _maybe_plot_training_state(
                before_evaluations,
                after_evaluations,
                training_state["plot_every"],
                model,
                collocation_points,
                observation_points,
                ref_solution,
            )

        training_state["number_of_steps"] += _run_lbfgs_with_evaluation_budget(
            optimizer,
            grads,
            max_iter,
            evaluation_counter=lambda: training_state[
                "number_of_evaluations"
            ],
            on_step_completed=on_step_completed,
        )
