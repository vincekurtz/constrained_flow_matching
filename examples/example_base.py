import argparse
from pathlib import Path
from typing import Tuple

import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import training


class FlowExample:
    """Reusable scaffolding for 2-D flow-matching examples.

    Handles argument parsing, training, saving/loading, sample generation,
    and plotting. Each concrete example script supplies the dataset, model,
    and hyperparameters specific to that experiment.
    """

    def __init__(
        self,
        dataset,
        model,
        save_path: str,
        plot_lims: Tuple[float, float] = (-3.0, 3.0),
    ):
        self.dataset = dataset
        self.model = model
        self.save_path = Path(save_path)
        self.plot_lims = plot_lims

    @staticmethod
    def build_arg_parser(default_save_path: str) -> argparse.ArgumentParser:
        """Return a parser with --train, --generate, and --save-path flags."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--train",
            action="store_true",
            help="Train the flow-matching model and save the trained model.",
        )
        parser.add_argument(
            "--generate",
            action="store_true",
            help="Load the trained model and visualize generated samples.",
        )
        parser.add_argument(
            "--save-path",
            type=str,
            default=default_save_path,
        )
        return parser

    def train(
        self,
        num_epochs: int = 500,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 0,
        print_frequency: int = 10,
    ):
        """Train the model on the dataset and save the result to disk."""
        model, normalizer = training.train(
            dataset=self.dataset,
            model=self.model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            print_frequency=print_frequency,
        )
        print("Saving trained model and normalizer to", self.save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as f:
            cloudpickle.dump({"model": model, "normalizer": normalizer}, f)

    def generate(self, num_samples: int = 1000, dt: float = 0.001):
        """Load the saved model and plot generated samples."""
        print("Loading trained model and normalizer from", self.save_path)
        with open(self.save_path, "rb") as f:
            data = cloudpickle.load(f)
        model = data["model"]
        normalizer = data["normalizer"]

        # Hackily assume that the constraint manifold is the unit circle.
        # TODO: generalize to other constraint manifolds.
        def g(x):
            """The constraint manifold g(x) = 0."""
            x = normalizer.unnormalize(x)
            return jnp.sum(jnp.square(x), axis=-1) - 1.0

        print("Generating samples...")
        rng = jax.random.key(42)
        x = jax.random.normal(rng, (num_samples, 2))  # Initial Gaussian noise

        def _step_fn(x, t):
            """Single forward Euler step on the flow ODE xdot = v(x, t)."""
            t_reshaped = jnp.full((x.shape[0],), t)
            x_dot = model(x, t_reshaped)  # v(x, t) from the learned model

            g_x = g(x)[:, None]  # (batch_size, 1)
            grad_g = jax.vmap(jax.grad(g))(x)

            # Lagrange multiplier, analytical solution,
            #   λ = − [∇ g(x) ∇ g(x)']⁻¹∇g(x) v(x, t)
            # This projects all flows to be tangent to the constraint manifold.
            lmbda = -jnp.sum(grad_g * x_dot, axis=-1) / jnp.sum(
                grad_g**2, axis=-1
            )
            x_dot = x_dot + lmbda[:, None] * grad_g

            # quadratic penalty on g(x) to encourage trajectories to stay on the
            # constraint manifold
            penalty_strength = 10.0
            x_dot = x_dot - penalty_strength * g_x * grad_g

            x_next = x + dt * x_dot
            return x_next, x_next

        # Generate samples by integrating the flow ODE. This is a fast
        # (compiled, batched) JAX version of:
        #   for t in timesteps:
        #       x += dt * model(x, t)
        #       xs.append(x)
        timesteps = jnp.arange(0, 1.0, dt)
        x, xs = jax.lax.scan(_step_fn, x, timesteps)

        # Measure constraint violation
        g_x = g(x)
        print("Constraint violation: mean |g(x)| =", jnp.mean(jnp.abs(g_x)))
        print("Constraint violation: max |g(x)| =", jnp.max(jnp.abs(g_x)))
        print("Constraint violation: std |g(x)| =", jnp.std(jnp.abs(g_x)))

        # Push data samples back to the original scale
        x = normalizer.unnormalize(x)

        print("Plotting")
        lo, hi = self.plot_lims
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        ax[0].set_title("Training Data")
        x_train = jnp.array(self.dataset.data)
        ax[0].scatter(x_train[:, 0], x_train[:, 1], alpha=0.5)
        ax[0].grid()
        ax[0].set_aspect("equal")
        ax[0].set_xlim(lo, hi)
        ax[0].set_ylim(lo, hi)

        ax[1].set_title("Generated Samples")
        ax[1].scatter(x[:, 0], x[:, 1], alpha=0.5)
        ax[1].grid()
        ax[1].set_aspect("equal")

        ax[2].set_title("Flow Trajectories")
        xs = normalizer.unnormalize(xs)
        ax[2].scatter(
            xs[0, :, 0], xs[0, :, 1], alpha=0.5, label="Initial Noise"
        )
        ax[2].scatter(
            xs[-1, :, 0], xs[-1, :, 1], alpha=0.5, label="Final Samples"
        )
        ax[2].plot(xs[:, 0:50, 0], xs[:, 0:50, 1], "k--")
        ax[2].grid()
        ax[2].set_aspect("equal")
        ax[2].legend()

        plt.tight_layout()
        plt.show()

    def run(self, args, parser=None, **train_kwargs):
        """Dispatch to train/generate based on parsed CLI args.

        Any keyword arguments are forwarded to :meth:`train`, so callers can
        override num_epochs, batch_size, learning_rate, seed, and
        print_frequency.
        """
        if args.train:
            self.train(**train_kwargs)
        if args.generate:
            self.generate()
        if not args.train and not args.generate:
            if parser is not None:
                parser.print_help()
