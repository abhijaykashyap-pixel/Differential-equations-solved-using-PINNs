# Differential Equations Solved Using PINNs

This repo contains implementations of Physics-Informed Neural Networks (PINNs) to solve differential equations without traditional numerical solvers — the network learns the solution by minimizing the PDE residual directly, using automatic differentiation instead of mesh-based methods.

Four problems are solved here:

| Notebook | Equation | Type |
|---|---|---|
| `exponential.ipynb` | dy/dx = y | 1st order ODE |
| `poisson.ipynb` | 1D Poisson's equation | 2nd order ODE (displacement in simply supported beams) |
| `Heat equation.ipynb` | 1D Heat equation | PDE |
| `PINN2.ipynb` (`pinn-euler-beam` branch) | Euler-Bernoulli beam equation, EI·y⁗ = q | 4th order ODE |

## Euler-Bernoulli Beam (PINN2.ipynb)

Solves the deflection of a simply supported beam under a uniform load using a fully-connected network (2 hidden layers, 64 units, Tanh activation), trained on:
- **Physics loss** — PDE residual `d⁴y/dx⁴ - q/EI` evaluated at randomly sampled collocation points each epoch
- **Boundary loss** — enforces `y = 0` and bending moment `M = -EI·y'' = 0` at both ends

**Validation:** predictions are checked against the closed-form analytical solution
`y(x) = q/(24EI) · (x⁴ - 2Lx³ + L³x)`, giving a **relative L2 error of 0.77%** against ground truth — not just a low training loss, but a solution that matches the true physics.

## Setup

```bash
pip install torch numpy matplotlib
```

Each notebook is self-contained — open in Jupyter and run top to bottom.

## Background

PINNs (Raissi et al., 2019) embed the governing differential equation directly into the loss function via automatic differentiation, so the network is constrained to obey the physics rather than just fitting data points. This is mesh-free and works well for problems where you have the governing equation but limited or no labeled data — the tradeoff is training a network for each new configuration rather than a general-purpose numerical solver.

## Branches

- `main` — exponential, Poisson, and heat equation solvers
- `pinn-euler-beam` — Euler-Bernoulli beam solver with analytical validation
