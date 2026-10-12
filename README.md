# PINN Room Heating

This project models heat propagation in a 2D room using three approaches: a finite-difference solver, a standard neural network and a physics-informed neural network (PINN) implemented in JAX.

The PINN reconstructs the temperature field from nine noisy sensor measurements, combining data-driven learning with the underlying heat equation. The three approaches can be compared using the same simulation setup.

A full project report is available in [`documents/Project1_report.pdf`](documents/Project1_report.pdf).

## Run

Install the dependencies:

```bash
uv sync
```

Run the individual models with:

```bash
uv run python scripts/run_fdm.py
uv run python scripts/run_nn.py
uv run python scripts/run_pinn.py
```

Model and simulation parameters are defined in `config.yaml`.

## Authors

- Kristoffer Eide
- Elisa

TMA4320 – Introduction to Scientific Computing, Spring 2026
