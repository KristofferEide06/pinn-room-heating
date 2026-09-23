# PINN Room Heating

This project models heating of a 2D room with the heat equation, solved three
ways: finite differences, a plain neural network, and a physics-informed neural
network (PINN) in JAX. The PINN reconstructs the temperature field from nine
noisy sensor measurements.

The solvers are run with `scripts/run_fdm.py`, `scripts/run_nn.py` and
`scripts/run_pinn.py`. All parameters are set in `config.yaml`.

## Authors
- Kristoffer Eide
- Elisa

TMA4320 - Introduction to Scientific Computing, Spring 2026
