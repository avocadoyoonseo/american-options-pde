# American Options PDE Solver

This project implements numerical methods for pricing American put options by solving the Black-Scholes linear complementarity problem (LCP) using finite difference methods.

## Features

- European option pricing with Crank-Nicolson (CN) scheme (baseline)
- American option pricing with Projected Successive Over-Relaxation (PSOR)
- Penalty method implementation as an alternative to PSOR
- Adaptive Mesh Refinement (AMR) near the exercise boundary
- Validation against Binomial (CRR) and Longstaff-Schwartz methods

## Project Structure

```
american-pde-cpp/
  include/              # Header files
    grid.hpp            # Spatial grid implementation
    tridiag.hpp         # Tridiagonal matrix solver
    cn.hpp              # Crank-Nicolson scheme
    psor.hpp            # Projected SOR solver
    penalty.hpp         # Penalty method implementation
    amr.hpp             # Adaptive mesh refinement
    utils.hpp           # Utility functions
  src/                  # Source files
    main_cn_psor.cpp    # Main program for CN + PSOR
    main_penalty.cpp    # Main program for penalty method
    main_amr.cpp        # Main program for AMR
  tests/                # Test files
    test_tridiag.cpp    # Test for tridiagonal solver
    test_psor.cpp       # Test for PSOR implementation
  data/                 # Output data
  Makefile              # Build system
```

## Building and Running

```bash
make                 # Build all executables
make test            # Run tests
make clean           # Clean build files
```

## Input Parameters

- S0: Initial stock price
- K: Strike price
- r: Risk-free rate
- q: Dividend yield
- σ: Volatility
- T: Time to maturity
- Grid parameters (number of time steps, spatial steps)

## Output

- Option prices for various strikes and maturities
- Early-exercise boundary S*(t)
- Convergence tables and error analysis

## Validation

The results are validated against:
- CRR binomial tree model
- Longstaff-Schwartz Monte Carlo method

Error metrics include basis points (bps) difference and convergence rates.
