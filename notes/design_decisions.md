# Design Decisions — Modular Safe-PIML

This document records the rationale behind every major design choice in the paper, organized by the section where the choice is introduced. JKSU-ES reviewers value explicit justifications for design decisions.

## Architecture Choices

### Residual network: 2×32 (1250 parameters)
- **Justification:** Scalability analysis (Table 7b) shows 2×32 achieves near-asymptotic violation performance (0.84%) while remaining within the 117 KB deployed binary budget on ARM Cortex-M7.
- **Alternative considered:** 3×64 (1090 params) achieves marginally better 1.08% but exceeds the binary budget and provides diminishing returns.

### OSQP solver for safety filter QP
- **Justification:** Operator-splitting method avoids matrix factorizations per iteration → predictable per-iteration cost + small code footprint (~60 KB compiled) suitable for microcontrollers.
- **Alternative considered:** Interior-point solvers (ECOS, SCS) require dense linear algebra and larger runtime footprint.

### First-input-only objective in safety filter QP (Eq. 5)
- **Justification:** Penalizing only ||u_{0|k} - u_prop|| reduces the QP to a low-dimensional problem, accelerating the solve. This follows the minimal-invasiveness principle of Wabersich & Zeilinger (2021).

## Hyperparameter Choices

### λ_phys = 0.01
- **Justification:** Small enough that data fidelity dominates early training (good fit), large enough to enforce Jacobian constraint at convergence. Follows curriculum-style weighting per Nghiem et al. (2023).

### σ̄ = 0.50
- **Justification:** Minimum value achieving near-optimal violation rate in sensitivity analysis (Table 7b). σ̄ = 0.10 restricts capacity (RMSE 0.062); σ̄ = 2.0 loosens error bound (violations 2.68%).

### MPC horizons: N=20 (pendulum), N=15 (converter)
- **Justification:** Selected to ensure terminal set reachability from all initial conditions within the constraint set, following Mayne et al. (2005).

### ±5% parameter variation (DC-DC)
- **Justification:** Standard tolerance class for film capacitors and molded inductors in buck converter circuits (Schwan et al., 2023).

## Benchmark Selection

### Inverted pendulum (2 states, nonlinear)
- **Rationale:** Well-understood sin(x)-x nonlinearity the residual must capture; original PSF results from Wabersich & Zeilinger (2021) provide directly comparable baseline.

### DC-DC buck converter (2 states, linear)
- **Rationale:** Complementary to pendulum — tests where tube MPC already achieves zero violations, demonstrating the method's value as reduced conservatism rather than safety improvement.
- **Known limitation:** Both benchmarks are low-dimensional. A 4+ state benchmark (cart-pole, quadrotor) would strengthen the contribution.

## PPO Performance Controller

- **Reward:** r_k = -||x||_Q^2 - ||u||_R^2, Q=diag(10,1), R=0.1
- **Training:** 2×10^6 steps, Stable-Baselines3, lr=3e-4, clip=0.2, minibatch=64
- **Rationale:** Intentionally aggressive (no constraint awareness) to stress-test the safety filter with frequent proposed violations.

## Latency Estimation

- **Method:** FLOP-based scaling from i7-12700H host to Cortex-M7 target using ~84 MFLOPS measured throughput.
- **Limitation:** Upper bound; does not account for cache effects. Not validated on physical hardware.
