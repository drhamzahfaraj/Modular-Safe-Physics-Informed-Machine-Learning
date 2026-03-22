#!/usr/bin/env python3
"""
Training Data Generation for Both Benchmarks
Generates 5000 LQR-excited trajectory samples per benchmark.
Data split: 80/20 train/validation, seed=42.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
import json, os

np.random.seed(42)
N_SAMPLES = 5000
TRAIN_FRAC = 0.80

# === PENDULUM ===
g, ell, m, Ts = 9.81, 0.5, 0.5, 0.02
A_p = np.array([[1.0, Ts], [Ts*g/ell, 1.0]])
B_p = np.array([[0.0], [Ts/(m*ell**2)]])
Q_p, R_p = np.diag([10.0, 1.0]), np.array([[0.1]])
P_p = solve_discrete_are(A_p, B_p, Q_p, R_p)
K_p = -np.linalg.solve(R_p + B_p.T@P_p@B_p, B_p.T@P_p@A_p)

def pend_true(x, u):
    return np.array([x[0]+Ts*x[1], x[1]+Ts*(g/ell*np.sin(x[0])+u/(m*ell**2))])

pend_data = {'x':[], 'u':[], 'x_next':[], 'residual':[]}
for i in range(N_SAMPLES):
    x = np.array([np.random.uniform(-2, 2), np.random.uniform(-6, 6)])
    u = (K_p @ x)[0] + 2.0*np.random.randn()  # LQR + exploration noise
    u = np.clip(u, -10, 10)
    x_next = pend_true(x, u) + 0.05*(2*np.random.rand(2)-1)
    nom_next = A_p @ x + B_p.flatten() * u
    pend_data['x'].append(x.tolist())
    pend_data['u'].append(float(u))
    pend_data['x_next'].append(x_next.tolist())
    pend_data['residual'].append((x_next - nom_next).tolist())

# === DC-DC ===
A_d = np.array([[0.971,-0.010],[1.732,0.970]])
B_d = np.array([[0.149],[0.181]])
x_eq = np.array([0.05, 5.0]); u_eq = 0.35
Q_d, R_d = np.diag([100.0, 10.0]), np.array([[1.0]])
P_d = solve_discrete_are(A_d, B_d, Q_d, R_d)
K_d = -np.linalg.solve(R_d + B_d.T@P_d@B_d, B_d.T@P_d@A_d)

dcdc_data = {'x':[], 'u':[], 'x_next':[], 'residual':[]}
for i in range(N_SAMPLES):
    x = x_eq + np.array([0.05*np.random.randn(), 1.0*np.random.randn()])
    u = np.clip((K_d@(x-x_eq))[0]+u_eq+0.1*np.random.randn(), 0, 1)
    # ±5% parameter variation
    A_var = A_d*(1+0.05*(2*np.random.rand(*A_d.shape)-1))
    x_next = A_var@x + B_d.flatten()*u + B_d.flatten()*0.1*(2*np.random.rand()-1)
    nom_next = A_d@x + B_d.flatten()*u
    dcdc_data['x'].append(x.tolist())
    dcdc_data['u'].append(float(u))
    dcdc_data['x_next'].append(x_next.tolist())
    dcdc_data['residual'].append((x_next - nom_next).tolist())

# Split
n_train = int(N_SAMPLES * TRAIN_FRAC)
for data, name in [(pend_data, 'pendulum'), (dcdc_data, 'dcdc')]:
    data['train_idx'] = list(range(n_train))
    data['val_idx'] = list(range(n_train, N_SAMPLES))
    data['metadata'] = {
        'n_samples': N_SAMPLES, 'train_frac': TRAIN_FRAC,
        'seed': 42, 'benchmark': name,
    }

if __name__ == '__main__':
    outdir = os.path.dirname(__file__)
    for data, name in [(pend_data, 'pendulum'), (dcdc_data, 'dcdc')]:
        path = os.path.join(outdir, f'{name}_training_data.json')
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Generated {name}: {N_SAMPLES} samples ({n_train} train, "
              f"{N_SAMPLES-n_train} val) -> {path}")
