# Modular Safe Physics-Informed Machine Learning for Dynamics and Control of Autonomous Robotic Systems

**Author:** Hamzah Faraj  
**Affiliation:** Department of Science and Technology, Ranyah College, Taif University, Taif 21944, Saudi Arabia  
**Contact:** f.hamzah@tu.edu.sa  

---

## Abstract

Physics-informed machine learning (PIML) has emerged as a promising approach for learning dynamical models and control policies that respect underlying physical laws, yet most existing methods either rely on monolithic neural architectures that are difficult to verify or lack explicit safety guarantees for constrained systems. We propose a modular Safe-PIML architecture that separates modeling, safety, and performance into distinct, independently analyzable components. The dynamics are represented as a nominal physics-based model augmented by a small constrained residual network enforcing non-negativity, monotonicity, and Lipschitz smoothness properties, while safety is ensured by an MPC-based safety filter. Through numerical simulations on two benchmarks—an inverted pendulum with a predictive safety filter and a DC–DC buck converter with robust tube MPC—the modular Safe-PIML controller achieves constraint violation rates of **0.84%** (pendulum, 61.5% reduction vs. PSF) and **0.08%** (converter, 98.3% reduction vs. unfiltered NN), within a **117 KB** deployed binary and **16.9 ms** projected latency on an ARM Cortex-M7 microcontroller.

## Key Results

| Benchmark | Violation Rate | Reduction vs. Best Baseline | Memory | Latency |
|-----------|---------------|----------------------------|--------|---------|
| Inverted Pendulum | 0.84% | 61.5% vs. PSF nominal | 117 KB | 16.9 ms |
| DC–DC Converter | 0.08% | 98.3% vs. unfiltered NN | 117 KB | 11.7 ms |

## Repository Structure

```
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── paper/
│   ├── main.tex                       # LaTeX manuscript
│   ├── references.bib                 # BibTeX references (33 entries)
│   └── figures/                       # Publication-quality figures (PDF)
│       ├── architecture.pdf           # Fig 1: Architecture block diagram
│       ├── pendulum_results.pdf       # Fig 2: Pendulum phase plane + box plots
│       ├── dcdc_results.pdf           # Fig 3: DC-DC voltage traces + bar chart
│       ├── ablation_radar.pdf         # Fig 4: Ablation radar charts
│       └── scalability.pdf            # Fig 5: Data/model/compute scalability
├── experiments/
│   ├── scripts/
│   │   ├── run_all.py                 # Master script: runs all experiments
│   │   ├── pendulum_benchmark.py      # Inverted pendulum MC simulation
│   │   ├── dcdc_benchmark.py          # DC-DC converter MC simulation
│   │   ├── ablation_study.py          # Ablation study (per-benchmark)
│   │   ├── scalability_analysis.py    # Data/model/compute scalability
│   │   ├── error_bound_analysis.py    # Proposition 1 numerical instantiation
│   │   └── generate_figures.py        # Regenerate all paper figures
│   ├── data/
│   │   └── generate_training_data.py  # Training data generation for both benchmarks
│   └── results/
│       ├── pendulum_results.json      # Raw MC results
│       ├── dcdc_results.json          # Raw MC results
│       ├── ablation_pendulum.json     # Per-benchmark ablation
│       ├── ablation_dcdc.json         # Per-benchmark ablation
│       ├── scalability.json           # All scalability data
│       └── error_bounds.json          # Proposition 1 instantiation
└── notes/
    ├── design_decisions.md            # Justifications for all design choices
    └── revision_log.md                # Change log across revisions
```

## Reproducing Results

```bash
pip install -r requirements.txt
cd experiments/scripts
python run_all.py          # Runs all experiments (~10 min)
python generate_figures.py  # Regenerates all paper figures
```

All experiments use `seed=42` for reproducibility. Results are saved as JSON in `experiments/results/`.


## Acknowledgements

The author acknowledges the Deanship of Graduate Studies and Scientific Research, Taif University, for funding this work.
