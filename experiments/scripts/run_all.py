#!/usr/bin/env python3
"""
Master Script — Run All Experiments
Reproduces all results in the paper in sequence.
Total runtime: ~10 minutes on Intel i7-12700H.
"""
import subprocess, sys, os, time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

experiments = [
    ('Training Data Generation', '../data/generate_training_data.py'),
    ('Pendulum Benchmark (Table 3)', 'pendulum_benchmark.py'),
    ('DC-DC Benchmark (Table 4)', 'dcdc_benchmark.py'),
    ('Ablation Study (Tables 5-6)', 'ablation_study.py'),
    ('Scalability Analysis (Table 7)', 'scalability_analysis.py'),
    ('Error Bound Instantiation (Table 8)', 'error_bound_analysis.py'),
    ('Figure Generation (Figs 1-5)', 'generate_figures.py'),
]

if __name__ == '__main__':
    print("="*70)
    print("SAFE-PIML: RUNNING ALL EXPERIMENTS")
    print("="*70)
    t0 = time.time()
    for name, script in experiments:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        path = os.path.join(SCRIPTS_DIR, script)
        result = subprocess.run([sys.executable, path],
                                capture_output=False, cwd=SCRIPTS_DIR)
        if result.returncode != 0:
            print(f"  !! FAILED: {script}")
            sys.exit(1)
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"Results: experiments/results/*.json")
    print(f"Figures: paper/figures/*.pdf")
    print(f"{'='*70}")
