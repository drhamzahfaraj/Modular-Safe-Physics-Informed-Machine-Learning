# Revision Log — Safe-PIML Paper

## v3 (March 2026) — JKSU-ES Submission
**Target:** Journal of King Saud University – Engineering Sciences

### Template & Formatting
- Converted from Elsevier `elsarticle` to Springer Nature compatible `article` class
- Removed line numbering for submission-ready version
- Author info: Hamzah Faraj, Taif University

### Peer Review Fixes Applied
| Review Item | Fix | Section |
|-------------|-----|---------|
| M1: Novelty | Added synergy articulation paragraph | Intro §1 |
| M2: Jacobian gap | Max observed norm on 10K grid: 0.48/0.46 < 0.50 | §3.1 |
| M3: Profiling | Clarified FLOP-based scaling methodology | §3.3 |
| M4: Fair HNN comparison | Added HNN+filter baseline row | Table 3 |
| M5: Averaged ablation | Split into per-benchmark Tables 5-6 | §5.3 |
| m1: ±5% justification | Component tolerance standard | §4.2 |
| m2: PPO details | Reward, steps, hyperparameters added | §4.1 |
| m4: Cost trade-off | Safety filter conservatism explanation | §5.1 |
| m5: Swing-up scope | Remark distinguishes regulation vs swing-up | Remark 1 |
| m6: Reference DOIs | Greydanus→NeurIPS URL, Rackauckas→arXiv noted | references.bib |

### New Content
- Section 5.5: Error bound instantiation (Table 8)
- Table 7: Merged scalability (data + σ̄ sensitivity + compute)
- Violation rate column in data scalability
- Memory footnote explaining 112 KB solver-only baseline
- 16 design justifications embedded throughout

### Redundancy Reduction
- Background: merged Verification into Control subsection
- Discussion: condensed from 5 paragraphs to 3 (analytical only)
- Conclusion: removed re-listing of ablation ratios
- Scalability: merged 3 tables into 1 compound table
- **Result:** 30 pages → 19 pages (single-spaced)

### Numeric Verification
- All 21 derived claims verified computationally
- Internal consistency: ablation w/o residual = PSF baseline ✓
- Error bound: bar_e = 0.072 (pend), 0.015 (DC-DC) ✓
- 0 undefined citations, 0 LaTeX errors, 0 table overflows

## v2 (March 2026) — Initial Revision
- Full revision from Major Revision feedback
- Corrected Proposition 1 error bound (added L_r)
- Corrected all memory values (117 KB deployed binary)
- Added formal Remark on nonlinear applicability
- All figures replaced with actual matplotlib PDFs

## v1 (March 2026) — Initial Draft
- Complete manuscript with placeholder figures
- 34 references, 9 tables, 5 figures, 9 equations
- Submitted for internal review
