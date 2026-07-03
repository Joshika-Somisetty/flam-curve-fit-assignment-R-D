# AI R&D Assignment — Parametric Curve Fitting

## Final Answer

| Parameter | Value |
|---|---|
| **θ** | 30° &nbsp;(= π/6 rad = 0.523599 rad) |
| **M** | 0.03 |
| **X** | 55 |

### Required LaTeX / Desmos string (per submission format)

```
\left(t*\cos(0.5235988)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235988)+55,42+t*\sin(0.5235988)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235988)\right)
```
with domain `6 ≤ t ≤ 60`.

Paste this into Desmos to verify it reproduces the given point cloud exactly (see `fit_overlay.png` for a static version of this check).

---

## Approach

### 1. Understanding the problem
The curve is:

```
x(t) = t·cos(θ) − e^(M|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)
```

`xy_data.csv` gives 1,500 (x, y) points sampled from this curve for `t ∈ [6, 60]`, **but the points are not labeled with their generating `t`, and are not in order**. So this isn't a simple point-wise regression (there's no direct `(t_i, x_i, y_i)` triple to fit against) — it's a genuine curve-fitting problem: find (θ, M, X) such that the *shape* of the candidate curve, swept over `t ∈ [6, 60]`, best explains the *cloud* of points.

### 2. Objective function
For a candidate (θ, M, X):
1. Densely sample the candidate curve over `t ∈ [6, 60]` (e.g. thousands of points), producing a fine polyline approximation of the curve.
2. For every data point, compute the distance to the **nearest point on that sampled curve** (a point-to-curve distance, found efficiently with a k-d tree).
3. Sum these distances across all 1,500 points. This is directly aligned with the assignment's own L1 evaluation metric (distance between sampled points on the expected vs. predicted curve).

Minimizing this sum over (θ, M, X) finds the curve that best "covers" the observed point cloud.

### 3. Optimization strategy
- **Global search:** `scipy.optimize.differential_evolution` over the given bounds (0° < θ < 50°, −0.05 < M < 0.05, 0 < X < 100), using a smooth squared-distance (L2) objective for stable convergence across the population.
- **Local polish:** `scipy.optimize.minimize` (Nelder-Mead) on the true L1 point-to-curve objective, starting from the global search result.
- **Resolution refinement:** repeat the local polish with a much finer curve sampling grid (4,000 → 40,000 points along `t`). This shrinks any error caused by only sampling the curve at finitely many points, isolating the *true* residual.

### 4. Result and validation
The optimizer converged to clean values: **θ = 30°, M = 0.03, X = 55**.

As the curve-sampling resolution increases, the residual point-to-curve distance shrinks roughly proportionally (mean distance ≈ 0.004 at 4,000 samples → ≈ 0.0004 at 40,000 samples). This is the signature of pure grid-discretization error, not genuine model mismatch — i.e., these are the *exact* underlying parameters used to generate the data, not just a close numerical approximation. A visual overlay of the fitted curve against the raw data cloud (`fit_overlay.png`) confirms the curve passes directly through the data with no visible offset.

---

## Files
- `fit_curve.py` — full fitting pipeline (global search + local polish + resolution refinement)
- `data/xy_data.csv` — provided dataset
- `fit_overlay.png` — visual validation: data points vs. fitted curve
- `requirements.txt` — Python dependencies

## Running it
```bash
pip install -r requirements.txt
python fit_curve.py
```
