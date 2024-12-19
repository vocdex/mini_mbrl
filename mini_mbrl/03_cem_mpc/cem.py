"""Cross-Entropy Method (CEM) for multivariate optimization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal


def objective_function(x1, x2, noise_scale=0.5):
    # Base objective
    base_value = -((x1**2 - 2 * x1 + 1) ** 2) - (x2**2 - 12 * x2 + 36) ** 2 + 6
    if isinstance(x1, np.ndarray):
        noise = np.random.normal(0, noise_scale, size=x1.shape)
    else:
        noise = np.random.normal(0, noise_scale)

    return base_value + noise


mu = np.array([0.0, 0.0])  # Mean vector
cov = np.array([[5.0, 0.0], [0.0, 5.0]])  # Covariance matrix (diagonal) for simplicity

MIN_VARIANCE = 0.1
NOISE_SCALE = 0

# Pre-calculate grid points for visualization
x1_range = np.linspace(-20, 20, 100)
x2_range = np.linspace(-20, 20, 100)
x1, x2 = np.meshgrid(x1_range, x2_range)
pos = np.dstack((x1, x2))

mus = []
covs = []

# Iterate CEM
for iteration in range(50):
    # Generate samples from the multivariate Gaussian
    samples = np.random.multivariate_normal(mu, cov, 100)

    # Evaluate samples using vectorized operation with noise
    scores = objective_function(samples[:, 0], samples[:, 1], noise_scale=NOISE_SCALE)

    # Select elite samples (top 10%)
    elite_idx = np.argsort(scores)[-10:]
    elite_samples = samples[elite_idx]

    # Update parameters
    mu = np.mean(elite_samples, axis=0)
    cov = np.cov(elite_samples, rowvar=False)

    # Ensure minimum variance
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, MIN_VARIANCE)
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    mus.append(mu)
    covs.append(cov)


# Plotting
fig, ax = plt.subplots(figsize=(10, 8))

Z = -((x1**2 - 2 * x1 + 1) ** 2) - (x2**2 - 2 * x2 + 1) ** 2 + 6
background = ax.contour(x1, x2, Z, levels=3, colors="gray", alpha=0.3)
contour = ax.contourf(x1, x2, np.zeros_like(x1), levels=30, cmap="viridis")
title = ax.set_title("Iteration 0", pad=15, y=1.02)

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(True, alpha=0.3)


def update(frame):
    # Clear previous contours
    for c in ax.collections:
        c.remove()

    # Plot background objective function
    background = ax.contour(x1, x2, Z, levels=20, colors="gray", alpha=0.3)

    # Create a multivariate Gaussian distribution
    rv = multivariate_normal(mean=mus[frame], cov=covs[frame])
    density = rv.pdf(pos)

    contour = ax.contourf(x1, x2, density, levels=30, cmap="viridis", alpha=0.7)

    title.set_text(f"Iteration {frame}")

    return ax.collections + [title]


ani = FuncAnimation(fig, update, frames=len(mus), interval=200, blit=False)

plt.show()
