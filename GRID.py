import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig
from matplotlib.widgets import Slider

# Fix random seed for reproducibility
np.random.seed(42)

# Define the matrix A
N = 5  # Matrix size
A = np.random.randn(N, N)  # Random example matrix

# Compute Gershgorin Circles
diagonal = np.diag(A)  # Extract diagonal elements
row_sums = np.sum(np.abs(A), axis=1) - np.abs(diagonal)  # Compute row sums excluding diagonal
gershgorin_centers = diagonal  # Centers of the circles
gershgorin_radii = row_sums  # Radii of the circles

# Compute eigenvalues of A
eigenvalues = eig(A)[0]  # Extract eigenvalues of A

# Define the bounding region from Gershgorin's theorem
center_real = np.real(gershgorin_centers)
center_imag = np.imag(gershgorin_centers)
radius_max = np.max(gershgorin_radii)

x_min, x_max = np.min(center_real - radius_max), np.max(center_real + radius_max)
y_min, y_max = np.min(center_imag - radius_max), np.max(center_imag + radius_max)

# Define grid based on Gershgorin circles
m = 100  # Number of grid points in each dimension
x_range = np.linspace(x_min, x_max, m)
y_range = np.linspace(y_min, y_max, m)

# Create grid
X, Y = np.meshgrid(x_range, y_range)
sigmin = np.zeros((m, m))

# Compute smallest singular values over the grid
for j in range(m):
    for k in range(m):
        Z = (X[j, k] + 1j * Y[j, k]) * np.eye(N) - A
        singular_values = svd(Z, compute_uv=False)  # Get all singular values
        sigmin[j, k] = min(singular_values)  # Smallest singular value

# Initialize figure and plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.2)  # Adjust space for slider
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title("Pseudospectrum with Gershgorin-Based Grid")

# Add a slider for epsilon selection
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])  # Position of slider
epsilon_slider = Slider(ax_slider, 'Epsilon', 10**-3, 10**0, valinit=10**-2, valstep=10**-3)

# Function to update plot when slider changes
def update(val):
    ax.clear()
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_title(f"Pseudospectrum with Epsilon = {epsilon_slider.val:.3f}")
    
    # Draw only the epsilon contour in red
    ax.contour(X, Y, sigmin, levels=[epsilon_slider.val], colors='red', linewidths=2)
    
    # Plot Gershgorin circles for reference
    for i in range(N):
        circle = plt.Circle((np.real(gershgorin_centers[i]), np.imag(gershgorin_centers[i])),
                            gershgorin_radii[i], color='white', fill=False, linestyle='dashed')
        ax.add_patch(circle)
    
    # Plot eigenvalues as small gray crosses
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='gray', marker='x', s=50, label="Eigenvalues")
    
    fig.canvas.draw_idle()

# Connect slider to update function
epsilon_slider.on_changed(update)

# Show interactive plot
plt.show()