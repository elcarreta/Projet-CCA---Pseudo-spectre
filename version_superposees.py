import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from matplotlib.widgets import Slider, CheckButtons

# --- CONFIGURATION ---
np.random.seed(42)
plt.close('all')

N = 5
A = np.random.randn(N, N) / np.sqrt(N)
eigenvalues = eig(A)[0]

# Paramètres Newton
K = 100
tol = 1e-6
epsilon_init = 0.1
max_iter = 100

# Paramètres GRID
m = 150
diagonal = np.diag(A)
row_sums = np.sum(np.abs(A), axis=1) - np.abs(diagonal)
gershgorin_centers = diagonal
gershgorin_radii = row_sums

center_real = np.real(gershgorin_centers)
center_imag = np.imag(gershgorin_centers)
radius_max = np.max(gershgorin_radii)

x_min, x_max = np.min(center_real - radius_max), np.max(center_real + radius_max)
y_min, y_max = np.min(center_imag - radius_max), np.max(center_imag + radius_max)

x_range = np.linspace(x_min, x_max, m)
y_range = np.linspace(y_min, y_max, m)
X, Y = np.meshgrid(x_range, y_range)
sigmin = np.zeros((m, m))

for j in range(m):
    for k in range(m):
        Z = (X[j, k] + 1j * Y[j, k]) * np.eye(N) - A
        singular_values = svd(Z, compute_uv=False)
        sigmin[j, k] = np.min(singular_values)

# --- FONCTIONS NEWTON ---
def compute_singular_triplet(z, A):
    B = z * np.eye(A.shape[0]) - A
    U, S, Vh = svd(B)
    return S[-1], U[:, -1], Vh[-1, :].conj()

def compute_boundary_point(z0, d, epsilon, A):
    theta = 0.0
    z = z0
    for _ in range(max_iter):
        sigma_min, u_min, v_min = compute_singular_triplet(z, A)
        h = sigma_min - epsilon
        if np.abs(h) < tol:
            break
        grad_g = np.vdot(v_min, u_min)
        h_prime = np.real(np.conj(d) * grad_g)
        if np.abs(h_prime) < 1e-10:
            break
        theta -= h / h_prime
        z = z0 + theta * d
    return z

def compute_pseudospectrum(A, epsilon, K):
    lambda0 = eigenvalues[0]
    d0 = np.exp(1j * np.random.uniform(0, 2*np.pi))
    z1 = compute_boundary_point(lambda0, d0, epsilon, A)
    points = [z1]
    for _ in range(1, K):
        _, u_min, v_min = compute_singular_triplet(points[-1], A)
        grad_g = np.vdot(v_min, u_min)
        r_k = 1j * grad_g / np.abs(grad_g)
        tau_k = 0.1 * (1 + 0.1 * np.random.randn())
        z_pred = points[-1] + tau_k * r_k
        d_k = grad_g / np.abs(grad_g)
        z_corrected = compute_boundary_point(z_pred, d_k, epsilon, A)
        points.append(z_corrected)
    return np.array(points)

# --- PLOT ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Courbe GRID (dans une liste)
contour_plot = [ax.contour(X, Y, sigmin, levels=[epsilon_init], colors='red', linewidths=2)]

# Courbe NEWTON
newton_pts = compute_pseudospectrum(A, epsilon_init, K)
newton_plot = ax.scatter(np.real(newton_pts), np.imag(newton_pts), color='lime', s=12, marker='o', label='Newton')

# Valeurs propres
ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='blue', marker='x', s=80, label='Valeurs propres')

# Cercles Gershgorin
for i in range(N):
    circle = plt.Circle((np.real(gershgorin_centers[i]), np.imag(gershgorin_centers[i])),
                        gershgorin_radii[i], color='white', fill=False, linestyle='dashed', alpha=0.4)
    ax.add_patch(circle)

ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title(f"Pseudospectre combiné (ε={epsilon_init:.3f})")
ax.grid(True)
ax.legend()

# Slider epsilon
slider_ax = plt.axes([0.3, 0.1, 0.4, 0.03])
epsilon_slider = Slider(slider_ax, 'ε', valmin=0.01, valmax=1.0, valinit=epsilon_init, valstep=0.01)

# Checkbuttons
check_ax = plt.axes([0.05, 0.5, 0.15, 0.1])
check = CheckButtons(check_ax, ['Newton', 'Grid'], [True, True])

def update(val):
    epsilon = epsilon_slider.val
    # Contour GRID
    for coll in contour_plot[0].collections:
        coll.remove()
    contour_plot[0] = ax.contour(X, Y, sigmin, levels=[epsilon], colors='red', linewidths=2)

    # Courbe Newton
    pts = compute_pseudospectrum(A, epsilon, K)
    newton_plot.set_offsets(np.column_stack((np.real(pts), np.imag(pts))))

    ax.set_title(f"Pseudospectre combiné (ε={epsilon:.3f})")
    fig.canvas.draw_idle()

epsilon_slider.on_changed(update)

def toggle_visibility(label):
    if label == 'Newton':
        visible = not newton_plot.get_visible()
        newton_plot.set_visible(visible)
    elif label == 'Grid':
        for coll in contour_plot[0].collections:
            coll.set_visible(not coll.get_visible())
    fig.canvas.draw_idle()

check.on_clicked(toggle_visibility)

plt.show()
