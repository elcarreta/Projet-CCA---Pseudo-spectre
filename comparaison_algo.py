import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import svd, eig
import pandas as pd

# --- Fonctions utilitaires (identiques à celles précédemment définies) ---

def compute_singular_triplet(z, A):
    B = z * np.eye(A.shape[0]) - A
    U, S, Vh = svd(B)
    sigma_min = S[-1]
    u_min = U[:, -1]
    v_min = Vh[-1, :].conj()
    return sigma_min, u_min, v_min

def compute_boundary_point(z0, d, epsilon, A, max_iter=100, tol=1e-6):
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
        theta_new = theta - h / h_prime
        z = z0 + theta_new * d
        theta = theta_new
    return z

def compute_pseudospectrum_newton(A, epsilon, K=100):
    eigenvalues = eig(A)[0]
    lambda0 = eigenvalues[0]
    d0 = np.exp(1j * np.random.uniform(0, 2*np.pi))
    z1 = compute_boundary_point(lambda0, d0, epsilon, A)
    points = [z1]
    for k in range(1, K):
        _, u_min, v_min = compute_singular_triplet(points[-1], A)
        grad_g = np.vdot(v_min, u_min)
        r_k = 1j * grad_g / np.abs(grad_g)
        tau_k = 0.1 * (1 + 0.1 * np.random.randn())
        z_pred = points[-1] + tau_k * r_k
        d_k = grad_g / np.abs(grad_g)
        z_corrected = compute_boundary_point(z_pred, d_k, epsilon, A)
        points.append(z_corrected)
    return np.array(points)

def compute_pseudospectrum_grid(A, m=100):
    N = A.shape[0]
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
            sigmin[j, k] = min(singular_values)
    return sigmin

# --- Comparaison des temps ---

sizes = [5, 10, 15, 30, 60]
times_newton = []
times_grid = []

for N in sizes:
    A = np.random.randn(N, N) / np.sqrt(N)

    # Méthode NEWTON
    start = time.time()
    compute_pseudospectrum_newton(A, epsilon=0.1, K=100)
    times_newton.append(time.time() - start)

    # Méthode GRID
    start = time.time()
    compute_pseudospectrum_grid(A, m=100)
    times_grid.append(time.time() - start)

# --- Tableau résultats ---

df = pd.DataFrame({
    "Taille matrice": sizes,
    "Temps Newton (s)": times_newton,
    "Temps Grid (s)": times_grid
})
print(df)

# --- Graphique comparatif ---
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_newton, marker='o', label='Méthode Newton')
plt.plot(sizes, times_grid, marker='s', label='Méthode Grid')
plt.xlabel('Taille de la matrice (N)')
plt.ylabel('Temps d\'exécution (secondes)')
plt.title('Comparaison des temps d\'exécution : Newton vs Grid')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Ajout de nouvelles tailles de matrices pour la comparaison
sizes_extended = [5, 10, 15, 30, 60, 100, 200]
times_newton_ext = []
times_grid_ext = []

for N in sizes_extended:
    print(f"Traitement de N = {N}")
    A = np.random.randn(N, N) / np.sqrt(N)

    # Temps méthode NEWTON
    start = time.time()
    compute_pseudospectrum_newton(A, epsilon=0.1, K=100)
    times_newton_ext.append(time.time() - start)

    # Temps méthode GRID
    start = time.time()
    compute_pseudospectrum_grid(A, m=100)
    times_grid_ext.append(time.time() - start)

# Construction du tableau final
df_extended = pd.DataFrame({
    "Taille matrice": sizes_extended,
    "Temps Newton (s)": times_newton_ext,
    "Temps Grid (s)": times_grid_ext
})
print(df_extended)

# Graphique mis à jour
plt.figure(figsize=(10, 6))
plt.plot(sizes_extended, times_newton_ext, marker='o', label='Méthode Newton')
plt.plot(sizes_extended, times_grid_ext, marker='s', label='Méthode Grid')
plt.xlabel('Taille de la matrice (N)')
plt.ylabel('Temps d\'exécution (secondes)')
plt.title('Comparaison des temps d\'exécution : Newton vs Grid (étendu)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
