import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from matplotlib.widgets import Slider

# --- CONFIGURATION GLOBALE ---
plt.close('all')
np.random.seed(42)

N = 5  # Taille de la matrice
A = np.random.randn(N, N) / np.sqrt(N)  # Matrice aléatoire normalisée
eigenvalues = eig(A)[0]  # Valeurs propres de A

K = 100  # Nombre de points sur la frontière
tol = 1e-6  # Tolérance pour Newton
epsilon_init = 0.1  # Valeur initiale de ε
max_iter = 100  # Itérations max pour Newton

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

# Initialisation des plots
pseudospectrum_plot = ax.scatter([], [], color='red', s=12, marker='o', alpha=0.7, label='Pseudospectre')
ev_plot = ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), 
                    color='blue', marker='x', s=80, label='Valeurs propres')

def compute_singular_triplet(z, A):
    """Calcule la plus petite valeur singulière et ses vecteurs."""
    B = z * np.eye(A.shape[0]) - A
    U, S, Vh = svd(B)
    sigma_min = S[-1]
    u_min = U[:, -1]
    v_min = Vh[-1, :].conj()  # Vecteur singulier droit (conjugué)
    return sigma_min, u_min, v_min

def compute_boundary_point(z0, d, epsilon, A):
    """Trouve un point sur ∂Λ_ε(A) via Newton."""
    theta = 0.0
    z = z0
    for _ in range(max_iter):
        sigma_min, u_min, v_min = compute_singular_triplet(z, A)
        h = sigma_min - epsilon
        if np.abs(h) < tol:
            break
        
        # Correction clé : d doit être un scalaire complexe, pas un array
        grad_g = np.vdot(v_min, u_min)  # Gradient ∇g(z) = v_min* u_min
        h_prime = np.real(np.conj(d) * grad_g)  # Dérivée directionnelle
        
        if np.abs(h_prime) < 1e-10:
            break
            
        theta_new = theta - h / h_prime
        z = z0 + theta_new * d
        theta = theta_new
    return z

def compute_pseudospectrum(A, epsilon, K):
    """Calcule K points sur ∂Λ_ε(A)."""
    # Point initial : valeur propre + direction aléatoire
    lambda0 = eigenvalues[0]
    d0 = np.exp(1j * np.random.uniform(0, 2*np.pi))  # Direction unitaire aléatoire
    
    z1 = compute_boundary_point(lambda0, d0, epsilon, A)
    points = [z1]
    
    for k in range(1, K):
        # Direction tangente (orthogonale au gradient)
        _, u_min, v_min = compute_singular_triplet(points[-1], A)
        grad_g = np.vdot(v_min, u_min)
        r_k = 1j * grad_g / np.abs(grad_g)  # Rotation de π/2
        
        # Pas adaptatif (empirique)
        tau_k = 0.1 * (1 + 0.1 * np.random.randn())
        z_pred = points[-1] + tau_k * r_k
        
        # Correction
        d_k = grad_g / np.abs(grad_g)  # Direction du gradient
        z_corrected = compute_boundary_point(z_pred, d_k, epsilon, A)
        points.append(z_corrected)
    
    return np.array(points)

def update_pseudospectre(epsilon):
    points = compute_pseudospectrum(A, epsilon, K)
    pseudospectrum_plot.set_offsets(np.column_stack((np.real(points), np.imag(points))))
    ax.set_title(f'Pseudospectre (ε={epsilon:.3f})')
    fig.canvas.draw_idle()

# Initialisation
update_pseudospectre(epsilon_init)

# Configuration des axes
margin = 1.5 * np.max(np.abs(eigenvalues))
ax.set_xlim(np.real(eigenvalues).min() - margin, np.real(eigenvalues).max() + margin)
ax.set_ylim(np.imag(eigenvalues).min() - margin, np.imag(eigenvalues).max() + margin)
ax.grid(True)
ax.legend()

# Slider interactif
slider_ax = plt.axes([0.3, 0.1, 0.4, 0.03])
epsilon_slider = Slider(
    ax=slider_ax,
    label='ε',
    valmin=0.01,
    valmax=1.0,
    valinit=epsilon_init,
    valstep=0.01,
    color='#5c7cfa'
)
epsilon_slider.on_changed(update_pseudospectre)

plt.show()