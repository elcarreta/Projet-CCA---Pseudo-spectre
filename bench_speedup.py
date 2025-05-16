import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from multiprocessing import Pool, cpu_count
import time

# ------------------- Matrix generation -------------------
def generate_matrix(N=5, seed=42):
    np.random.seed(seed)
    return np.random.randn(N, N) / np.sqrt(N)

# ------------------- GRID method -------------------
def compute_sigmin_at_point(point, A):
    x, y = point
    Z = (x + 1j * y) * np.eye(A.shape[0]) - A
    singular_values = svd(Z, compute_uv=False)
    return np.min(singular_values)

def benchmark_grid(A, m=100, cores=1):
    x = np.linspace(-2, 2, m)
    y = np.linspace(-2, 2, m)
    X, Y = np.meshgrid(x, y)
    points = [(X[j, k], Y[j, k]) for j in range(m) for k in range(m)]

    start = time.perf_counter()
    with Pool(processes=cores) as pool:
        _ = pool.starmap(compute_sigmin_at_point, [(pt, A) for pt in points])
    end = time.perf_counter()
    return end - start

# ------------------- Newton method -------------------
def compute_singular_triplet(z, A):
    B = z * np.eye(A.shape[0]) - A
    U, S, Vh = svd(B)
    return S[-1], U[:, -1], Vh[-1, :].conj()

def compute_boundary_point(z0, d, epsilon, A, tol=1e-6, max_iter=100):
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

def compute_pseudospectrum(A, epsilon, K=100):
    eigenvalues = eig(A)[0]
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
    return points

def run_single_newton(A, epsilon, K):
    compute_pseudospectrum(A, epsilon, K)
    return True

def benchmark_newton(A, epsilon=0.26, K=100, cores=1):
    start = time.perf_counter()
    with Pool(processes=cores) as pool:
        _ = pool.starmap(run_single_newton, [(A, epsilon, K)] * cores)
    end = time.perf_counter()
    return end - start

# ------------------- Run benchmark + speedup -------------------
def main():
    core_list = [1, 2, 4, 6, 8]
    core_list = [c for c in core_list if c <= cpu_count()]
    A = generate_matrix()

    grid_times = []
    newton_times = []

    print("Benchmark Grid et Newton (prédiction-correction)")
    for cores in core_list:
        print(f"→ Test avec {cores} cœur(s)...")
        t_grid = benchmark_grid(A, m=100, cores=cores)
        t_newton = benchmark_newton(A, epsilon=0.26, K=100, cores=cores)
        grid_times.append(t_grid)
        newton_times.append(t_newton)
        print(f"   Grid: {t_grid:.2f} s | Newton: {t_newton:.2f} s")

    # Tracer les courbes de temps
    plt.figure(figsize=(10, 6))
    plt.plot(core_list, grid_times, marker='o', label='Méthode Grid')
    plt.plot(core_list, newton_times, marker='s', label='Méthode Newton')
    plt.xlabel("Nombre de cœurs")
    plt.ylabel("Temps (secondes)")
    plt.title("Temps d'exécution : Grid vs Newton")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tracer les courbes de speedup
    print("→ Calcul et affichage des courbes de speedup...")
    baseline_grid = grid_times[0] if grid_times[0] != 0 else 1e-6
    baseline_newton = newton_times[0] if newton_times[0] != 0 else 1e-6

    speedup_grid = [baseline_grid / t for t in grid_times]
    speedup_newton = [baseline_newton / t for t in newton_times]

    plt.figure(figsize=(10, 6))
    plt.plot(core_list, speedup_grid, marker='o', label='Speedup Grid')
    plt.plot(core_list, speedup_newton, marker='s', label='Speedup Newton')
    plt.plot(core_list, core_list, 'k--', label='Speedup idéal')
    plt.xlabel("Nombre de cœurs")
    plt.ylabel("Speedup")
    plt.title("Speedup : Grid vs Newton")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

